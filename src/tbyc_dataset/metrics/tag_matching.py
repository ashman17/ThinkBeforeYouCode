from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import ensure_directory, read_json, write_json


def compute_tag_matching_metrics(
    *,
    owner: str,
    repo: str,
    output_root: str,
    model_id: str,
    issue_number: Optional[int] = None,
) -> Dict[str, Any]:
    repo_ref = RepositoryRef(owner=owner, name=repo)
    root = Path(output_root)
    model_dir = _model_dir_name(model_id)
    extracted_dir = root / "extractions" / repo_ref.fs_slug
    derived_dir = root / "derived" / model_dir / repo_ref.fs_slug
    metrics_dir = root / "metrics" / model_dir / repo_ref.fs_slug
    ensure_directory(metrics_dir)

    extracted = _load_issue_type_details(extracted_dir)
    derived = _load_issue_type_details(derived_dir)

    issue_numbers = sorted(set(extracted.keys()) | set(derived.keys()))
    if issue_number is not None:
        issue_numbers = [number for number in issue_numbers if number == issue_number]

    per_issue: List[Dict[str, Any]] = []
    overall_acc = _TagAccumulator()
    for number in issue_numbers:
        issue_result = _score_issue(number, extracted.get(number, {}), derived.get(number, {}))
        per_issue.append(issue_result)
        overall_acc.merge(_TagAccumulator.from_mapping(issue_result["pooled"]))

    per_type = _score_per_type(issue_numbers=issue_numbers, extracted=extracted, derived=derived)
    macro = _macro_average([item.get("pooled", {}) for item in per_issue])

    report: Dict[str, Any] = {
        "repository": repo_ref.slug,
        "model_id": model_id,
        "metric": "matching_type_tag_overlap",
        "issue_count": len(per_issue),
        "issue_number_filter": issue_number,
        "comparison_rule": "compare tags only for types present in both human and llm for the issue",
        "overall": overall_acc.to_metrics(),
        "macro_average": macro,
        "per_type": per_type,
        "per_issue": per_issue,
    }
    write_json(metrics_dir / "tag_matching.json", report)
    return report


def _score_issue(
    issue_number: int,
    human_types: Mapping[str, Mapping[str, Any]],
    llm_types: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    matching_types = sorted(set(human_types.keys()) & set(llm_types.keys()))
    compared: List[Dict[str, Any]] = []
    issue_acc = _TagAccumulator()
    for type_name in matching_types:
        human_tags = set(human_types.get(type_name, {}).get("tags", set()))
        llm_tags = set(llm_types.get(type_name, {}).get("tags", set()))
        type_metrics = _set_metrics(human_tags, llm_tags)
        compared.append(
            {
                "type": type_name,
                "human_summary_count": int(human_types.get(type_name, {}).get("summary_count", 0)),
                "llm_summary_count": int(llm_types.get(type_name, {}).get("summary_count", 0)),
                "human_tags": sorted(human_tags),
                "llm_tags": sorted(llm_tags),
                **type_metrics,
            }
        )
        issue_acc.merge(_TagAccumulator.from_mapping(type_metrics))

    return {
        "issue_number": issue_number,
        "human_type_count": len(human_types),
        "llm_type_count": len(llm_types),
        "matching_type_count": len(matching_types),
        "pooled": issue_acc.to_metrics(),
        "per_matching_type": compared,
    }


def _score_per_type(
    *,
    issue_numbers: Sequence[int],
    extracted: Mapping[int, Mapping[str, Mapping[str, Any]]],
    derived: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    all_types: Set[str] = set()
    for number in issue_numbers:
        all_types.update(extracted.get(number, {}).keys())
        all_types.update(derived.get(number, {}).keys())

    per_type: List[Dict[str, Any]] = []
    for type_name in sorted(all_types):
        type_acc = _TagAccumulator()
        compared_issue_count = 0
        human_issue_count = 0
        llm_issue_count = 0
        for number in issue_numbers:
            human_has = type_name in extracted.get(number, {})
            llm_has = type_name in derived.get(number, {})
            if human_has:
                human_issue_count += 1
            if llm_has:
                llm_issue_count += 1
            if not (human_has and llm_has):
                continue
            compared_issue_count += 1

            human_tags = set(extracted.get(number, {}).get(type_name, {}).get("tags", set()))
            llm_tags = set(derived.get(number, {}).get(type_name, {}).get("tags", set()))
            metrics = _set_metrics(human_tags, llm_tags)
            type_acc.merge(_TagAccumulator.from_mapping(metrics))

        per_type.append(
            {
                "type": type_name,
                "issue_count": len(issue_numbers),
                "human_issue_count": human_issue_count,
                "llm_issue_count": llm_issue_count,
                "compared_issue_count": compared_issue_count,
                **type_acc.to_metrics(),
            }
        )
    return per_type


def _load_issue_type_details(base_dir: Path) -> Dict[int, Dict[str, Dict[str, Any]]]:
    mapping: Dict[int, Dict[str, Dict[str, Any]]] = {}
    if not base_dir.exists():
        return mapping

    for path in sorted(base_dir.glob("issue_*.json")):
        payload = read_json(path)
        raw_issue_number = payload.get("issue", {}).get("issue_number") or payload.get("issue_number")
        if raw_issue_number is None:
            stem = path.stem
            if not stem.startswith("issue_"):
                continue
            raw_issue_number = stem.split("_", 1)[1]
        try:
            issue_number = int(raw_issue_number)
        except (TypeError, ValueError):
            continue

        mapping[issue_number] = _extract_type_details_from_payload(payload)
    return mapping


def _extract_type_details_from_payload(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    issue = payload.get("issue")
    if not isinstance(issue, dict):
        return {}
    comments = issue.get("comments")
    if not isinstance(comments, list):
        return {}

    details: Dict[str, Dict[str, Any]] = {}
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        artifacts = comment.get("artifacts")
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            type_name = artifact.get("type")
            if not isinstance(type_name, str) or not type_name.strip():
                continue

            bucket = details.setdefault(type_name.strip(), {"tags": set(), "summary_count": 0})
            summary = artifact.get("summary")
            if isinstance(summary, str) and summary.strip():
                bucket["summary_count"] = int(bucket.get("summary_count", 0)) + 1

            tags = artifact.get("tags")
            if not isinstance(tags, list):
                continue
            tag_bucket = bucket.setdefault("tags", set())
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    tag_bucket.add(tag.strip())
    return details


def _set_metrics(human_tags: Set[str], llm_tags: Set[str]) -> Dict[str, Any]:
    intersection = human_tags & llm_tags
    union = human_tags | llm_tags

    precision = _ratio(len(intersection), len(llm_tags))
    recall = _ratio(len(intersection), len(human_tags))
    f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
    jaccard = _ratio(len(intersection), len(union))

    return {
        "human_tag_count": len(human_tags),
        "llm_tag_count": len(llm_tags),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator) / float(denominator)


def _macro_average(items: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not items:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
        }
    return {
        "precision": _mean(float(item.get("precision", 0.0)) for item in items),
        "recall": _mean(float(item.get("recall", 0.0)) for item in items),
        "f1": _mean(float(item.get("f1", 0.0)) for item in items),
        "jaccard": _mean(float(item.get("jaccard", 0.0)) for item in items),
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def _model_dir_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__") or "unknown-model"


@dataclass
class _TagAccumulator:
    human_tag_count: int = 0
    llm_tag_count: int = 0
    intersection_count: int = 0
    union_count: int = 0

    def merge(self, other: "_TagAccumulator") -> None:
        self.human_tag_count += other.human_tag_count
        self.llm_tag_count += other.llm_tag_count
        self.intersection_count += other.intersection_count
        self.union_count += other.union_count

    def to_metrics(self) -> Dict[str, Any]:
        precision = _ratio(self.intersection_count, self.llm_tag_count)
        recall = _ratio(self.intersection_count, self.human_tag_count)
        f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
        jaccard = _ratio(self.intersection_count, self.union_count)
        return {
            "human_tag_count": self.human_tag_count,
            "llm_tag_count": self.llm_tag_count,
            "intersection_count": self.intersection_count,
            "union_count": self.union_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
        }

    @staticmethod
    def from_mapping(values: Mapping[str, Any]) -> "_TagAccumulator":
        return _TagAccumulator(
            human_tag_count=int(values.get("human_tag_count", 0)),
            llm_tag_count=int(values.get("llm_tag_count", 0)),
            intersection_count=int(values.get("intersection_count", 0)),
            union_count=int(values.get("union_count", 0)),
        )
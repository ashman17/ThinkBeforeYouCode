from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import ensure_directory, read_json, write_json


def compute_type_matching_metrics(
    *,
    owner: str,
    repo: str,
    output_root: str,
    model_id: str,
    issue_number: Optional[int] = None,
    derived_root_dirname: str = "derived",
    metrics_root_dirname: str = "metrics",
) -> Dict[str, Any]:
    repo_ref = RepositoryRef(owner=owner, name=repo)
    root = Path(output_root)
    model_dir = _model_dir_name(model_id)
    extracted_dir = root / "extractions" / repo_ref.fs_slug
    derived_dir = root / derived_root_dirname / model_dir / repo_ref.fs_slug
    metrics_dir = root / metrics_root_dirname / model_dir / repo_ref.fs_slug
    ensure_directory(metrics_dir)

    extracted = _load_issue_type_sets(extracted_dir)
    derived = _load_issue_type_sets(derived_dir)

    issue_numbers = sorted(set(extracted.keys()) | set(derived.keys()))
    if issue_number is not None:
        issue_numbers = [number for number in issue_numbers if number == issue_number]

    per_issue: List[Dict[str, Any]] = []
    for number in issue_numbers:
        human_types = extracted.get(number, set())
        llm_types = derived.get(number, set())
        metrics = _set_metrics(human_types, llm_types)
        per_issue.append(
            {
                "issue_number": number,
                "human_types": sorted(human_types),
                "llm_types": sorted(llm_types),
                **metrics,
            }
        )

    macro = _macro_average(per_issue)
    global_human_types = set().union(*(extracted.get(number, set()) for number in issue_numbers)) if issue_numbers else set()
    global_llm_types = set().union(*(derived.get(number, set()) for number in issue_numbers)) if issue_numbers else set()
    overall = _set_metrics(global_human_types, global_llm_types)
    per_type = _per_type_metrics(issue_numbers=issue_numbers, extracted=extracted, derived=derived)

    report: Dict[str, Any] = {
        "repository": repo_ref.slug,
        "model_id": model_id,
        "metric": "aggregated_type_matching",
        "issue_count": len(per_issue),
        "issue_number_filter": issue_number,
        "overall": {
            "human_types": sorted(global_human_types),
            "llm_types": sorted(global_llm_types),
            **overall,
        },
        "macro_average": macro,
        "per_type": per_type,
        "per_issue": per_issue,
    }
    write_json(metrics_dir / "type_matching.json", report)
    return report


def _load_issue_type_sets(base_dir: Path) -> Dict[int, Set[str]]:
    mapping: Dict[int, Set[str]] = {}
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

        mapping[issue_number] = _extract_types_from_payload(payload)
    return mapping


def _extract_types_from_payload(payload: Mapping[str, Any]) -> Set[str]:
    issue = payload.get("issue")
    if not isinstance(issue, dict):
        return set()
    comments = issue.get("comments")
    if not isinstance(comments, list):
        return set()

    types: Set[str] = set()
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        artifacts = comment.get("artifacts")
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            artifact_type = artifact.get("type")
            if isinstance(artifact_type, str) and artifact_type.strip():
                types.add(artifact_type.strip())
    return types


def _set_metrics(human_types: Set[str], llm_types: Set[str]) -> Dict[str, Any]:
    intersection = human_types & llm_types
    union = human_types | llm_types

    precision = _ratio(len(intersection), len(llm_types))
    recall = _ratio(len(intersection), len(human_types))
    f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
    jaccard = _ratio(len(intersection), len(union))

    return {
        "human_type_count": len(human_types),
        "llm_type_count": len(llm_types),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


def _per_type_metrics(
    *,
    issue_numbers: Sequence[int],
    extracted: Mapping[int, Set[str]],
    derived: Mapping[int, Set[str]],
) -> List[Dict[str, Any]]:
    all_types = sorted(
        set().union(*(extracted.get(number, set()) for number in issue_numbers))
        | set().union(*(derived.get(number, set()) for number in issue_numbers))
    ) if issue_numbers else []

    results: List[Dict[str, Any]] = []
    for type_name in all_types:
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for number in issue_numbers:
            human_has = type_name in extracted.get(number, set())
            llm_has = type_name in derived.get(number, set())
            if human_has and llm_has:
                tp += 1
            elif (not human_has) and llm_has:
                fp += 1
            elif human_has and (not llm_has):
                fn += 1
            else:
                tn += 1

        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
        jaccard = _ratio(tp, tp + fp + fn)

        results.append(
            {
                "type": type_name,
                "issue_count": len(issue_numbers),
                "human_issue_count": tp + fn,
                "llm_issue_count": tp + fp,
                "true_positive_count": tp,
                "false_positive_count": fp,
                "false_negative_count": fn,
                "true_negative_count": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard": jaccard,
            }
        )
    return results


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator) / float(denominator)


def _macro_average(per_issue: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not per_issue:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
        }

    return {
        "precision": _mean(float(item.get("precision", 0.0)) for item in per_issue),
        "recall": _mean(float(item.get("recall", 0.0)) for item in per_issue),
        "f1": _mean(float(item.get("f1", 0.0)) for item in per_issue),
        "jaccard": _mean(float(item.get("jaccard", 0.0)) for item in per_issue),
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def _model_dir_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__") or "unknown-model"

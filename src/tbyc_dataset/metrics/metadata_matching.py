from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import ensure_directory, read_json, write_json


SIMILARITY_METRICS: Tuple[str, ...] = (
    "max_all",
    "token_f1",
    "sequence_ratio",
    "token_jaccard",
    "char_3gram_jaccard",
    "token_containment",
)


def compute_metadata_matching_metrics(
    *,
    owner: str,
    repo: str,
    output_root: str,
    model_id: str,
    issue_number: Optional[int] = None,
    similarity_threshold: float = 0.82,
    similarity_metric: str = "max_all",
) -> Dict[str, Any]:
    if similarity_metric == "all":
        return compute_metadata_matching_metrics_all(
            owner=owner,
            repo=repo,
            output_root=output_root,
            model_id=model_id,
            issue_number=issue_number,
            similarity_threshold=similarity_threshold,
        )

    if similarity_metric not in SIMILARITY_METRICS:
        raise ValueError(
            f"Unsupported similarity_metric '{similarity_metric}'. "
            f"Expected one of: {', '.join(SIMILARITY_METRICS)}"
        )

    return _compute_metadata_matching_single(
        owner=owner,
        repo=repo,
        output_root=output_root,
        model_id=model_id,
        issue_number=issue_number,
        similarity_threshold=similarity_threshold,
        similarity_metric=similarity_metric,
    )


def compute_metadata_matching_metrics_all(
    *,
    owner: str,
    repo: str,
    output_root: str,
    model_id: str,
    issue_number: Optional[int] = None,
    similarity_threshold: float = 0.82,
) -> Dict[str, Any]:
    reports_by_metric: Dict[str, Dict[str, Any]] = {}
    for metric in SIMILARITY_METRICS:
        reports_by_metric[metric] = _compute_metadata_matching_single(
            owner=owner,
            repo=repo,
            output_root=output_root,
            model_id=model_id,
            issue_number=issue_number,
            similarity_threshold=similarity_threshold,
            similarity_metric=metric,
            write_single_report=False,
        )

    repo_ref = RepositoryRef(owner=owner, name=repo)
    root = Path(output_root)
    model_dir = _model_dir_name(model_id)
    metrics_dir = root / "metrics" / model_dir / repo_ref.fs_slug
    ensure_directory(metrics_dir)

    issue_count = 0
    if reports_by_metric:
        first_metric = next(iter(reports_by_metric.keys()))
        issue_count = int(reports_by_metric[first_metric].get("issue_count", 0))

    summary_by_metric: Dict[str, Dict[str, Any]] = {}
    for metric_name, report in reports_by_metric.items():
        summary_by_metric[metric_name] = {
            "overall": report.get("overall", {}).get("pooled", {}),
            "macro_average": report.get("macro_average", {}),
        }

    combined: Dict[str, Any] = {
        "repository": repo_ref.slug,
        "model_id": model_id,
        "metric": "metadata_phrase_matching_all",
        "issue_count": issue_count,
        "issue_number_filter": issue_number,
        "similarity": {
            "method": "all",
            "methods": list(SIMILARITY_METRICS),
            "components": [
                "token_f1",
                "sequence_ratio",
                "token_jaccard",
                "char_3gram_jaccard",
                "token_containment",
            ],
            "threshold": similarity_threshold,
        },
        "summary_by_metric": summary_by_metric,
        "reports_by_metric": reports_by_metric,
    }
    write_json(metrics_dir / "metadata_matching_all.json", combined)
    return combined


def _compute_metadata_matching_single(
    *,
    owner: str,
    repo: str,
    output_root: str,
    model_id: str,
    issue_number: Optional[int],
    similarity_threshold: float,
    similarity_metric: str,
    write_single_report: bool = True,
) -> Dict[str, Any]:
    repo_ref = RepositoryRef(owner=owner, name=repo)
    root = Path(output_root)
    model_dir = _model_dir_name(model_id)
    extracted_dir = root / "extractions" / repo_ref.fs_slug
    derived_dir = root / "derived" / model_dir / repo_ref.fs_slug
    metrics_dir = root / "metrics" / model_dir / repo_ref.fs_slug
    ensure_directory(metrics_dir)

    extracted = _load_issue_type_field_values(extracted_dir)
    derived = _load_issue_type_field_values(derived_dir)

    issue_numbers = sorted(set(extracted.keys()) | set(derived.keys()))
    if issue_number is not None:
        issue_numbers = [number for number in issue_numbers if number == issue_number]

    per_issue: List[Dict[str, Any]] = []
    for number in issue_numbers:
        human_map = extracted.get(number, {})
        llm_map = derived.get(number, {})
        issue_report = _score_issue(
            issue_number=number,
            human_map=human_map,
            llm_map=llm_map,
            similarity_threshold=similarity_threshold,
            similarity_metric=similarity_metric,
        )
        per_issue.append(issue_report)

    macro = _macro_average([item.get("pooled", {}) for item in per_issue])
    overall = _score_union_across_issues(
        issue_numbers=issue_numbers,
        extracted=extracted,
        derived=derived,
        similarity_threshold=similarity_threshold,
        similarity_metric=similarity_metric,
    )

    report: Dict[str, Any] = {
        "repository": repo_ref.slug,
        "model_id": model_id,
        "metric": "metadata_phrase_matching",
        "issue_count": len(per_issue),
        "issue_number_filter": issue_number,
        "similarity": {
            "method": similarity_metric,
            "components": [
                "token_f1",
                "sequence_ratio",
                "token_jaccard",
                "char_3gram_jaccard",
                "token_containment",
            ],
            "threshold": similarity_threshold,
        },
        "overall": overall,
        "macro_average": macro,
        "per_issue": per_issue,
    }
    if write_single_report:
        write_json(metrics_dir / "metadata_matching.json", report)
    return report


def _score_union_across_issues(
    *,
    issue_numbers: Sequence[int],
    extracted: Mapping[int, Mapping[str, Mapping[str, Sequence[str]]]],
    derived: Mapping[int, Mapping[str, Mapping[str, Sequence[str]]]],
    similarity_threshold: float,
    similarity_metric: str,
) -> Dict[str, Any]:
    union_human: Dict[str, Dict[str, List[str]]] = {}
    union_llm: Dict[str, Dict[str, List[str]]] = {}

    for issue in issue_numbers:
        _merge_type_field_values(union_human, extracted.get(issue, {}))
        _merge_type_field_values(union_llm, derived.get(issue, {}))

    overall_payload = _score_type_maps(
        union_human,
        union_llm,
        similarity_threshold,
        similarity_metric,
    )
    return {
        "type_count": len(overall_payload["per_type"]),
        "pooled": overall_payload["pooled"],
        "per_type": overall_payload["per_type"],
    }


def _score_issue(
    *,
    issue_number: int,
    human_map: Mapping[str, Mapping[str, Sequence[str]]],
    llm_map: Mapping[str, Mapping[str, Sequence[str]]],
    similarity_threshold: float,
    similarity_metric: str,
) -> Dict[str, Any]:
    payload = _score_type_maps(
        human_map,
        llm_map,
        similarity_threshold,
        similarity_metric,
    )
    return {
        "issue_number": issue_number,
        "type_count": len(payload["per_type"]),
        "pooled": payload["pooled"],
        "per_type": payload["per_type"],
    }


def _score_type_maps(
    human_map: Mapping[str, Mapping[str, Sequence[str]]],
    llm_map: Mapping[str, Mapping[str, Sequence[str]]],
    similarity_threshold: float,
    similarity_metric: str,
) -> Dict[str, Any]:
    type_names = sorted(set(human_map.keys()) | set(llm_map.keys()))
    type_reports: List[Dict[str, Any]] = []
    aggregate = _Accumulator()

    for type_name in type_names:
        human_fields = human_map.get(type_name, {})
        llm_fields = llm_map.get(type_name, {})
        field_names = sorted(set(human_fields.keys()) | set(llm_fields.keys()))

        field_reports: List[Dict[str, Any]] = []
        type_acc = _Accumulator()
        for field_name in field_names:
            human_values = [value for value in human_fields.get(field_name, []) if value]
            llm_values = [value for value in llm_fields.get(field_name, []) if value]
            score = _score_values(
                human_values,
                llm_values,
                similarity_threshold,
                similarity_metric,
            )
            field_reports.append(
                {
                    "field": field_name,
                    **score.as_dict(),
                }
            )
            type_acc.add(score)

        type_report = {
            "type": type_name,
            "field_count": len(field_reports),
            "pooled": type_acc.to_metrics(),
            "fields": field_reports,
        }
        type_reports.append(type_report)
        aggregate.merge(type_acc)

    return {
        "pooled": aggregate.to_metrics(),
        "per_type": type_reports,
    }


def _score_values(
    human_values: Sequence[str],
    llm_values: Sequence[str],
    threshold: float,
    similarity_metric: str,
) -> "_ValueScore":
    human_count = len(human_values)
    llm_count = len(llm_values)

    tp = 0
    fp = llm_count
    fn = human_count

    if human_values and llm_values:
        pairs = _greedy_pairs(human_values, llm_values, similarity_metric)
        tp = sum(1 for _, _, similarity in pairs if similarity >= threshold)
        fp = llm_count - tp
        fn = human_count - tp

    precision = _ratio(tp, llm_count)
    recall = _ratio(tp, human_count)
    f1 = _harmonic(precision, recall)
    jaccard = _ratio(tp, tp + fp + fn)

    sum_best_llm = sum(_best_similarity(value, human_values, similarity_metric) for value in llm_values)
    sum_best_human = sum(_best_similarity(value, llm_values, similarity_metric) for value in human_values)
    soft_precision = 1.0 if llm_count == 0 else sum_best_llm / float(llm_count)
    soft_recall = 1.0 if human_count == 0 else sum_best_human / float(human_count)
    soft_f1 = _harmonic(soft_precision, soft_recall)

    return _ValueScore(
        human_count=human_count,
        llm_count=llm_count,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        sum_best_llm=sum_best_llm,
        sum_best_human=sum_best_human,
        soft_precision=soft_precision,
        soft_recall=soft_recall,
        soft_f1=soft_f1,
    )


def _greedy_pairs(
    human_values: Sequence[str],
    llm_values: Sequence[str],
    similarity_metric: str,
) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for human_index, human_value in enumerate(human_values):
        for llm_index, llm_value in enumerate(llm_values):
            similarity = _phrase_similarity(human_value, llm_value, similarity_metric)
            candidates.append((similarity, human_index, llm_index))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_human: Set[int] = set()
    used_llm: Set[int] = set()
    pairs: List[Tuple[int, int, float]] = []

    for similarity, human_index, llm_index in candidates:
        if human_index in used_human or llm_index in used_llm:
            continue
        used_human.add(human_index)
        used_llm.add(llm_index)
        pairs.append((human_index, llm_index, similarity))
    return pairs


def _best_similarity(source: str, candidates: Sequence[str], similarity_metric: str) -> float:
    if not candidates:
        return 0.0
    return max(_phrase_similarity(source, candidate, similarity_metric) for candidate in candidates)


def _phrase_similarity(left: str, right: str, similarity_metric: str) -> float:
    left_norm = _normalize_phrase(left)
    right_norm = _normalize_phrase(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0

    token_score = _token_f1(left_norm, right_norm)
    sequence_score = SequenceMatcher(None, left_norm, right_norm).ratio()
    token_jaccard = _token_jaccard(left_norm, right_norm)
    char_jaccard = _char_ngram_jaccard(left_norm, right_norm, n=3)
    containment = _token_containment(left_norm, right_norm)

    if similarity_metric == "token_f1":
        return token_score
    if similarity_metric == "sequence_ratio":
        return sequence_score
    if similarity_metric == "token_jaccard":
        return token_jaccard
    if similarity_metric == "char_3gram_jaccard":
        return char_jaccard
    if similarity_metric == "token_containment":
        return containment

    # Default: robust ensemble for short phrase matching.
    return max(token_score, sequence_score, token_jaccard, char_jaccard, containment)


def _token_f1(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    precision = float(overlap) / float(len(right_tokens))
    recall = float(overlap) / float(len(left_tokens))
    return _harmonic(precision, recall)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return float(len(left_tokens & right_tokens)) / float(len(left_tokens | right_tokens))


def _token_containment(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return float(overlap) / float(min(len(left_tokens), len(right_tokens)))


def _char_ngram_jaccard(left: str, right: str, n: int = 3) -> float:
    left_grams = _char_ngrams(left, n)
    right_grams = _char_ngrams(right, n)
    if not left_grams and not right_grams:
        return 1.0
    if not left_grams or not right_grams:
        return 0.0
    return float(len(left_grams & right_grams)) / float(len(left_grams | right_grams))


def _char_ngrams(text: str, n: int) -> Set[str]:
    if n <= 0:
        return set()
    padded = f" {text} "
    if len(padded) < n:
        return {padded}
    return {padded[index : index + n] for index in range(0, len(padded) - n + 1)}


def _normalize_phrase(text: str) -> str:
    lowered = text.strip().lower()
    collapsed = re.sub(r"[^a-z0-9\s]", " ", lowered)
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    return collapsed


def _load_issue_type_field_values(base_dir: Path) -> Dict[int, Dict[str, Dict[str, List[str]]]]:
    mapping: Dict[int, Dict[str, Dict[str, List[str]]]] = {}
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
        mapping[issue_number] = _extract_type_field_values_from_payload(payload)
    return mapping


def _extract_type_field_values_from_payload(payload: Mapping[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    issue = payload.get("issue")
    if not isinstance(issue, dict):
        return {}
    comments = issue.get("comments")
    if not isinstance(comments, list):
        return {}

    result: Dict[str, Dict[str, List[str]]] = {}
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
            metadata = artifact.get("metadata")
            if not isinstance(artifact_type, str) or not artifact_type.strip() or not isinstance(metadata, dict):
                continue

            type_name = artifact_type.strip()
            type_bucket = result.setdefault(type_name, {})
            for field, raw_value in metadata.items():
                if not isinstance(field, str) or not field.strip():
                    continue
                value = _metadata_value_to_string(raw_value)
                if not value:
                    continue
                type_bucket.setdefault(field.strip(), []).append(value)
    return result


def _metadata_value_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_metadata_value_to_string(item) for item in value if _metadata_value_to_string(item))
    if isinstance(value, dict):
        parts = []
        for key in sorted(value.keys()):
            key_str = str(key).strip()
            val_str = _metadata_value_to_string(value.get(key))
            if key_str and val_str:
                parts.append(f"{key_str}: {val_str}")
        return " ; ".join(parts)
    return str(value).strip()


def _merge_type_field_values(
    target: Dict[str, Dict[str, List[str]]], source: Mapping[str, Mapping[str, Sequence[str]]]
) -> None:
    for type_name, fields in source.items():
        type_bucket = target.setdefault(type_name, {})
        for field_name, values in fields.items():
            type_bucket.setdefault(field_name, []).extend(value for value in values if value)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator) / float(denominator)


def _harmonic(precision: float, recall: float) -> float:
    if (precision + recall) == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _macro_average(items: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not items:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
            "soft_precision": 0.0,
            "soft_recall": 0.0,
            "soft_f1": 0.0,
        }

    return {
        "precision": _mean(float(item.get("precision", 0.0)) for item in items),
        "recall": _mean(float(item.get("recall", 0.0)) for item in items),
        "f1": _mean(float(item.get("f1", 0.0)) for item in items),
        "jaccard": _mean(float(item.get("jaccard", 0.0)) for item in items),
        "soft_precision": _mean(float(item.get("soft_precision", 0.0)) for item in items),
        "soft_recall": _mean(float(item.get("soft_recall", 0.0)) for item in items),
        "soft_f1": _mean(float(item.get("soft_f1", 0.0)) for item in items),
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def _model_dir_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__") or "unknown-model"


@dataclass
class _ValueScore:
    human_count: int
    llm_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    jaccard: float
    sum_best_llm: float
    sum_best_human: float
    soft_precision: float
    soft_recall: float
    soft_f1: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "human_value_count": self.human_count,
            "llm_value_count": self.llm_count,
            "true_positive_count": self.tp,
            "false_positive_count": self.fp,
            "false_negative_count": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "jaccard": self.jaccard,
            "soft_precision": self.soft_precision,
            "soft_recall": self.soft_recall,
            "soft_f1": self.soft_f1,
        }


@dataclass
class _Accumulator:
    human_count: int = 0
    llm_count: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    sum_best_llm: float = 0.0
    sum_best_human: float = 0.0

    def add(self, value_score: _ValueScore) -> None:
        self.human_count += value_score.human_count
        self.llm_count += value_score.llm_count
        self.tp += value_score.tp
        self.fp += value_score.fp
        self.fn += value_score.fn
        self.sum_best_llm += value_score.sum_best_llm
        self.sum_best_human += value_score.sum_best_human

    def merge(self, other: "_Accumulator") -> None:
        self.human_count += other.human_count
        self.llm_count += other.llm_count
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.sum_best_llm += other.sum_best_llm
        self.sum_best_human += other.sum_best_human

    def to_metrics(self) -> Dict[str, Any]:
        precision = _ratio(self.tp, self.llm_count)
        recall = _ratio(self.tp, self.human_count)
        f1 = _harmonic(precision, recall)
        jaccard = _ratio(self.tp, self.tp + self.fp + self.fn)

        soft_precision = 1.0 if self.llm_count == 0 else self.sum_best_llm / float(self.llm_count)
        soft_recall = 1.0 if self.human_count == 0 else self.sum_best_human / float(self.human_count)
        soft_f1 = _harmonic(soft_precision, soft_recall)

        return {
            "human_value_count": self.human_count,
            "llm_value_count": self.llm_count,
            "true_positive_count": self.tp,
            "false_positive_count": self.fp,
            "false_negative_count": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
            "soft_precision": soft_precision,
            "soft_recall": soft_recall,
            "soft_f1": soft_f1,
        }
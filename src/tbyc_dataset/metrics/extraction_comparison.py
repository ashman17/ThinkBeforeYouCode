from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from tbyc_dataset.metrics.metadata_matching import _Accumulator, _score_values
from tbyc_dataset.metrics.summary_matching import _build_overall_summary, _build_scorers, _combine_summaries, _coverage_metrics, _empty_scores, _score_pair
from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import ensure_directory, read_json, write_json


def compute_extraction_comparison_metrics(
    *,
    owner: str,
    repo: str,
    output_root: str,
    issue_number: Optional[int] = None,
    similarity_threshold: float = 0.82,
    similarity_metric: str = "max_all",
    codebert_model: str = "microsoft/codebert-base",
    bertscore_model: str = "microsoft/codebert-base",
    bleurt_model: str = "Elron/bleurt-base-512",
    bleurt_postprocess: str = "sigmoid",
    bleurt_clip_min: Optional[float] = 0.0,
    bleurt_sigmoid_temperature: float = 2.0,
    bleurt_sigmoid_bias: float = 0.0,
) -> Dict[str, Any]:
    repo_ref = RepositoryRef(owner=owner, name=repo)
    root = Path(output_root)
    llm_dir = root / "extractions" / repo_ref.fs_slug
    regex_dir = root / "extractions_regex" / repo_ref.fs_slug
    metrics_dir = root / "metrics_regex_comparison" / repo_ref.fs_slug
    ensure_directory(metrics_dir)

    llm_issues = _load_issue_payloads(llm_dir)
    regex_issues = _load_issue_payloads(regex_dir)
    issue_numbers = sorted(set(llm_issues.keys()) | set(regex_issues.keys()))
    if issue_number is not None:
        issue_numbers = [number for number in issue_numbers if number == issue_number]

    scorers = _build_scorers(
        codebert_model=codebert_model,
        bertscore_model=bertscore_model,
        bleurt_model=bleurt_model,
        bleurt_postprocess=bleurt_postprocess,
        bleurt_clip_min=bleurt_clip_min,
        bleurt_sigmoid_temperature=bleurt_sigmoid_temperature,
        bleurt_sigmoid_bias=bleurt_sigmoid_bias,
    )

    per_issue: List[Dict[str, Any]] = []
    all_llm_tag_frequency: Dict[str, int] = {}
    all_regex_tag_frequency: Dict[str, int] = {}
    for number in issue_numbers:
        llm_issue = llm_issues.get(number, _empty_issue(number))
        regex_issue = regex_issues.get(number, _empty_issue(number))
        issue_report = _score_issue(
            issue_number=number,
            llm_issue=llm_issue,
            regex_issue=regex_issue,
            similarity_threshold=similarity_threshold,
            similarity_metric=similarity_metric,
            scorers=scorers,
        )
        per_issue.append(issue_report)
        _merge_frequency(all_llm_tag_frequency, issue_report["tag_frequency"]["llm"])
        _merge_frequency(all_regex_tag_frequency, issue_report["tag_frequency"]["regex"])

    macro = _macro_average(per_issue)
    overall = _build_overall(per_issue)
    per_type = _score_per_type(
        issue_numbers=issue_numbers,
        llm_issues=llm_issues,
        regex_issues=regex_issues,
        similarity_threshold=similarity_threshold,
        similarity_metric=similarity_metric,
        scorers=scorers,
    )

    report: Dict[str, Any] = {
        "repository": repo_ref.slug,
        "metric": "llm_vs_regex_extraction_comparison",
        "issue_count": len(per_issue),
        "issue_number_filter": issue_number,
        "reference_side": "regex",
        "candidate_side": "llm",
        "similarity": {
            "method": similarity_metric,
            "threshold": similarity_threshold,
        },
        "models": {
            "codebert": codebert_model,
            "bertscore": bertscore_model,
            "bleurt": bleurt_model,
        },
        "overall": overall,
        "macro_average": macro,
        "per_type": per_type,
        "per_issue": per_issue,
        "tag_frequency": {
            "llm": all_llm_tag_frequency,
            "regex": all_regex_tag_frequency,
        },
    }
    write_json(metrics_dir / "comparison.json", report)
    return report


def _score_issue(
    *,
    issue_number: int,
    llm_issue: Mapping[str, Any],
    regex_issue: Mapping[str, Any],
    similarity_threshold: float,
    similarity_metric: str,
    scorers: Mapping[str, Any],
) -> Dict[str, Any]:
    llm_types = _extract_types(llm_issue)
    regex_types = _extract_types(regex_issue)
    type_metrics = _set_metrics(regex_types, llm_types)

    llm_type_details = _extract_type_details(llm_issue)
    regex_type_details = _extract_type_details(regex_issue)
    tag_payload = _score_tag_issue(regex_type_details, llm_type_details)

    llm_metadata = _extract_type_field_values(llm_issue)
    regex_metadata = _extract_type_field_values(regex_issue)
    metadata_payload = _score_metadata_issue(
        regex_map=regex_metadata,
        llm_map=llm_metadata,
        similarity_threshold=similarity_threshold,
        similarity_metric=similarity_metric,
    )

    llm_summaries = _extract_type_summaries(llm_issue)
    regex_summaries = _extract_type_summaries(regex_issue)
    summary_payload = _score_summary_issue(regex_types=regex_summaries, llm_types=llm_summaries, scorers=scorers)

    llm_comment_count = len(llm_issue.get("comments", [])) if isinstance(llm_issue.get("comments"), list) else 0
    regex_comment_count = len(regex_issue.get("comments", [])) if isinstance(regex_issue.get("comments"), list) else 0
    llm_artifact_count = int(llm_issue.get("artifact_count", 0))
    regex_artifact_count = int(regex_issue.get("artifact_count", 0))

    return {
        "issue_number": issue_number,
        "counts": {
            "llm_comment_count": llm_comment_count,
            "regex_comment_count": regex_comment_count,
            "llm_artifact_count": llm_artifact_count,
            "regex_artifact_count": regex_artifact_count,
            "artifact_count_ratio": _safe_ratio(llm_artifact_count, regex_artifact_count),
            "artifact_count_delta": llm_artifact_count - regex_artifact_count,
        },
        "type": type_metrics,
        "tag": tag_payload["pooled"],
        "metadata": metadata_payload["pooled"],
        "summary": summary_payload["scores_with_unmatched_penalty"],
        "summary_coverage": summary_payload["coverage"],
        "per_matching_type": {
            "tag": tag_payload["per_matching_type"],
            "metadata": metadata_payload["per_type"],
            "summary": summary_payload["per_matching_type"],
        },
        "tag_frequency": {
            "llm": _tag_frequency_from_issue(llm_issue),
            "regex": _tag_frequency_from_issue(regex_issue),
        },
    }


def _build_overall(per_issue: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    llm_comment_count = sum(int(item.get("counts", {}).get("llm_comment_count", 0)) for item in per_issue)
    regex_comment_count = sum(int(item.get("counts", {}).get("regex_comment_count", 0)) for item in per_issue)
    llm_artifact_count = sum(int(item.get("counts", {}).get("llm_artifact_count", 0)) for item in per_issue)
    regex_artifact_count = sum(int(item.get("counts", {}).get("regex_artifact_count", 0)) for item in per_issue)

    issue_rows = [
        {
            "scores": item.get("summary", _empty_scores()),
            "matching_type_count": int(item.get("summary_coverage", {}).get("matching_type_recall", 0.0) > 0.0),
            "weighted_scores": item.get("summary", _empty_scores()),
            "coverage": item.get("summary_coverage", {}),
        }
        for item in per_issue
    ]

    return {
        "counts": {
            "llm_comment_count": llm_comment_count,
            "regex_comment_count": regex_comment_count,
            "llm_artifact_count": llm_artifact_count,
            "regex_artifact_count": regex_artifact_count,
            "artifact_count_ratio": _safe_ratio(llm_artifact_count, regex_artifact_count),
            "artifact_count_delta": llm_artifact_count - regex_artifact_count,
        },
        "type": _macro_block([item.get("type", {}) for item in per_issue]),
        "tag": _macro_block([item.get("tag", {}) for item in per_issue]),
        "metadata": _macro_block([item.get("metadata", {}) for item in per_issue], include_soft=True),
        "summary": _build_overall_summary(per_issue=issue_rows, per_type=[]),
    }


def _macro_average(per_issue: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "type": _macro_block([item.get("type", {}) for item in per_issue]),
        "tag": _macro_block([item.get("tag", {}) for item in per_issue]),
        "metadata": _macro_block([item.get("metadata", {}) for item in per_issue], include_soft=True),
        "summary": {
            "codebert": {"cosine": _mean(float(item.get("summary", {}).get("codebert", {}).get("cosine", 0.0)) for item in per_issue)},
            "bertscore": {
                "precision": _mean(float(item.get("summary", {}).get("bertscore", {}).get("precision", 0.0)) for item in per_issue),
                "recall": _mean(float(item.get("summary", {}).get("bertscore", {}).get("recall", 0.0)) for item in per_issue),
                "f1": _mean(float(item.get("summary", {}).get("bertscore", {}).get("f1", 0.0)) for item in per_issue),
            },
            "bleurt": {"score": _mean(float(item.get("summary", {}).get("bleurt", {}).get("score", 0.0)) for item in per_issue)},
            "coverage": {
                "matching_type_precision": _mean(float(item.get("summary_coverage", {}).get("matching_type_precision", 0.0)) for item in per_issue),
                "matching_type_recall": _mean(float(item.get("summary_coverage", {}).get("matching_type_recall", 0.0)) for item in per_issue),
                "matching_type_f1": _mean(float(item.get("summary_coverage", {}).get("matching_type_f1", 0.0)) for item in per_issue),
            },
        },
        "counts": {
            "artifact_count_ratio": _mean(float(item.get("counts", {}).get("artifact_count_ratio", 0.0)) for item in per_issue),
            "artifact_count_delta": _mean(float(item.get("counts", {}).get("artifact_count_delta", 0.0)) for item in per_issue),
        },
    }


def _score_per_type(
    *,
    issue_numbers: Sequence[int],
    llm_issues: Mapping[int, Mapping[str, Any]],
    regex_issues: Mapping[int, Mapping[str, Any]],
    similarity_threshold: float,
    similarity_metric: str,
    scorers: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    all_types: Set[str] = set()
    for number in issue_numbers:
        all_types.update(_extract_types(llm_issues.get(number, {})))
        all_types.update(_extract_types(regex_issues.get(number, {})))

    rows: List[Dict[str, Any]] = []
    for type_name in sorted(all_types):
        type_issue_rows: List[Dict[str, Any]] = []
        for number in issue_numbers:
            llm_issue = llm_issues.get(number, _empty_issue(number))
            regex_issue = regex_issues.get(number, _empty_issue(number))
            llm_type_details = _extract_type_details(llm_issue)
            regex_type_details = _extract_type_details(regex_issue)
            llm_has = type_name in llm_type_details
            regex_has = type_name in regex_type_details
            if not (llm_has or regex_has):
                continue

            type_metrics = _set_metrics({type_name} if regex_has else set(), {type_name} if llm_has else set())
            tag_payload = _score_tag_issue(
                {type_name: regex_type_details.get(type_name, {})} if regex_has else {},
                {type_name: llm_type_details.get(type_name, {})} if llm_has else {},
            )
            metadata_payload = _score_metadata_issue(
                regex_map={type_name: _extract_type_field_values(regex_issue).get(type_name, {})} if regex_has else {},
                llm_map={type_name: _extract_type_field_values(llm_issue).get(type_name, {})} if llm_has else {},
                similarity_threshold=similarity_threshold,
                similarity_metric=similarity_metric,
            )
            summary_payload = _score_summary_issue(
                regex_types={type_name: _extract_type_summaries(regex_issue).get(type_name, [])} if regex_has else {},
                llm_types={type_name: _extract_type_summaries(llm_issue).get(type_name, [])} if llm_has else {},
                scorers=scorers,
            )
            type_issue_rows.append(
                {
                    "type": type_metrics,
                    "tag": tag_payload["pooled"],
                    "metadata": metadata_payload["pooled"],
                    "summary": summary_payload["scores_with_unmatched_penalty"],
                }
            )

        rows.append(
            {
                "type": type_name,
                "issue_count": len(type_issue_rows),
                "type_metrics": _macro_block([item.get("type", {}) for item in type_issue_rows]),
                "tag_metrics": _macro_block([item.get("tag", {}) for item in type_issue_rows]),
                "metadata_metrics": _macro_block([item.get("metadata", {}) for item in type_issue_rows], include_soft=True),
                "summary_metrics": {
                    "codebert_cosine": _mean(float(item.get("summary", {}).get("codebert", {}).get("cosine", 0.0)) for item in type_issue_rows),
                    "bertscore_f1": _mean(float(item.get("summary", {}).get("bertscore", {}).get("f1", 0.0)) for item in type_issue_rows),
                    "bleurt_score": _mean(float(item.get("summary", {}).get("bleurt", {}).get("score", 0.0)) for item in type_issue_rows),
                },
            }
        )
    return rows


def _score_tag_issue(
    regex_types: Mapping[str, Mapping[str, Any]],
    llm_types: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    matching_types = sorted(set(regex_types.keys()) & set(llm_types.keys()))
    rows: List[Dict[str, Any]] = []
    metrics_rows: List[Mapping[str, Any]] = []
    for type_name in matching_types:
        regex_tags = set(regex_types.get(type_name, {}).get("tags", set()))
        llm_tags = set(llm_types.get(type_name, {}).get("tags", set()))
        metrics = _set_metrics(regex_tags, llm_tags)
        rows.append({"type": type_name, **metrics})
        metrics_rows.append(metrics)
    return {
        "pooled": _macro_block(metrics_rows),
        "per_matching_type": rows,
    }


def _score_metadata_issue(
    *,
    regex_map: Mapping[str, Mapping[str, Sequence[str]]],
    llm_map: Mapping[str, Mapping[str, Sequence[str]]],
    similarity_threshold: float,
    similarity_metric: str,
) -> Dict[str, Any]:
    type_names = sorted(set(regex_map.keys()) | set(llm_map.keys()))
    per_type: List[Dict[str, Any]] = []
    aggregate = _Accumulator()
    for type_name in type_names:
        regex_fields = regex_map.get(type_name, {})
        llm_fields = llm_map.get(type_name, {})
        field_names = sorted(set(regex_fields.keys()) | set(llm_fields.keys()))
        type_acc = _Accumulator()
        for field_name in field_names:
            score = _score_values(
                [v for v in regex_fields.get(field_name, []) if v],
                [v for v in llm_fields.get(field_name, []) if v],
                similarity_threshold,
                similarity_metric,
            )
            type_acc.merge(score)
            aggregate.merge(score)
        per_type.append({"type": type_name, **type_acc.to_metrics()})
    return {"pooled": aggregate.to_metrics(), "per_type": per_type}


def _score_summary_issue(
    *,
    regex_types: Mapping[str, Sequence[str]],
    llm_types: Mapping[str, Sequence[str]],
    scorers: Mapping[str, Any],
) -> Dict[str, Any]:
    matching_types = sorted(set(regex_types.keys()) & set(llm_types.keys()))
    compared: List[Dict[str, Any]] = []
    score_rows: List[Mapping[str, Any]] = []
    for type_name in matching_types:
        regex_summary = _combine_summaries(regex_types.get(type_name, []))
        llm_summary = _combine_summaries(llm_types.get(type_name, []))
        score_row = _score_pair(regex_summary, llm_summary, scorers)
        score_rows.append(score_row)
        compared.append({"type": type_name, "scores": score_row})
    coverage = _coverage_metrics(
        human_count=len(regex_types),
        llm_count=len(llm_types),
        matched_count=len(matching_types),
    )
    matched_only = _summary_macro_average(score_rows)
    return {
        "coverage": coverage,
        "scores_with_unmatched_penalty": _apply_penalty(matched_only, coverage["matching_type_recall"]),
        "per_matching_type": compared,
    }


def _apply_penalty(scores: Mapping[str, Any], factor: float) -> Dict[str, Any]:
    factor = max(0.0, min(1.0, float(factor)))
    return {
        "codebert": {"cosine": float(scores.get("codebert", {}).get("cosine", 0.0)) * factor},
        "bertscore": {
            "precision": float(scores.get("bertscore", {}).get("precision", 0.0)) * factor,
            "recall": float(scores.get("bertscore", {}).get("recall", 0.0)) * factor,
            "f1": float(scores.get("bertscore", {}).get("f1", 0.0)) * factor,
        },
        "bleurt": {"score": float(scores.get("bleurt", {}).get("score", 0.0)) * factor},
    }


def _summary_macro_average(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return _empty_scores()
    return {
        "codebert": {"cosine": _mean(float(item.get("codebert", {}).get("cosine", 0.0)) for item in rows)},
        "bertscore": {
            "precision": _mean(float(item.get("bertscore", {}).get("precision", 0.0)) for item in rows),
            "recall": _mean(float(item.get("bertscore", {}).get("recall", 0.0)) for item in rows),
            "f1": _mean(float(item.get("bertscore", {}).get("f1", 0.0)) for item in rows),
        },
        "bleurt": {"score": _mean(float(item.get("bleurt", {}).get("score", 0.0)) for item in rows)},
    }


def _load_issue_payloads(base_dir: Path) -> Dict[int, Mapping[str, Any]]:
    mapping: Dict[int, Mapping[str, Any]] = {}
    if not base_dir.exists():
        return mapping
    for path in sorted(base_dir.glob("issue_*.json")):
        payload = read_json(path)
        issue = payload.get("issue", {})
        raw = issue.get("issue_number", payload.get("issue_number"))
        try:
            mapping[int(raw)] = issue if isinstance(issue, dict) else {}
        except Exception:
            continue
    return mapping


def _empty_issue(issue_number: int) -> Dict[str, Any]:
    return {"issue_number": issue_number, "comments": [], "artifact_count": 0}


def _extract_types(issue: Mapping[str, Any]) -> Set[str]:
    types: Set[str] = set()
    for artifact in _iter_artifacts(issue):
        artifact_type = artifact.get("type")
        if isinstance(artifact_type, str) and artifact_type.strip():
            types.add(artifact_type.strip())
    return types


def _extract_type_details(issue: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    details: Dict[str, Dict[str, Any]] = {}
    for artifact in _iter_artifacts(issue):
        type_name = artifact.get("type")
        if not isinstance(type_name, str) or not type_name.strip():
            continue
        bucket = details.setdefault(type_name.strip(), {"tags": set(), "summary_count": 0})
        summary = artifact.get("summary")
        if isinstance(summary, str) and summary.strip():
            bucket["summary_count"] = int(bucket.get("summary_count", 0)) + 1
        tags = artifact.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    bucket["tags"].add(tag.strip())
    return details


def _extract_type_field_values(issue: Mapping[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    result: Dict[str, Dict[str, List[str]]] = {}
    for artifact in _iter_artifacts(issue):
        type_name = artifact.get("type")
        metadata = artifact.get("metadata")
        if not isinstance(type_name, str) or not type_name.strip() or not isinstance(metadata, dict):
            continue
        type_bucket = result.setdefault(type_name.strip(), {})
        for field_name, value in metadata.items():
            if not isinstance(field_name, str) or not field_name.strip():
                continue
            if value in (None, "", "unknown"):
                continue
            type_bucket.setdefault(field_name.strip(), []).append(str(value).strip())
    return result


def _extract_type_summaries(issue: Mapping[str, Any]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for artifact in _iter_artifacts(issue):
        type_name = artifact.get("type")
        summary = artifact.get("summary")
        if isinstance(type_name, str) and type_name.strip() and isinstance(summary, str) and summary.strip():
            result.setdefault(type_name.strip(), []).append(summary.strip())
    return result


def _tag_frequency_from_issue(issue: Mapping[str, Any]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for artifact in _iter_artifacts(issue):
        tags = artifact.get("tags", [])
        if not isinstance(tags, list):
            continue
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                freq[tag.strip()] = freq.get(tag.strip(), 0) + 1
    return freq


def _iter_artifacts(issue: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    comments = issue.get("comments")
    if not isinstance(comments, list):
        return []
    artifacts: List[Mapping[str, Any]] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        items = comment.get("artifacts")
        if not isinstance(items, list):
            continue
        for artifact in items:
            if isinstance(artifact, dict):
                artifacts.append(artifact)
    return artifacts


def _set_metrics(reference_values: Set[str], candidate_values: Set[str]) -> Dict[str, Any]:
    intersection = reference_values & candidate_values
    union = reference_values | candidate_values
    precision = _safe_ratio(len(intersection), len(candidate_values))
    recall = _safe_ratio(len(intersection), len(reference_values))
    f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
    jaccard = _safe_ratio(len(intersection), len(union))
    return {
        "reference_count": len(reference_values),
        "candidate_count": len(candidate_values),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


def _macro_block(rows: Sequence[Mapping[str, Any]], *, include_soft: bool = False) -> Dict[str, float]:
    keys = ["precision", "recall", "f1", "jaccard"]
    if include_soft:
        keys.extend(["soft_precision", "soft_recall", "soft_f1"])
    result: Dict[str, float] = {}
    for key in keys:
        result[key] = _mean(float(row.get(key, 0.0)) for row in rows)
    return result


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator) / float(denominator)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def _merge_frequency(dst: Dict[str, int], src: Mapping[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + int(value)

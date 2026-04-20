from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import ensure_directory, read_json, write_json


def compute_summary_matching_metrics(
    *,
    owner: str,
    repo: str,
    output_root: str,
    model_id: str,
    issue_number: Optional[int] = None,
    codebert_model: str = "microsoft/codebert-base",
    bertscore_model: str = "microsoft/codebert-base",
    bleurt_model: str = "Elron/bleurt-base-512",
) -> Dict[str, Any]:
    repo_ref = RepositoryRef(owner=owner, name=repo)
    root = Path(output_root)
    model_dir = _model_dir_name(model_id)
    extracted_dir = root / "extractions" / repo_ref.fs_slug
    derived_dir = root / "derived" / model_dir / repo_ref.fs_slug
    metrics_dir = root / "metrics" / model_dir / repo_ref.fs_slug
    ensure_directory(metrics_dir)

    extracted = _load_issue_type_summaries(extracted_dir)
    derived = _load_issue_type_summaries(derived_dir)

    issue_numbers = sorted(set(extracted.keys()) | set(derived.keys()))
    if issue_number is not None:
        issue_numbers = [number for number in issue_numbers if number == issue_number]

    scorers = _build_scorers(
        codebert_model=codebert_model,
        bertscore_model=bertscore_model,
        bleurt_model=bleurt_model,
    )

    per_issue: List[Dict[str, Any]] = []
    for number in issue_numbers:
        per_issue.append(
            _score_issue(
                issue_number=number,
                human_types=extracted.get(number, {}),
                llm_types=derived.get(number, {}),
                scorers=scorers,
            )
        )

    per_type = _score_per_type(issue_numbers=issue_numbers, extracted=extracted, derived=derived, scorers=scorers)
    overall = _build_overall_summary(per_issue=per_issue, per_type=per_type)

    report: Dict[str, Any] = {
        "repository": repo_ref.slug,
        "model_id": model_id,
        "metric": "summary_similarity_matching_types",
        "issue_count": len(per_issue),
        "issue_number_filter": issue_number,
        "comparison_rule": "aggregate summaries by type, compare only for types present in both human and llm",
        "models": {
            "codebert": codebert_model,
            "bertscore": bertscore_model,
            "bleurt": bleurt_model,
        },
        "overall": overall,
        "per_type": per_type,
        "per_issue": per_issue,
    }
    write_json(metrics_dir / "summary_matching.json", report)
    return report


def _score_issue(
    *,
    issue_number: int,
    human_types: Mapping[str, Sequence[str]],
    llm_types: Mapping[str, Sequence[str]],
    scorers: Mapping[str, "_BaseScorer"],
) -> Dict[str, Any]:
    matching_types = sorted(set(human_types.keys()) & set(llm_types.keys()))
    compared: List[Dict[str, Any]] = []
    pooled_score_rows: List[Mapping[str, Any]] = []
    weighted_rows: List[Tuple[Mapping[str, Any], float]] = []

    for type_name in matching_types:
        human_summary = _combine_summaries(human_types.get(type_name, []))
        llm_summary = _combine_summaries(llm_types.get(type_name, []))
        score_row = _score_pair(human_summary, llm_summary, scorers)
        pooled_score_rows.append(score_row)
        weight = float(max(1, len(human_types.get(type_name, []))))
        weighted_rows.append((score_row, weight))
        compared.append(
            {
                "type": type_name,
                "human_summary_count": len(human_types.get(type_name, [])),
                "llm_summary_count": len(llm_types.get(type_name, [])),
                "human_summary_chars": len(human_summary),
                "llm_summary_chars": len(llm_summary),
                "scores": score_row,
            }
        )

    coverage = _coverage_metrics(
        human_count=len(human_types),
        llm_count=len(llm_types),
        matched_count=len(matching_types),
    )
    matched_scores = _macro_average_scores(pooled_score_rows)
    weighted_scores = _weighted_average_scores(weighted_rows)
    penalized_scores = _apply_penalty_to_scores(matched_scores, coverage["matching_type_recall"])

    return {
        "issue_number": issue_number,
        "human_type_count": len(human_types),
        "llm_type_count": len(llm_types),
        "matching_type_count": len(matching_types),
        "coverage": coverage,
        "scores": matched_scores,
        "weighted_scores": weighted_scores,
        "scores_with_unmatched_penalty": penalized_scores,
        "per_matching_type": compared,
    }


def _score_per_type(
    *,
    issue_numbers: Sequence[int],
    extracted: Mapping[int, Mapping[str, Sequence[str]]],
    derived: Mapping[int, Mapping[str, Sequence[str]]],
    scorers: Mapping[str, "_BaseScorer"],
) -> List[Dict[str, Any]]:
    all_types: Set[str] = set()
    for number in issue_numbers:
        all_types.update(extracted.get(number, {}).keys())
        all_types.update(derived.get(number, {}).keys())

    results: List[Dict[str, Any]] = []
    for type_name in sorted(all_types):
        human_issue_count = 0
        llm_issue_count = 0
        matched_issue_count = 0
        human_chunks: List[str] = []
        llm_chunks: List[str] = []
        support_weight = 0.0
        for number in issue_numbers:
            human_list = list(extracted.get(number, {}).get(type_name, []))
            llm_list = list(derived.get(number, {}).get(type_name, []))
            if human_list:
                human_issue_count += 1
            if llm_list:
                llm_issue_count += 1
            if human_list and llm_list:
                matched_issue_count += 1
                human_chunks.extend(human_list)
                llm_chunks.extend(llm_list)
                support_weight += float(max(1, len(human_list)))

        if matched_issue_count > 0:
            human_summary = _combine_summaries(human_chunks)
            llm_summary = _combine_summaries(llm_chunks)
            scores_matched_only = _score_pair(human_summary, llm_summary, scorers)
        else:
            human_summary = ""
            llm_summary = ""
            scores_matched_only = _empty_scores()

        coverage = _coverage_metrics(
            human_count=human_issue_count,
            llm_count=llm_issue_count,
            matched_count=matched_issue_count,
        )
        penalized_scores = _apply_penalty_to_scores(scores_matched_only, coverage["matching_type_recall"])

        results.append(
            {
                "type": type_name,
                "issue_count": len(issue_numbers),
                "human_issue_count": human_issue_count,
                "llm_issue_count": llm_issue_count,
                "matched_issue_count": matched_issue_count,
                "human_summary_count": len(human_chunks),
                "llm_summary_count": len(llm_chunks),
                "human_summary_chars": len(human_summary),
                "llm_summary_chars": len(llm_summary),
                "coverage": coverage,
                "support_weight": support_weight,
                "scores": scores_matched_only,
                "scores_with_unmatched_penalty": penalized_scores,
            }
        )

    return results


def _load_issue_type_summaries(base_dir: Path) -> Dict[int, Dict[str, List[str]]]:
    mapping: Dict[int, Dict[str, List[str]]] = {}
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

        mapping[issue_number] = _extract_type_summaries(payload)
    return mapping


def _extract_type_summaries(payload: Mapping[str, Any]) -> Dict[str, List[str]]:
    issue = payload.get("issue")
    if not isinstance(issue, dict):
        return {}
    comments = issue.get("comments")
    if not isinstance(comments, list):
        return {}

    result: Dict[str, List[str]] = {}
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
            summary = artifact.get("summary")
            if not isinstance(type_name, str) or not type_name.strip():
                continue
            if not isinstance(summary, str) or not summary.strip():
                continue
            result.setdefault(type_name.strip(), []).append(summary.strip())
    return result


def _combine_summaries(summaries: Sequence[str]) -> str:
    cleaned = [item.strip() for item in summaries if item and item.strip()]
    if not cleaned:
        return ""
    return "\n".join(cleaned)


def _score_pair(human_summary: str, llm_summary: str, scorers: Mapping[str, "_BaseScorer"]) -> Dict[str, Any]:
    score: Dict[str, Any] = {}
    for name, scorer in scorers.items():
        score[name] = scorer.score(human_summary, llm_summary)
    return score


def _macro_average_scores(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "codebert": {"cosine": 0.0},
            "bertscore": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "bleurt": {"score": 0.0},
        }

    codebert_values = [float(item.get("codebert", {}).get("cosine", 0.0)) for item in rows]
    bertscore_precision = [float(item.get("bertscore", {}).get("precision", 0.0)) for item in rows]
    bertscore_recall = [float(item.get("bertscore", {}).get("recall", 0.0)) for item in rows]
    bertscore_f1 = [float(item.get("bertscore", {}).get("f1", 0.0)) for item in rows]
    bleurt_values = [float(item.get("bleurt", {}).get("score", 0.0)) for item in rows]

    return {
        "codebert": {"cosine": _mean(codebert_values)},
        "bertscore": {
            "precision": _mean(bertscore_precision),
            "recall": _mean(bertscore_recall),
            "f1": _mean(bertscore_f1),
        },
        "bleurt": {"score": _mean(bleurt_values)},
    }


def _weighted_average_scores(weighted_rows: Sequence[Tuple[Mapping[str, Any], float]]) -> Dict[str, Any]:
    if not weighted_rows:
        return _empty_scores()

    total_weight = sum(max(0.0, float(weight)) for _, weight in weighted_rows)
    if total_weight <= 0.0:
        return _empty_scores()

    def _weighted_value(path: Sequence[str]) -> float:
        acc = 0.0
        for row, weight in weighted_rows:
            curr: Any = row
            for key in path:
                if not isinstance(curr, Mapping):
                    curr = 0.0
                    break
                curr = curr.get(key, 0.0)
            acc += float(curr) * float(max(0.0, weight))
        return acc / total_weight

    return {
        "codebert": {"cosine": _weighted_value(("codebert", "cosine"))},
        "bertscore": {
            "precision": _weighted_value(("bertscore", "precision")),
            "recall": _weighted_value(("bertscore", "recall")),
            "f1": _weighted_value(("bertscore", "f1")),
        },
        "bleurt": {"score": _weighted_value(("bleurt", "score"))},
    }


def _empty_scores() -> Dict[str, Any]:
    return {
        "codebert": {"cosine": 0.0},
        "bertscore": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "bleurt": {"score": 0.0},
    }


def _apply_penalty_to_scores(scores: Mapping[str, Any], penalty: float) -> Dict[str, Any]:
    scale = max(0.0, min(1.0, float(penalty)))
    return {
        "codebert": {"cosine": float(scores.get("codebert", {}).get("cosine", 0.0)) * scale},
        "bertscore": {
            "precision": float(scores.get("bertscore", {}).get("precision", 0.0)) * scale,
            "recall": float(scores.get("bertscore", {}).get("recall", 0.0)) * scale,
            "f1": float(scores.get("bertscore", {}).get("f1", 0.0)) * scale,
        },
        "bleurt": {"score": float(scores.get("bleurt", {}).get("score", 0.0)) * scale},
    }


def _coverage_metrics(*, human_count: int, llm_count: int, matched_count: int) -> Dict[str, float]:
    precision = _ratio(matched_count, llm_count)
    recall = _ratio(matched_count, human_count)
    f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "matching_type_precision": precision,
        "matching_type_recall": recall,
        "matching_type_f1": f1,
    }


def _build_overall_summary(
    *,
    per_issue: Sequence[Mapping[str, Any]],
    per_type: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    issue_rows_all = [item.get("scores", _empty_scores()) for item in per_issue]
    issue_rows_matched_only = [item.get("scores", _empty_scores()) for item in per_issue if int(item.get("matching_type_count", 0)) > 0]
    issue_rows_penalized = [item.get("scores_with_unmatched_penalty", _empty_scores()) for item in per_issue]

    weighted_rows: List[Tuple[Mapping[str, Any], float]] = []
    for issue in per_issue:
        weight = float(max(0, int(issue.get("matching_type_count", 0))))
        weighted_rows.append((issue.get("weighted_scores", _empty_scores()), weight))

    coverage_rows = [item.get("coverage", {}) for item in per_issue]
    coverage_macro = {
        "matching_type_precision": _mean(float(item.get("matching_type_precision", 0.0)) for item in coverage_rows),
        "matching_type_recall": _mean(float(item.get("matching_type_recall", 0.0)) for item in coverage_rows),
        "matching_type_f1": _mean(float(item.get("matching_type_f1", 0.0)) for item in coverage_rows),
    }

    per_type_matched_rows = [item.get("scores", _empty_scores()) for item in per_type if int(item.get("matched_issue_count", 0)) > 0]
    per_type_penalized_rows = [item.get("scores_with_unmatched_penalty", _empty_scores()) for item in per_type]

    matched_only_macro = _macro_average_scores(issue_rows_matched_only)
    return {
        "matched_only_macro": matched_only_macro,
        "all_issues_macro_with_unmatched_penalty": _macro_average_scores(issue_rows_penalized),
        "all_issues_macro_raw": _macro_average_scores(issue_rows_all),
        "support_weighted_matched_only": _weighted_average_scores(weighted_rows),
        "coverage": coverage_macro,
        "penalized_overall": _apply_penalty_to_scores(matched_only_macro, coverage_macro["matching_type_recall"]),
        "per_type_matched_only_macro": _macro_average_scores(per_type_matched_rows),
        "per_type_macro_with_unmatched_penalty": _macro_average_scores(per_type_penalized_rows),
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator) / float(denominator)


def _model_dir_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__") or "unknown-model"


class _BaseScorer:
    def score(self, reference: str, candidate: str) -> Dict[str, float]:
        raise NotImplementedError


class _CodeBERTScorer(_BaseScorer):
    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency/environment specific
            raise RuntimeError(f"CodeBERT scorer dependencies unavailable: {exc}")

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def score(self, reference: str, candidate: str) -> Dict[str, float]:
        if not reference and not candidate:
            return {"cosine": 1.0}
        if not reference or not candidate:
            return {"cosine": 0.0}

        ref_embedding = self._encode(reference)
        cand_embedding = self._encode(candidate)
        cosine = float(self._torch.nn.functional.cosine_similarity(ref_embedding, cand_embedding, dim=0).item())
        return {"cosine": max(-1.0, min(1.0, cosine))}

    def _encode(self, text: str):
        with self._torch.no_grad():
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state.squeeze(0)
            mask = encoded["attention_mask"].squeeze(0).unsqueeze(-1).to(hidden.dtype)
            masked_hidden = hidden * mask
            token_count = self._torch.clamp(mask.sum(), min=1.0)
            pooled = masked_hidden.sum(dim=0) / token_count
            return pooled


class _BERTScoreScorer(_BaseScorer):
    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency/environment specific
            raise RuntimeError(f"BERTScore scorer dependencies unavailable: {exc}")

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def score(self, reference: str, candidate: str) -> Dict[str, float]:
        if not reference and not candidate:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not reference or not candidate:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        ref_tokens = self._encode_tokens(reference)
        cand_tokens = self._encode_tokens(candidate)
        if ref_tokens.shape[0] == 0 and cand_tokens.shape[0] == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if ref_tokens.shape[0] == 0 or cand_tokens.shape[0] == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        ref_norm = self._torch.nn.functional.normalize(ref_tokens, p=2, dim=1)
        cand_norm = self._torch.nn.functional.normalize(cand_tokens, p=2, dim=1)
        similarity = self._torch.matmul(cand_norm, ref_norm.T)

        precision = float(similarity.max(dim=1).values.mean().item())
        recall = float(similarity.max(dim=0).values.mean().item())
        f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1}

    def _encode_tokens(self, text: str):
        with self._torch.no_grad():
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state.squeeze(0)
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0).bool()

            special_ids = set(self._tokenizer.all_special_ids)
            keep_mask = attention_mask.clone()
            for idx in range(input_ids.shape[0]):
                if int(input_ids[idx].item()) in special_ids:
                    keep_mask[idx] = False

            if not bool(keep_mask.any().item()):
                return hidden.new_zeros((0, hidden.shape[-1]))
            return hidden[keep_mask]


class _BLEURTScorer(_BaseScorer):
    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency/environment specific
            raise RuntimeError(f"BLEURT scorer dependencies unavailable: {exc}")

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

    def score(self, reference: str, candidate: str) -> Dict[str, float]:
        if not reference and not candidate:
            return {"score": 1.0}
        if not reference or not candidate:
            return {"score": 0.0}

        with self._torch.no_grad():
            encoded = self._tokenizer(
                reference,
                candidate,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            outputs = self._model(**encoded)
            logits = outputs.logits
            if logits.ndim == 2:
                value = float(logits.squeeze(0).squeeze(-1).item())
            else:
                value = float(logits.item())
            return {"score": value}


@dataclass
class _UnavailableScorer(_BaseScorer):
    key: str
    reason: str

    def score(self, reference: str, candidate: str) -> Dict[str, float]:
        if self.key == "codebert":
            return {"cosine": 0.0, "unavailable": 1.0}
        if self.key == "bertscore":
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "unavailable": 1.0}
        return {"score": 0.0, "unavailable": 1.0}


def _build_scorers(*, codebert_model: str, bertscore_model: str, bleurt_model: str) -> Dict[str, _BaseScorer]:
    scorers: Dict[str, _BaseScorer] = {}

    try:
        scorers["codebert"] = _CodeBERTScorer(codebert_model)
    except Exception as exc:  # pragma: no cover - dependency/environment specific
        scorers["codebert"] = _UnavailableScorer("codebert", str(exc))

    try:
        scorers["bertscore"] = _BERTScoreScorer(bertscore_model)
    except Exception as exc:  # pragma: no cover - dependency/environment specific
        scorers["bertscore"] = _UnavailableScorer("bertscore", str(exc))

    try:
        scorers["bleurt"] = _BLEURTScorer(bleurt_model)
    except Exception as exc:  # pragma: no cover - dependency/environment specific
        scorers["bleurt"] = _UnavailableScorer("bleurt", str(exc))

    return scorers
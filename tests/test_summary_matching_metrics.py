from __future__ import annotations

from pathlib import Path

from tbyc_dataset.metrics import summary_matching
from tbyc_dataset.metrics.summary_matching import compute_summary_matching_metrics
from tbyc_dataset.storage import write_json


class _StaticScorer:
    def __init__(self, payload):
        self.payload = payload

    def score(self, reference: str, candidate: str):
        return dict(self.payload)


def test_compute_summary_matching_metrics_matching_types_only(monkeypatch, tmp_path: Path) -> None:
    model_id = "test-model"
    extracted_issue = {
        "issue": {
            "issue_number": 1,
            "comments": [
                {
                    "artifacts": [
                        {"type": "problem_statement", "summary": "Human problem summary"},
                        {"type": "question", "summary": "Human question summary"},
                    ]
                }
            ],
        }
    }
    derived_issue = {
        "issue": {
            "issue_number": 1,
            "comments": [
                {
                    "artifacts": [
                        {"type": "problem_statement", "summary": "LLM problem summary"},
                        {"type": "proposed_solution", "summary": "LLM solution summary"},
                    ]
                }
            ],
        }
    }

    write_json(tmp_path / "extractions" / "octo__repo" / "issue_1.json", extracted_issue)
    write_json(tmp_path / "derived" / model_id / "octo__repo" / "issue_1.json", derived_issue)

    monkeypatch.setattr(
        summary_matching,
        "_build_scorers",
        lambda **kwargs: {
            "codebert": _StaticScorer({"cosine": 0.7}),
            "bertscore": _StaticScorer({"precision": 0.6, "recall": 0.5, "f1": 0.545}),
            "bleurt": _StaticScorer({"score": 0.4}),
        },
    )

    report = compute_summary_matching_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
        model_id=model_id,
    )

    issue = report["per_issue"][0]
    assert issue["matching_type_count"] == 1
    assert issue["scores"]["codebert"]["cosine"] == 0.7
    assert issue["scores"]["bertscore"]["f1"] == 0.545
    assert issue["scores"]["bleurt"]["score"] == 0.4
    assert issue["coverage"]["matching_type_recall"] == 0.5
    assert issue["scores_with_unmatched_penalty"]["codebert"]["cosine"] == 0.35

    assert "matched_only_macro" in report["overall"]
    assert "coverage" in report["overall"]
    assert "penalized_overall" in report["overall"]


def test_compute_summary_matching_metrics_per_type_aggregation(monkeypatch, tmp_path: Path) -> None:
    model_id = "test-model"
    extracted_issue = {
        "issue": {
            "issue_number": 2,
            "comments": [{"artifacts": [{"type": "problem_statement", "summary": "A"}]}],
        }
    }
    derived_issue = {
        "issue": {
            "issue_number": 2,
            "comments": [{"artifacts": [{"type": "problem_statement", "summary": "B"}]}],
        }
    }

    write_json(tmp_path / "extractions" / "octo__repo" / "issue_2.json", extracted_issue)
    write_json(tmp_path / "derived" / model_id / "octo__repo" / "issue_2.json", derived_issue)

    monkeypatch.setattr(
        summary_matching,
        "_build_scorers",
        lambda **kwargs: {
            "codebert": _StaticScorer({"cosine": 0.8}),
            "bertscore": _StaticScorer({"precision": 0.9, "recall": 0.8, "f1": 0.847}),
            "bleurt": _StaticScorer({"score": 0.33}),
        },
    )

    report = compute_summary_matching_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
        model_id=model_id,
    )

    per_type = {item["type"]: item for item in report["per_type"]}
    assert per_type["problem_statement"]["matched_issue_count"] == 1
    assert per_type["problem_statement"]["scores"]["codebert"]["cosine"] == 0.8


def test_compute_summary_matching_metrics_unmatched_type_scores_are_zero(monkeypatch, tmp_path: Path) -> None:
    model_id = "test-model"
    extracted_issue = {
        "issue": {
            "issue_number": 3,
            "comments": [{"artifacts": [{"type": "question", "summary": "Human only summary"}]}],
        }
    }
    derived_issue = {
        "issue": {
            "issue_number": 3,
            "comments": [{"artifacts": [{"type": "problem_statement", "summary": "LLM only summary"}]}],
        }
    }

    write_json(tmp_path / "extractions" / "octo__repo" / "issue_3.json", extracted_issue)
    write_json(tmp_path / "derived" / model_id / "octo__repo" / "issue_3.json", derived_issue)

    monkeypatch.setattr(
        summary_matching,
        "_build_scorers",
        lambda **kwargs: {
            "codebert": _StaticScorer({"cosine": 0.8}),
            "bertscore": _StaticScorer({"precision": 0.9, "recall": 0.8, "f1": 0.847}),
            "bleurt": _StaticScorer({"score": 0.33}),
        },
    )

    report = compute_summary_matching_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
        model_id=model_id,
    )

    per_type = {item["type"]: item for item in report["per_type"]}
    assert per_type["question"]["matched_issue_count"] == 0
    assert per_type["question"]["scores"]["codebert"]["cosine"] == 0.0
    assert per_type["question"]["scores"]["bertscore"]["f1"] == 0.0
    assert per_type["question"]["scores"]["bleurt"]["score"] == 0.0
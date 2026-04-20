from __future__ import annotations

from pathlib import Path

from tbyc_dataset.metrics.tag_matching import compute_tag_matching_metrics
from tbyc_dataset.storage import write_json


def test_compute_tag_matching_metrics_matching_types_only(tmp_path: Path) -> None:
    model_id = "test-model"
    extracted_issue = {
        "issue": {
            "issue_number": 1,
            "comments": [
                {
                    "artifacts": [
                        {
                            "type": "problem_statement",
                            "summary": "Human summary",
                            "tags": ["bug", "runtime_error"],
                        },
                        {
                            "type": "question",
                            "summary": "Human question",
                            "tags": ["needs_info"],
                        },
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
                        {
                            "type": "problem_statement",
                            "summary": "LLM summary",
                            "tags": ["bug", "performance_issue"],
                        },
                        {
                            "type": "proposed_solution",
                            "summary": "LLM solution",
                            "tags": ["fix"],
                        },
                    ]
                }
            ],
        }
    }

    write_json(tmp_path / "extractions" / "octo__repo" / "issue_1.json", extracted_issue)
    write_json(tmp_path / "derived" / model_id / "octo__repo" / "issue_1.json", derived_issue)

    report = compute_tag_matching_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
        model_id=model_id,
    )

    issue = report["per_issue"][0]
    assert issue["matching_type_count"] == 1
    assert issue["pooled"]["intersection_count"] == 1
    assert issue["pooled"]["human_tag_count"] == 2
    assert issue["pooled"]["llm_tag_count"] == 2
    assert issue["pooled"]["precision"] == 0.5
    assert issue["pooled"]["recall"] == 0.5


def test_compute_tag_matching_metrics_per_type_counts(tmp_path: Path) -> None:
    model_id = "test-model"
    extracted_issue = {
        "issue": {
            "issue_number": 2,
            "comments": [
                {
                    "artifacts": [
                        {"type": "problem_statement", "summary": "S1", "tags": ["bug"]},
                    ]
                }
            ],
        }
    }
    derived_issue = {
        "issue": {
            "issue_number": 2,
            "comments": [
                {
                    "artifacts": [
                        {"type": "problem_statement", "summary": "S2", "tags": ["bug", "incorrect_behavior"]},
                    ]
                }
            ],
        }
    }

    write_json(tmp_path / "extractions" / "octo__repo" / "issue_2.json", extracted_issue)
    write_json(tmp_path / "derived" / model_id / "octo__repo" / "issue_2.json", derived_issue)

    report = compute_tag_matching_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
        model_id=model_id,
    )

    per_type = {item["type"]: item for item in report["per_type"]}
    assert per_type["problem_statement"]["compared_issue_count"] == 1
    assert per_type["problem_statement"]["precision"] == 0.5
    assert per_type["problem_statement"]["recall"] == 1.0
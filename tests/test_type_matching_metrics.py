from __future__ import annotations

from pathlib import Path

from tbyc_dataset.metrics.type_matching import compute_type_matching_metrics
from tbyc_dataset.storage import write_json


def test_compute_type_matching_metrics_issue_level_set_overlap(tmp_path: Path) -> None:
    model_id = "test-model"
    extracted_issue = {
        "issue": {
            "issue_number": 1,
            "comments": [
                {"artifacts": [{"type": "problem_statement"}, {"type": "question"}]},
                {"artifacts": [{"type": "proposed_solution"}]},
            ]
        }
    }
    derived_issue = {
        "issue": {
            "issue_number": 1,
            "comments": [
                {"artifacts": [{"type": "problem_statement"}]},
                {"artifacts": [{"type": "alternative_solution"}]},
            ]
        }
    }

    write_json(tmp_path / "extractions" / "octo__repo" / "issue_1.json", extracted_issue)
    write_json(tmp_path / "derived" / model_id / "octo__repo" / "issue_1.json", derived_issue)

    report = compute_type_matching_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
        model_id=model_id,
    )

    assert report["issue_count"] == 1
    issue = report["per_issue"][0]
    assert issue["intersection_count"] == 1
    assert issue["union_count"] == 4
    assert issue["precision"] == 0.5
    assert round(issue["recall"], 6) == round(1.0 / 3.0, 6)
    assert issue["jaccard"] == 0.25

    per_type = {item["type"]: item for item in report["per_type"]}
    assert per_type["problem_statement"]["precision"] == 1.0
    assert per_type["problem_statement"]["recall"] == 1.0
    assert per_type["problem_statement"]["f1"] == 1.0
    assert per_type["question"]["recall"] == 0.0
    assert per_type["alternative_solution"]["precision"] == 0.0


def test_compute_type_matching_metrics_handles_missing_side_as_empty_set(tmp_path: Path) -> None:
    model_id = "test-model"
    extracted_issue = {
        "issue": {
            "issue_number": 2,
            "comments": [{"artifacts": [{"type": "problem_statement"}]}],
        }
    }
    write_json(tmp_path / "extractions" / "octo__repo" / "issue_2.json", extracted_issue)

    report = compute_type_matching_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
        model_id=model_id,
    )

    issue = report["per_issue"][0]
    assert issue["llm_type_count"] == 0
    assert issue["precision"] == 1.0
    assert issue["recall"] == 0.0
    assert issue["f1"] == 0.0
    assert issue["jaccard"] == 0.0

    per_type = {item["type"]: item for item in report["per_type"]}
    assert per_type["problem_statement"]["llm_issue_count"] == 0
    assert per_type["problem_statement"]["human_issue_count"] == 1
    assert per_type["problem_statement"]["f1"] == 0.0
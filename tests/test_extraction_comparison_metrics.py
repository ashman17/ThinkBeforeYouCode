from __future__ import annotations

from pathlib import Path

from tbyc_dataset.metrics.extraction_comparison import compute_extraction_comparison_metrics
from tbyc_dataset.storage import write_json


def test_compute_extraction_comparison_metrics_scores_llm_against_regex(tmp_path: Path) -> None:
    llm_issue = {
        "issue": {
            "issue_number": 1,
            "artifact_count": 2,
            "comments": [
                {
                    "artifacts": [
                        {
                            "type": "problem_statement",
                            "summary": "Build fails on Linux",
                            "tags": ["bug", "linux"],
                            "metadata": {"problem_kind": "bug", "affected_area": "build", "user_or_system_impact": "user"},
                        },
                        {
                            "type": "question",
                            "summary": "Can you share the logs?",
                            "tags": ["log_error"],
                            "metadata": {"question_kind": "clarification", "target_area": "logs", "urgency": "unknown"},
                        },
                    ]
                }
            ],
        }
    }
    regex_issue = {
        "issue": {
            "issue_number": 1,
            "artifact_count": 2,
            "comments": [
                {
                    "artifacts": [
                        {
                            "type": "problem_statement",
                            "summary": "Build fails on Linux",
                            "tags": ["bug", "linux"],
                            "metadata": {"problem_kind": "bug", "affected_area": "build", "user_or_system_impact": "user"},
                        },
                        {
                            "type": "proposed_solution",
                            "summary": "Add a fallback",
                            "tags": ["fix_proposed"],
                            "metadata": {"solution_kind": "code_change", "target_area": "build", "expected_benefit": "stability"},
                        },
                    ]
                }
            ],
        }
    }

    write_json(tmp_path / "extractions" / "octo__repo" / "issue_1.json", llm_issue)
    write_json(tmp_path / "extractions_regex" / "octo__repo" / "issue_1.json", regex_issue)

    report = compute_extraction_comparison_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
    )

    assert report["issue_count"] == 1
    issue = report["per_issue"][0]
    assert issue["type"]["intersection_count"] == 1
    assert issue["type"]["union_count"] == 3
    assert issue["type"]["precision"] == 0.5
    assert issue["type"]["recall"] == 0.5
    assert issue["tag"]["f1"] == 1.0
    assert report["macro_average"]["type"]["f1"] == 0.5


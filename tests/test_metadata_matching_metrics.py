from __future__ import annotations

from pathlib import Path

from tbyc_dataset.metrics.metadata_matching import compute_metadata_matching_metrics
from tbyc_dataset.storage import write_json


def test_compute_metadata_matching_metrics_fuzzy_phrase_match(tmp_path: Path) -> None:
    model_id = "test-model"
    extracted_issue = {
        "issue": {
            "issue_number": 1,
            "comments": [
                {
                    "artifacts": [
                        {
                            "type": "problem_statement",
                            "metadata": {
                                "affected_area": "Kubelet checkpoint handling",
                                "problem_kind": "Incorrect behavior of a method",
                            },
                        }
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
                            "metadata": {
                                "affected_area": "Kubelet internal checkpoint handling",
                                "problem_kind": "Method behaves incorrectly",
                            },
                        }
                    ]
                }
            ],
        }
    }

    write_json(tmp_path / "extractions" / "octo__repo" / "issue_1.json", extracted_issue)
    write_json(tmp_path / "derived" / model_id / "octo__repo" / "issue_1.json", derived_issue)

    report = compute_metadata_matching_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
        model_id=model_id,
        similarity_threshold=0.45,
    )

    assert report["issue_count"] == 1
    issue = report["per_issue"][0]
    assert issue["issue_number"] == 1
    assert issue["pooled"]["precision"] == 1.0
    assert issue["pooled"]["recall"] == 1.0
    assert issue["pooled"]["f1"] == 1.0
    assert issue["pooled"]["soft_f1"] > 0.6


def test_compute_metadata_matching_metrics_handles_missing_side(tmp_path: Path) -> None:
    model_id = "test-model"
    extracted_issue = {
        "issue": {
            "issue_number": 2,
            "comments": [
                {
                    "artifacts": [
                        {
                            "type": "problem_statement",
                            "metadata": {
                                "affected_area": "Kubelet internals",
                            },
                        }
                    ]
                }
            ],
        }
    }
    write_json(tmp_path / "extractions" / "octo__repo" / "issue_2.json", extracted_issue)

    report = compute_metadata_matching_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
        model_id=model_id,
    )

    issue = report["per_issue"][0]
    assert issue["pooled"]["human_value_count"] == 1
    assert issue["pooled"]["llm_value_count"] == 0
    assert issue["pooled"]["precision"] == 1.0
    assert issue["pooled"]["recall"] == 0.0
    assert issue["pooled"]["f1"] == 0.0


def test_compute_metadata_matching_metrics_all_mode(tmp_path: Path) -> None:
    model_id = "test-model"
    extracted_issue = {
        "issue": {
            "issue_number": 3,
            "comments": [
                {
                    "artifacts": [
                        {
                            "type": "problem_statement",
                            "metadata": {
                                "affected_area": "Kubelet checkpoint handling",
                            },
                        }
                    ]
                }
            ],
        }
    }
    derived_issue = {
        "issue": {
            "issue_number": 3,
            "comments": [
                {
                    "artifacts": [
                        {
                            "type": "problem_statement",
                            "metadata": {
                                "affected_area": "Kubelet internal checkpoint handling",
                            },
                        }
                    ]
                }
            ],
        }
    }

    write_json(tmp_path / "extractions" / "octo__repo" / "issue_3.json", extracted_issue)
    write_json(tmp_path / "derived" / model_id / "octo__repo" / "issue_3.json", derived_issue)

    report = compute_metadata_matching_metrics(
        owner="octo",
        repo="repo",
        output_root=str(tmp_path),
        model_id=model_id,
        similarity_metric="all",
    )

    assert report["metric"] == "metadata_phrase_matching_all"
    assert "summary_by_metric" in report
    assert "token_f1" in report["summary_by_metric"]
    assert "max_all" in report["summary_by_metric"]
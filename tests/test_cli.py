from __future__ import annotations

import json

from tbyc_dataset import cli


def test_retrieve_code_chunks_prints_manifest_only(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(
        cli.CodeRetrievalPipeline,
        "run",
        lambda self, owner, repo, output_dir: {
            "manifest": {"repository": f"{owner}/{repo}", "index_dir": str(tmp_path / "index")},
            "issues": [{"issue": {"number": 1}, "results": [{"chunk_id": "c1"}]}],
        },
    )

    cli.main(
        [
            "retrieve-code-chunks",
            "--repo",
            "octo/repo",
            "--output-root",
            str(tmp_path),
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert sorted(payload.keys()) == ["manifest"]
    assert payload["manifest"]["repository"] == "octo/repo"


def test_generate_issue_thoughts_prints_summary(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(
        cli.IssueThoughtPipeline,
        "run",
        lambda self, owner, repo: {
            "repository": f"{owner}/{repo}",
            "response_dir": str(tmp_path / "responses" / "octo__repo"),
            "issue_count": 2,
            "responses": [{"issue_number": 1}, {"issue_number": 2}],
        },
    )

    cli.main(
        [
            "generate-issue-thoughts",
            "--repo",
            "octo/repo",
            "--output-root",
            str(tmp_path),
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["repository"] == "octo/repo"
    assert payload["issue_count"] == 2


def test_compute_type_metrics_prints_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "compute_type_matching_metrics",
        lambda owner, repo, output_root, model_id, issue_number=None: {
            "repository": f"{owner}/{repo}",
            "model_id": model_id,
            "issue_count": 3,
            "metric": "aggregated_type_matching",
            "overall": {"precision": 0.5, "recall": 0.4, "f1": 0.4444, "jaccard": 0.3},
            "macro_average": {"precision": 0.6, "recall": 0.5, "f1": 0.54, "jaccard": 0.4},
        },
    )

    cli.main(
        [
            "compute-type-metrics",
            "--repo",
            "octo/repo",
            "--output-root",
            "data",
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["repository"] == "octo/repo"
    assert payload["model_id"] == "qwen2.5:7b-instruct"
    assert payload["metric"] == "aggregated_type_matching"
    assert payload["issue_count"] == 3


def test_compute_metadata_metrics_prints_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "compute_metadata_matching_metrics",
        lambda owner, repo, output_root, model_id, issue_number=None, similarity_threshold=0.82, similarity_metric="max_all": {
            "repository": f"{owner}/{repo}",
            "model_id": model_id,
            "issue_count": 2,
            "metric": "metadata_phrase_matching",
            "similarity": {"method": similarity_metric, "threshold": similarity_threshold},
            "overall": {"pooled": {"precision": 0.7, "recall": 0.6, "f1": 0.646, "jaccard": 0.5}},
            "macro_average": {"precision": 0.72, "recall": 0.65, "f1": 0.68, "jaccard": 0.52},
        },
    )

    cli.main(
        [
            "compute-metadata-metrics",
            "--repo",
            "octo/repo",
            "--output-root",
            "data",
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["repository"] == "octo/repo"
    assert payload["model_id"] == "qwen2.5:7b-instruct"
    assert payload["metric"] == "metadata_phrase_matching"
    assert payload["issue_count"] == 2


def test_compute_tag_metrics_prints_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "compute_tag_matching_metrics",
        lambda owner, repo, output_root, model_id, issue_number=None: {
            "repository": f"{owner}/{repo}",
            "model_id": model_id,
            "issue_count": 4,
            "metric": "matching_type_tag_overlap",
            "overall": {"precision": 0.5, "recall": 0.45, "f1": 0.47, "jaccard": 0.31},
            "macro_average": {"precision": 0.55, "recall": 0.5, "f1": 0.52, "jaccard": 0.35},
        },
    )

    cli.main(
        [
            "compute-tag-metrics",
            "--repo",
            "octo/repo",
            "--output-root",
            "data",
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["repository"] == "octo/repo"
    assert payload["model_id"] == "qwen2.5:7b-instruct"
    assert payload["metric"] == "matching_type_tag_overlap"
    assert payload["issue_count"] == 4


def test_compute_summary_metrics_prints_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "compute_summary_matching_metrics",
        lambda owner, repo, output_root, model_id, issue_number=None, codebert_model="microsoft/codebert-base", bertscore_model="microsoft/codebert-base", bleurt_model="Elron/bleurt-base-512": {
            "repository": f"{owner}/{repo}",
            "model_id": model_id,
            "issue_count": 5,
            "metric": "summary_similarity_matching_types",
            "models": {
                "codebert": codebert_model,
                "bertscore": bertscore_model,
                "bleurt": bleurt_model,
            },
            "overall": {
                "codebert": {"cosine": 0.44},
                "bertscore": {"precision": 0.51, "recall": 0.49, "f1": 0.5},
                "bleurt": {"score": 0.12},
            },
        },
    )

    cli.main(
        [
            "compute-summary-metrics",
            "--repo",
            "octo/repo",
            "--output-root",
            "data",
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["repository"] == "octo/repo"
    assert payload["model_id"] == "qwen2.5:7b-instruct"
    assert payload["metric"] == "summary_similarity_matching_types"
    assert payload["issue_count"] == 5

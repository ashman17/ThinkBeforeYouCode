from __future__ import annotations

import json
from pathlib import Path

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
        lambda owner, repo, output_root, model_id, issue_number=None, codebert_model="microsoft/codebert-base", bertscore_model="microsoft/codebert-base", bleurt_model="Elron/bleurt-base-512", bleurt_postprocess="sigmoid", bleurt_clip_min=0.0, bleurt_sigmoid_temperature=2.0, bleurt_sigmoid_bias=0.0: {
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


def test_compute_extraction_comparison_prints_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "compute_extraction_comparison_metrics",
        lambda owner, repo, output_root, issue_number=None, similarity_threshold=0.82, similarity_metric="max_all", codebert_model="microsoft/codebert-base", bertscore_model="microsoft/codebert-base", bleurt_model="Elron/bleurt-base-512", bleurt_postprocess="sigmoid", bleurt_clip_min=0.0, bleurt_sigmoid_temperature=2.0, bleurt_sigmoid_bias=0.0: {
            "repository": f"{owner}/{repo}",
            "issue_count": 3,
            "reference_side": "regex",
            "candidate_side": "llm",
            "macro_average": {
                "type": {"f1": 0.5},
                "tag": {"f1": 0.4},
                "metadata": {"f1": 0.6, "soft_f1": 0.7},
                "summary": {
                    "bertscore": {"f1": 0.8},
                    "codebert": {"cosine": 0.3},
                    "bleurt": {"score": 0.2},
                },
            },
            "overall": {"counts": {"llm_artifact_count": 10, "regex_artifact_count": 12}},
        },
    )

    cli.main(
        [
            "compute-extraction-comparison",
            "--repo",
            "octo/repo",
            "--output-root",
            "data",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["repository"] == "octo/repo"
    assert payload["reference_side"] == "regex"
    assert payload["candidate_side"] == "llm"
    assert payload["overall"]["type_f1"] == 0.5
    assert payload["issue_count"] == 3


def test_compute_all_metrics_runs_all_targets_without_filters(monkeypatch, capsys, tmp_path) -> None:
    derived_root = tmp_path / "derived"
    (derived_root / "modelA" / "octo__repo").mkdir(parents=True)
    (derived_root / "modelA" / "octo__repo" / "issue_1.json").write_text("{}", encoding="utf-8")
    (derived_root / "modelB" / "octo__repo").mkdir(parents=True)
    (derived_root / "modelB" / "octo__repo" / "issue_1.json").write_text("{}", encoding="utf-8")

    calls = []

    def _type_metric(owner, repo, output_root, model_id, issue_number=None):
        calls.append(("type", owner, repo, model_id))
        return {"metric": "aggregated_type_matching", "issue_count": 1}

    def _metadata_metric(
        owner,
        repo,
        output_root,
        model_id,
        issue_number=None,
        similarity_threshold=0.82,
        similarity_metric="max_all",
    ):
        calls.append(("metadata", owner, repo, model_id))
        return {"metric": "metadata_phrase_matching", "issue_count": 1}

    def _tag_metric(owner, repo, output_root, model_id, issue_number=None):
        calls.append(("tag", owner, repo, model_id))
        return {"metric": "matching_type_tag_overlap", "issue_count": 1}

    def _summary_metric(
        owner,
        repo,
        output_root,
        model_id,
        issue_number=None,
        codebert_model="microsoft/codebert-base",
        bertscore_model="microsoft/codebert-base",
        bleurt_model="Elron/bleurt-base-512",
        bleurt_postprocess="sigmoid",
        bleurt_clip_min=0.0,
        bleurt_sigmoid_temperature=2.0,
        bleurt_sigmoid_bias=0.0,
    ):
        calls.append(("summary", owner, repo, model_id))
        return {"metric": "summary_similarity_matching_types", "issue_count": 1}

    monkeypatch.setattr(cli, "compute_type_matching_metrics", _type_metric)
    monkeypatch.setattr(cli, "compute_metadata_matching_metrics", _metadata_metric)
    monkeypatch.setattr(cli, "compute_tag_matching_metrics", _tag_metric)
    monkeypatch.setattr(cli, "compute_summary_matching_metrics", _summary_metric)

    cli.main([
        "compute-all-metrics",
        "--output-root",
        str(tmp_path),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["target_count"] == 2
    assert payload["metric_run_count"] == 8
    assert payload["failed_metric_count"] == 0
    assert len(calls) == 8


def test_compute_all_metrics_repo_filter_runs_all_models_for_repo(monkeypatch, capsys, tmp_path) -> None:
    derived_root = tmp_path / "derived"
    (derived_root / "modelA" / "octo__repo").mkdir(parents=True)
    (derived_root / "modelA" / "octo__repo" / "issue_1.json").write_text("{}", encoding="utf-8")
    (derived_root / "modelB" / "octo__repo").mkdir(parents=True)
    (derived_root / "modelB" / "octo__repo" / "issue_1.json").write_text("{}", encoding="utf-8")
    (derived_root / "modelA" / "other__project").mkdir(parents=True)
    (derived_root / "modelA" / "other__project" / "issue_1.json").write_text("{}", encoding="utf-8")

    calls = []

    def _metric_stub(owner, repo, output_root, model_id, issue_number=None, **kwargs):
        calls.append((owner, repo, model_id))
        return {"metric": "ok", "issue_count": 1}

    monkeypatch.setattr(cli, "compute_type_matching_metrics", _metric_stub)
    monkeypatch.setattr(cli, "compute_metadata_matching_metrics", _metric_stub)
    monkeypatch.setattr(cli, "compute_tag_matching_metrics", _metric_stub)
    monkeypatch.setattr(cli, "compute_summary_matching_metrics", _metric_stub)

    cli.main([
        "compute-all-metrics",
        "--output-root",
        str(tmp_path),
        "--repo",
        "octo/repo",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["target_count"] == 2
    assert payload["metric_run_count"] == 8
    assert all(call[0] == "octo" and call[1] == "repo" for call in calls)


def test_compute_all_metrics_one_shot_runs_metrics_and_leaderboard(monkeypatch, capsys, tmp_path) -> None:
    observed = {
        "compute_called": False,
        "leaderboard_called": False,
    }

    def _compute_all_metrics_stub(
        *,
        output_root,
        model_id_filter,
        repo_filter,
        issue_number,
        similarity_threshold,
        similarity_metric,
        codebert_model,
        bertscore_model,
        bleurt_model,
        bleurt_postprocess,
        bleurt_clip_min,
        bleurt_sigmoid_temperature,
        bleurt_sigmoid_bias,
    ):
        observed["compute_called"] = True
        assert output_root == tmp_path
        assert model_id_filter is None
        assert repo_filter is None
        assert issue_number is None
        return {
            "target_count": 3,
            "metric_run_count": 12,
            "succeeded_metric_count": 12,
            "failed_metric_count": 0,
            "targets": [
                {"repository": "octo/repo", "model_id": "modelA", "metrics": {}, "errors": []},
            ],
        }

    def _build_leaderboard_stub(
        *,
        output_root,
        repo_filter,
        model_id_filter,
        rrf_k,
        points_max,
        points_step,
    ):
        observed["leaderboard_called"] = True
        assert output_root == tmp_path
        assert repo_filter is None
        assert model_id_filter is None
        assert rrf_k == 60
        assert points_max == 100
        assert points_step == 1
        return {
            "metric": "rank_fusion_leaderboard",
            "repo_count": 2,
            "model_count": 4,
            "per_repo": [],
            "all_repos_combined": {"leaderboard": []},
        }

    monkeypatch.setattr(cli, "_compute_all_metrics", _compute_all_metrics_stub)
    monkeypatch.setattr(cli, "_build_rank_fusion_leaderboard", _build_leaderboard_stub)

    cli.main([
        "compute-all-metrics-one-shot",
        "--output-root",
        str(tmp_path),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert observed["compute_called"]
    assert observed["leaderboard_called"]
    assert payload["metric"] == "all_metrics_one_shot"
    assert payload["scope"] == "all_repositories_all_models_all_issues"
    assert payload["metric_runs"]["target_count"] == 3
    assert payload["leaderboard"]["repo_count"] == 2
    assert payload["leaderboard"]["model_count"] == 4


def _write_minimal_metric_files(base: Path, *, type_f1: float, meta_f1: float, meta_soft_f1: float, tag_f1: float, summary_f1: float, summary_bleurt: float, summary_codebert: float) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "type_matching.json").write_text(
        json.dumps({"macro_average": {"f1": type_f1}}),
        encoding="utf-8",
    )
    (base / "metadata_matching.json").write_text(
        json.dumps({"macro_average": {"f1": meta_f1, "soft_f1": meta_soft_f1}}),
        encoding="utf-8",
    )
    (base / "tag_matching.json").write_text(
        json.dumps({"overall": {"f1": tag_f1}}),
        encoding="utf-8",
    )
    (base / "summary_matching.json").write_text(
        json.dumps(
            {
                "overall": {
                    "all_issues_macro_with_unmatched_penalty": {
                        "bertscore": {"f1": summary_f1},
                        "bleurt": {"score": summary_bleurt},
                        "codebert": {"cosine": summary_codebert},
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_build_leaderboard_outputs_per_repo_and_combined(capsys, tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    _write_minimal_metric_files(
        metrics_root / "modelA" / "octo__repo",
        type_f1=0.8,
        meta_f1=0.6,
        meta_soft_f1=0.7,
        tag_f1=0.75,
        summary_f1=0.82,
        summary_bleurt=0.55,
        summary_codebert=0.91,
    )
    _write_minimal_metric_files(
        metrics_root / "modelB" / "octo__repo",
        type_f1=0.4,
        meta_f1=0.3,
        meta_soft_f1=0.35,
        tag_f1=0.5,
        summary_f1=0.55,
        summary_bleurt=0.3,
        summary_codebert=0.7,
    )

    cli.main(["build-leaderboard", "--output-root", str(tmp_path)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["metric"] == "rank_fusion_leaderboard"
    assert payload["repo_count"] == 1
    assert payload["model_count"] == 2
    assert payload["per_repo"][0]["repository"] == "octo/repo"
    assert payload["per_repo"][0]["leaderboard"][0]["model_id"] == "modelA"
    assert payload["all_repos_combined"]["leaderboard"][0]["score"] >= payload["all_repos_combined"]["leaderboard"][1]["score"]


def test_build_leaderboard_repo_filter(capsys, tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics"
    _write_minimal_metric_files(
        metrics_root / "modelA" / "octo__repo",
        type_f1=0.8,
        meta_f1=0.6,
        meta_soft_f1=0.7,
        tag_f1=0.75,
        summary_f1=0.82,
        summary_bleurt=0.55,
        summary_codebert=0.91,
    )
    _write_minimal_metric_files(
        metrics_root / "modelA" / "other__project",
        type_f1=0.1,
        meta_f1=0.1,
        meta_soft_f1=0.1,
        tag_f1=0.1,
        summary_f1=0.1,
        summary_bleurt=0.1,
        summary_codebert=0.1,
    )

    cli.main([
        "build-leaderboard",
        "--output-root",
        str(tmp_path),
        "--repo",
        "octo/repo",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["repo_count"] == 1
    assert payload["per_repo"][0]["repository"] == "octo/repo"


def test_visualize_metrics_prints_manifest(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "generate_metrics_visualizations",
        lambda output_root, graphs_root=None, repo=None, model_id=None, points_max=100, points_step=1: {
            "output_root": output_root,
            "graphs_root": str(tmp_path / "graphs"),
            "repo_filter": repo,
            "model_filter": model_id,
            "graph_count": 3,
            "files": ["a.png", "b.png", "c.png"],
        },
    )

    cli.main([
        "visualize-metrics",
        "--output-root",
        str(tmp_path),
        "--repo",
        "octo/repo",
        "--model-id",
        "modelA",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["graph_count"] == 3
    assert payload["repo_filter"] == "octo/repo"
    assert payload["model_filter"] == "modelA"

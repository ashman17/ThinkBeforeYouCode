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

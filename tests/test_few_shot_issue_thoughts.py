from __future__ import annotations

import json

from tbyc_dataset import cli
from tbyc_dataset.evaluation.prompt import build_issue_thought_prompt


def test_build_issue_thought_prompt_includes_few_shot_examples() -> None:
    prompt = build_issue_thought_prompt(
        {"number": 10, "title": "Current issue", "body": "Current body"},
        include_context=False,
        context_blocks=[],
        few_shot_examples=[
            {
                "issue_number": 1,
                "title": "Example issue",
                "body": "Example body",
                "response": "[Problem Statement] Example artifact",
            }
        ],
    )

    assert "Here are examples from the same repository" in prompt
    assert "Example 1" in prompt
    assert "Response:" in prompt
    assert "[Problem Statement] Example artifact" in prompt


def test_generate_issue_thoughts_few_shot_writes_to_responses_few_shot(monkeypatch, capsys, tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "octo__repo" / "issues"
    raw_dir.mkdir(parents=True)
    (raw_dir / "issue_7.json").write_text(
        json.dumps({"number": 7, "title": "Issue title", "body": "Issue body", "createdAt": "2024-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    eval_dir = tmp_path / "evaluation" / "octo__repo" / "results"
    eval_dir.mkdir(parents=True)
    (eval_dir / "issue_7.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    processed_dir = tmp_path / "processed" / "octo__repo"
    processed_dir.mkdir(parents=True)
    (processed_dir / "curated.jsonl").write_text("", encoding="utf-8")
    extraction_dir = tmp_path / "extractions" / "octo__repo"
    extraction_dir.mkdir(parents=True)
    (extraction_dir / "issue_1.json").write_text(
        json.dumps(
            {
                "issue": {
                    "issue_number": 1,
                    "comments": [
                        {
                            "artifacts": [
                                {"type": "problem_statement", "summary": "Example problem", "tags": [], "metadata": {}}
                            ]
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "tbyc_dataset.evaluation.thoughts._build_issue_thought_completion_runner",
        lambda settings: ("ollama", settings.model_id, lambda prompt: "[Problem Statement] Synthetic reply"),
    )

    cli.main(
        [
            "generate-issue-thoughts-few-shot",
            "--repo",
            "octo/repo",
            "--output-root",
            str(tmp_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["repository"] == "octo/repo"
    response_path = tmp_path / "responses_few-shot" / "qwen2.5:14b" / "octo__repo" / "issue_7.json"
    response_payload = json.loads(response_path.read_text(encoding="utf-8"))
    assert response_payload["prompt_flags"]["few_shot_from_extractions"] is True


def test_generate_issue_thoughts_regex_alias_writes_to_responses_few_shot(monkeypatch, capsys, tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "octo__repo" / "issues"
    raw_dir.mkdir(parents=True)
    (raw_dir / "issue_7.json").write_text(
        json.dumps({"number": 7, "title": "Issue title", "body": "Issue body", "createdAt": "2024-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    eval_dir = tmp_path / "evaluation" / "octo__repo" / "results"
    eval_dir.mkdir(parents=True)
    (eval_dir / "issue_7.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    processed_dir = tmp_path / "processed" / "octo__repo"
    processed_dir.mkdir(parents=True)
    (processed_dir / "curated.jsonl").write_text("", encoding="utf-8")
    extraction_dir = tmp_path / "extractions" / "octo__repo"
    extraction_dir.mkdir(parents=True)
    (extraction_dir / "issue_1.json").write_text(
        json.dumps(
            {
                "issue": {
                    "issue_number": 1,
                    "comments": [
                        {
                            "artifacts": [
                                {"type": "problem_statement", "summary": "Example problem", "tags": [], "metadata": {}}
                            ]
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "tbyc_dataset.evaluation.thoughts._build_issue_thought_completion_runner",
        lambda settings: ("ollama", settings.model_id, lambda prompt: "[Problem Statement] Synthetic reply"),
    )

    cli.main(
        [
            "generate-issue-thoughts-regex",
            "--repo",
            "octo/repo",
            "--output-root",
            str(tmp_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["repository"] == "octo/repo"
    response_path = tmp_path / "responses_few-shot" / "qwen2.5:14b" / "octo__repo" / "issue_7.json"
    response_payload = json.loads(response_path.read_text(encoding="utf-8"))
    assert response_payload["prompt_flags"]["few_shot_from_extractions"] is True

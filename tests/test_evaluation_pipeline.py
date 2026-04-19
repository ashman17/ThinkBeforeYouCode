from __future__ import annotations

import json
from pathlib import Path

from tbyc_dataset.evaluation.pipeline import (
    CodeRetrievalPipeline,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)


def test_load_issues_sorts_by_created_at(tmp_path: Path) -> None:
    issue_dir = tmp_path / "issues"
    issue_dir.mkdir(parents=True)
    (issue_dir / "issue_2.json").write_text(
        json.dumps(
            {
                "number": 2,
                "title": "Second",
                "body": "Body 2",
                "createdAt": "2024-01-02T00:00:00Z",
                "url": "https://example.com/2",
            }
        ),
        encoding="utf-8",
    )
    (issue_dir / "issue_1.json").write_text(
        json.dumps(
            {
                "number": 1,
                "title": "First",
                "body": "Body 1",
                "createdAt": "2024-01-01T00:00:00Z",
                "url": "https://example.com/1",
            }
        ),
        encoding="utf-8",
    )

    pipeline = CodeRetrievalPipeline(cache_dir=str(tmp_path))
    issues = pipeline._load_issues(issue_dir)

    assert [issue.number for issue in issues] == [1, 2]
    assert issues[0].query == "First\n\nBody 1"


def test_extract_python_chunks_uses_function_and_class_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "class Greeter:\n"
        "    def hello(self):\n"
        "        return 'hi'\n"
        "\n"
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    pipeline = CodeRetrievalPipeline(cache_dir=str(tmp_path))
    chunks = pipeline._chunk_file(source, "sample.py")

    assert [chunk.symbol_name for chunk in chunks] == ["Greeter", "hello", "add"]
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 6


def test_fallback_line_chunks_respects_overlap(tmp_path: Path) -> None:
    source = tmp_path / "sample.js"
    source.write_text("\n".join(f"line {index}" for index in range(1, 8)), encoding="utf-8")

    pipeline = CodeRetrievalPipeline(cache_dir=str(tmp_path), chunk_size=3, chunk_overlap=1)
    chunks = pipeline._fallback_line_chunks(
        relative_path="sample.txt",
        language="txt",
        lines=source.read_text(encoding="utf-8").splitlines(),
    )

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 3), (3, 5), (5, 7)]


def test_rrf_combines_rankings() -> None:
    fused = reciprocal_rank_fusion(
        [
            [(2, 10.0), (1, 9.0), (0, 8.0)],
            [(1, 0.9), (2, 0.8), (3, 0.7)],
        ],
        rrf_k=60,
        top_n=3,
    )

    assert [doc_id for doc_id, _ in fused] == [2, 1, 0]


def test_bm25_tokenizer_preserves_identifiers() -> None:
    tokens = tokenize_for_bm25("RepoMappingManifestAction foo_bar Bazel8")

    assert tokens == ["repomappingmanifestaction", "foo_bar", "bazel8"]

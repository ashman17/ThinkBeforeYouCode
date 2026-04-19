from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet


DEFAULT_SOURCE_EXTENSIONS = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".cxx",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".m",
        ".mm",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".scala",
        ".swift",
        ".ts",
        ".tsx",
    }
)


@dataclass(frozen=True)
class RetrievalSettings:
    cache_dir: Path
    embedding_model: str = "microsoft/codebert-base"
    top_n: int = 20
    bm25_top_k: int = 50
    dense_top_k: int = 50
    chunk_size: int = 50
    chunk_overlap: int = 10
    rrf_k: int = 60
    max_tokens: int = 512
    source_extensions: FrozenSet[str] = field(default_factory=lambda: DEFAULT_SOURCE_EXTENSIONS)

    def repo_cache_root(self, owner: str, repo: str) -> Path:
        return self.cache_dir / "repos" / f"{owner}__{repo}"

    def evaluation_root(self, owner: str, repo: str) -> Path:
        return self.cache_dir / "evaluation" / f"{owner}__{repo}"

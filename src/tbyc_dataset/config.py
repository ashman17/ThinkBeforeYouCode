from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class GitHubSettings:
    token: str
    endpoint: str = "https://api.github.com/graphql"
    issue_page_size: int = 25
    comment_page_size: int = 100
    timeline_page_size: int = 100
    max_retries: int = 4
    request_timeout_seconds: int = 60
    sleep_between_requests_seconds: float = 0.0


@dataclass(frozen=True)
class PipelineSettings:
    output_root: Path
    min_comments: int = 0
    max_comments: Optional[int] = None

    @property
    def raw_root(self) -> Path:
        return self.output_root / "raw"

    @property
    def processed_root(self) -> Path:
        return self.output_root / "processed"


def github_settings_from_env() -> GitHubSettings:
    # Load .env from the current working directory (and parent dirs) if present.
    # Existing environment variables remain unchanged.
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise ValueError(
            "GITHUB_TOKEN is required. Set it in your shell or place it in a .env file."
        )
    return GitHubSettings(token=token)


def pipeline_settings(
    output_root: str,
    min_comments: int,
    max_comments: Optional[int] = None,
) -> PipelineSettings:
    return PipelineSettings(
        output_root=Path(output_root),
        min_comments=min_comments,
        max_comments=max_comments,
    )

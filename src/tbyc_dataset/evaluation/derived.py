from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ollama import Client
from pydantic import BaseModel, Field, ValidationError

from tbyc_dataset.extraction.pipeline import Artifact, _normalize_artifact
from tbyc_dataset.extraction.prompt import METADATA_RULES_TEXT, PROMPT_VERSION, SYSTEM_PROMPT
from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import ensure_directory, read_json, write_json


LOGGER = logging.getLogger(__name__)

TOPIC_SPLIT_PATTERN = re.compile(r"\[[^\]\n]{1,120}\]")


class CommentArtifactBatch(BaseModel):
    comment_index: int
    artifacts: List[Artifact] = Field(default_factory=list)


class BatchArtifactEnvelope(BaseModel):
    comments: List[CommentArtifactBatch] = Field(default_factory=list)


@dataclass(frozen=True)
class DerivedExtractionSettings:
    model_id: str = "qwen2.5:14b"
    model_url: str = "http://localhost:11434"
    num_ctx: int = 32768
    issue_number: Optional[int] = None
    skip_existing: bool = True


def extract_derived_artifacts_from_responses(
    *,
    owner: str,
    repo: str,
    output_root: str,
    settings: DerivedExtractionSettings,
) -> Dict[str, Any]:
    repo_ref = RepositoryRef(owner=owner, name=repo)
    base_dir = Path(output_root)
    responses_dir = base_dir / "responses" / repo_ref.fs_slug
    derived_dir = base_dir / "derived" / repo_ref.fs_slug
    ensure_directory(derived_dir)

    response_paths = sorted(responses_dir.glob("issue_*.json"))
    if settings.issue_number is not None:
        response_paths = [path for path in response_paths if path.name == f"issue_{settings.issue_number}.json"]

    if not response_paths:
        raise FileNotFoundError(f"No response payloads found under {responses_dir}")

    LOGGER.info("loaded response issues=%s from %s", len(response_paths), responses_dir)
    client = Client(host=settings.model_url)

    issue_count = 0
    total_comments = 0
    failed_comments = 0
    total_artifacts = 0

    for path in _progress_iter(
        response_paths,
        desc="Extracting derived artifacts",
        unit="issue",
        total=len(response_paths),
    ):
        payload = read_json(path)
        issue_number = int(payload.get("issue_number"))
        output_path = derived_dir / f"issue_{issue_number}.json"

        if settings.skip_existing and output_path.exists():
            existing = read_json(output_path)
            issue_entry = existing.get("issue", {}) if isinstance(existing, dict) else {}
            comments = issue_entry.get("comments", []) if isinstance(issue_entry, dict) else []
            issue_count += 1
            total_comments += len(comments)
            failed_comments += sum(
                1
                for comment in comments
                if isinstance(comment, dict)
                and (comment.get("error_type") or comment.get("error_message"))
            )
            total_artifacts += int(issue_entry.get("artifact_count", 0) or 0)
            continue

        issue_entry = _extract_single_issue(
            payload=payload,
            client=client,
            model_id=settings.model_id,
            num_ctx=settings.num_ctx,
        )

        issue_payload = {
            "repository": repo_ref.slug,
            "prompt_version": PROMPT_VERSION,
            "source": "responses",
            "model_id": settings.model_id,
            "model_url": settings.model_url,
            "status": "completed",
            "issue": issue_entry,
        }
        write_json(output_path, issue_payload)

        issue_count += 1
        total_comments += len(issue_entry.get("comments", []))
        failed_comments += sum(
            1
            for comment in issue_entry.get("comments", [])
            if isinstance(comment, dict)
            and (comment.get("error_type") or comment.get("error_message"))
        )
        total_artifacts += int(issue_entry.get("artifact_count", 0) or 0)

    summary = {
        "repository": repo_ref.slug,
        "prompt_version": PROMPT_VERSION,
        "source": "responses",
        "model_id": settings.model_id,
        "model_url": settings.model_url,
        "issue_count": issue_count,
        "total_comment_count": total_comments,
        "failed_comment_count": failed_comments,
        "successful_comment_count": total_comments - failed_comments,
        "total_artifact_count": total_artifacts,
        "issue_number_filter": settings.issue_number,
        "status": "completed",
    }
    write_json(derived_dir / "summary.json", summary)
    return summary


def _extract_single_issue(
    *,
    payload: Mapping[str, Any],
    client: Client,
    model_id: str,
    num_ctx: int,
) -> Dict[str, Any]:
    issue = payload.get("issue", {}) if isinstance(payload.get("issue"), dict) else {}
    issue_number = payload.get("issue_number")
    issue_title = str(issue.get("title") or "")
    issue_url = str(issue.get("url") or "")
    response_text = str(payload.get("response") or "")

    split_comments = split_response_into_comments(response_text)

    issue_entry: Dict[str, Any] = {
        "issue_number": issue_number,
        "issue_title": issue_title,
        "issue_url": issue_url,
        "comments": [],
        "artifact_count": 0,
        "split_comment_count": len(split_comments),
        "original_response_chars": len(response_text),
    }

    indexed_comments = [
        {
            "comment_index": idx,
            "author": "llm",
            "link": "",
            "text": text,
        }
        for idx, text in enumerate(split_comments)
    ]
    comment_results = _extract_issue_comment_batches(
        client=client,
        model_id=model_id,
        num_ctx=num_ctx,
        repository=str(payload.get("repository") or ""),
        issue_number=issue_number,
        indexed_comments=indexed_comments,
    )

    for idx, text in enumerate(split_comments):
        batch = comment_results.get(idx)
        comment_result: Dict[str, Any] = {
            "comment_index": idx,
            "comment_author": "llm",
            "comment_link": "",
            "comment_text": text,
            "artifacts": [],
            "artifact_count": 0,
        }
        if isinstance(batch, dict) and batch.get("error_type"):
            comment_result["error_type"] = batch.get("error_type")
            comment_result["error_message"] = batch.get("error_message")
        else:
            artifacts = list(batch.get("artifacts", [])) if isinstance(batch, dict) else []
            comment_result["artifacts"] = artifacts
            comment_result["artifact_count"] = len(artifacts)
            issue_entry["artifact_count"] += len(artifacts)
        issue_entry["comments"].append(comment_result)

    return issue_entry


def _extract_issue_comment_batches(
    *,
    client: Client,
    model_id: str,
    num_ctx: int,
    repository: str,
    issue_number: Any,
    indexed_comments: Sequence[Mapping[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    if not indexed_comments:
        return {}

    user_prompt = "\n".join(
        (
            f"Repo: {repository}",
            f"Issue: {issue_number}",
            "Comments JSON (treat each comment_index independently):",
            json.dumps(indexed_comments, ensure_ascii=True, separators=(",", ":")),
            METADATA_RULES_TEXT,
            "Return JSON only: {\"comments\":[{\"comment_index\":<int>,\"artifacts\":[...]}]}",
            "Include one entry for every comment_index from the input.",
        )
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
            + "\nBatch mode: You will receive multiple comments in one request. "
            + "Process each comment separately and output per-comment artifacts.",
        },
        {"role": "user", "content": user_prompt},
    ]

    result_by_index: Dict[int, Dict[str, Any]] = {}
    try:
        response = client.chat(
            model=model_id,
            messages=messages,
            format=BatchArtifactEnvelope.model_json_schema(),
            options={"num_ctx": max(512, int(num_ctx))},
        )
        content = response.get("message", {}).get("content", "")
        envelope = BatchArtifactEnvelope.model_validate_json(content)
        for item in envelope.comments:
            idx = int(item.comment_index)
            result_by_index[idx] = {
                "artifacts": [_normalize_artifact(artifact) for artifact in item.artifacts]
            }
    except ValidationError as exc:
        for item in indexed_comments:
            idx = int(item.get("comment_index", -1))
            if idx < 0:
                continue
            result_by_index[idx] = {
                "error_type": "ValidationError",
                "error_message": str(exc),
            }
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        for item in indexed_comments:
            idx = int(item.get("comment_index", -1))
            if idx < 0:
                continue
            result_by_index[idx] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

    for item in indexed_comments:
        idx = int(item.get("comment_index", -1))
        if idx < 0 or idx in result_by_index:
            continue
        result_by_index[idx] = {"artifacts": []}
    return result_by_index


def split_response_into_comments(response: str) -> List[str]:
    text = response.strip()
    if not text:
        return []

    matches = list(TOPIC_SPLIT_PATTERN.finditer(text))
    if not matches:
        return [text]

    comments: List[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if segment:
            comments.append(segment)

    return comments or [text]


def _progress_iter(
    iterable: Iterable[Any],
    *,
    desc: str,
    unit: str,
    total: Optional[int] = None,
) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, total=total)
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ollama import Client
from pydantic import ValidationError

from tbyc_dataset.extraction.pipeline import _extract_comment_artifacts, _normalize_artifact
from tbyc_dataset.extraction.prompt import PROMPT_VERSION, iter_comment_prompt_jobs
from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import ensure_directory, read_json, write_json


LOGGER = logging.getLogger(__name__)

TOPIC_SPLIT_PATTERN = re.compile(r"\[[^\]\n]{1,120}\]")


@dataclass(frozen=True)
class DerivedExtractionSettings:
    # Model used to run extraction over response text (currently Ollama-backed).
    model_id: str = "qwen2.5:14b"
    # Model id used to locate response files; defaults to model_id for backward compatibility.
    responses_model_id: Optional[str] = None
    # Root directory used to locate source responses.
    responses_root_dirname: str = "responses"
    # Root directory used to write derived extraction outputs.
    derived_root_dirname: str = "derived"
    model_url: str = "http://localhost:11434"
    num_ctx: int = 32768
    limit_issues: Optional[int] = None
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
    responses_model_id = (settings.responses_model_id or settings.model_id).strip()
    responses_model_dir = _model_dir_name(responses_model_id)
    responses_dir = base_dir / settings.responses_root_dirname / responses_model_dir / repo_ref.fs_slug
    # Derived artifacts are keyed by the response model lineage so downstream metrics
    # remain aligned with the model that generated the responses.
    derived_dir = base_dir / settings.derived_root_dirname / responses_model_dir / repo_ref.fs_slug
    ensure_directory(derived_dir)

    response_paths = sorted(responses_dir.glob("issue_*.json"))
    if settings.issue_number is not None:
        response_paths = [path for path in response_paths if path.name == f"issue_{settings.issue_number}.json"]
    if settings.limit_issues is not None:
        response_paths = response_paths[: max(0, int(settings.limit_issues))]

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
            "responses_root_dirname": settings.responses_root_dirname,
            "derived_root_dirname": settings.derived_root_dirname,
            "model_id": responses_model_id,
            "responses_model_id": responses_model_id,
            "extraction_model_id": settings.model_id,
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
        "model_id": responses_model_id,
        "responses_model_id": responses_model_id,
        "extraction_model_id": settings.model_id,
        "prompt_version": PROMPT_VERSION,
        "source": "responses",
        "responses_root_dirname": settings.responses_root_dirname,
        "derived_root_dirname": settings.derived_root_dirname,
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
    record = _build_extraction_record(
        repository=str(payload.get("repository") or ""),
        issue_number=issue_number,
        issue_url=issue_url,
        comment_blocks=split_comments,
    )

    issue_entry: Dict[str, Any] = {
        "issue_number": issue_number,
        "issue_title": issue_title,
        "issue_url": issue_url,
        "comments": [],
        "artifact_count": 0,
        "split_comment_count": len(split_comments),
        "original_response_chars": len(response_text),
    }

    prompt_jobs = list(iter_comment_prompt_jobs(record, include_issue_body=True))
    for prompt_job in prompt_jobs:
        comment_result: Dict[str, Any] = {
            "comment_author": str(prompt_job.get("comment_author") or ""),
            "comment_link": str(prompt_job.get("comment_link") or ""),
            "artifacts": [],
            "artifact_count": 0,
        }
        try:
            envelope = _extract_comment_artifacts(
                client=client,
                model_id=model_id,
                messages=prompt_job["messages"],
                num_ctx=num_ctx,
            )
            artifacts = [_normalize_artifact(artifact) for artifact in envelope.artifacts]
            comment_result["artifacts"] = artifacts
            comment_result["artifact_count"] = len(artifacts)
            issue_entry["artifact_count"] += len(artifacts)
        except ValidationError as exc:
            comment_result["error_type"] = "ValidationError"
            comment_result["error_message"] = str(exc)
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            comment_result["error_type"] = type(exc).__name__
            comment_result["error_message"] = str(exc)

        issue_entry["comments"].append(comment_result)

    return issue_entry


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


def _build_extraction_record(
    *,
    repository: str,
    issue_number: Any,
    issue_url: str,
    comment_blocks: Sequence[str],
) -> Dict[str, Any]:
    return {
        "repository": repository,
        "issue_number": issue_number,
        "issue_url": issue_url,
        "deliberation_thread": [
            {
                "url": "",
                "created_at": "",
                "author_login": "llm",
                "author_association": "NONE",
                "body": block,
                "is_issue_body": False,
            }
            for block in comment_blocks
        ],
    }


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


def _model_dir_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__") or "unknown-model"

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

from ollama import Client
from pydantic import BaseModel, Field, ValidationError

from tbyc_dataset.config import PipelineSettings
from tbyc_dataset.extraction.prompt import (
    ALLOWED_TAGS,
    ARTIFACT_TYPES,
    METADATA_FIELDS_BY_TYPE,
    PROMPT_VERSION,
    iter_comment_prompt_jobs,
)
from tbyc_dataset.models import JSONDict, RepositoryRef
from tbyc_dataset.storage import (
    curated_dataset_path,
    extraction_repo_dir,
    read_json,
    read_jsonl,
    write_json,
)


LOGGER = logging.getLogger(__name__)
ALLOWED_TAG_SET = set(ALLOWED_TAGS)


class MultiIssueProgress:
    def __init__(self, slots: int) -> None:
        self.slots = max(1, slots)
        self.enabled = sys.stderr.isatty() and self.slots > 0
        self._lock = Lock()
        self._lines = ["" for _ in range(self.slots)]
        self._in_use = [False for _ in range(self.slots)]
        if self.enabled:
            # Reserve fixed terminal space so bars stay pinned at the bottom.
            sys.stderr.write("\n" * self.slots)
            sys.stderr.flush()
            self._render_locked()

    @staticmethod
    def _format_line(issue_number: Any, current: int, total: int, status: str = "running") -> str:
        bounded_total = max(total, 0)
        bounded_current = min(max(current, 0), bounded_total) if bounded_total else 0
        width = 28
        filled = int((bounded_current / bounded_total) * width) if bounded_total else 0
        bar = "#" * filled + "-" * (width - filled)
        suffix = ""
        if status == "done":
            suffix = " done"
        elif status == "skipped":
            suffix = " skipped"
        return f"Issue {issue_number}: [{bar}] {bounded_current}/{bounded_total} comments{suffix}"

    def _render_locked(self) -> None:
        if not self.enabled:
            return
        sys.stderr.write(f"\x1b[{self.slots}F")
        for line in self._lines:
            sys.stderr.write("\r\x1b[2K")
            sys.stderr.write(line)
            sys.stderr.write("\n")
        sys.stderr.flush()

    def acquire_slot(self, issue_number: Any, total: int) -> int:
        with self._lock:
            slot = 0
            for idx, in_use in enumerate(self._in_use):
                if not in_use:
                    slot = idx
                    break
            self._in_use[slot] = True
            self._lines[slot] = self._format_line(issue_number=issue_number, current=0, total=total)
            self._render_locked()
            return slot

    def update(self, slot: int, issue_number: Any, current: int, total: int) -> None:
        with self._lock:
            self._lines[slot] = self._format_line(issue_number=issue_number, current=current, total=total)
            self._render_locked()

    def finish(self, slot: int, issue_number: Any, total: int, *, skipped: bool = False) -> None:
        with self._lock:
            status = "skipped" if skipped else "done"
            self._lines[slot] = self._format_line(issue_number=issue_number, current=total, total=total, status=status)
            self._in_use[slot] = False
            self._render_locked()

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            sys.stderr.write(f"\x1b[{self.slots}F")
            for _ in range(self.slots):
                sys.stderr.write("\r\x1b[2K\n")
            sys.stderr.flush()


class Artifact(BaseModel):
    type: Literal[
        "problem_statement",
        "proposed_solution",
        "alternative_solution",
        "design_decision",
        "trade_off_argument",
        "rationale",
        "constraint",
        "assumption",
        "implementation_detail",
        "code_snippet",
        "algorithm_approach",
        "api_design",
        "data_structure_choice",
        "configuration_choice",
        "benchmark_result",
        "performance_claim",
        "test_case",
        "bug_reproduction_steps",
        "edge_case",
        "empirical_evidence",
        "question",
        "answer_clarification",
        "agreement",
        "disagreement",
        "suggestion",
        "critique",
        "task_assignment",
        "status_update",
        "priority_discussion",
        "blocking_issue",
        "dependency",
    ]
    summary: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArtifactEnvelope(BaseModel):
    artifacts: List[Artifact] = Field(default_factory=list)


@dataclass(frozen=True)
class ExtractionSettings:
    model_id: str
    model_url: str
    num_ctx: int = 8192
    limit_threads: Optional[int] = None
    issue_number: Optional[int] = None
    parallel_issues: int = 1
    skip_existing: bool = True


def _normalize_artifact(raw: Artifact) -> JSONDict:
    type_key = str(raw.type)
    allowed_metadata_keys = METADATA_FIELDS_BY_TYPE.get(type_key, ())
    normalized_metadata = {
        key: (raw.metadata.get(key) if raw.metadata.get(key) not in (None, "") else "unknown")
        for key in allowed_metadata_keys
    }
    normalized_tags = _normalize_tags(raw.tags)
    if not normalized_tags:
        normalized_tags = ["incorrect_behavior"]

    return {
        "type": type_key,
        "summary": raw.summary,
        "tags": normalized_tags,
        "metadata": normalized_metadata,
    }


def _normalize_tags(tags: List[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for tag in tags:
        candidate = str(tag).strip().lower().replace("-", "_").replace(" ", "_")
        if candidate in ALLOWED_TAG_SET and candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)
    return normalized


def _extract_comment_artifacts(
    client: Client,
    model_id: str,
    messages: List[Dict[str, str]],
    num_ctx: int,
) -> ArtifactEnvelope:
    response = client.chat(
        model=model_id,
        messages=messages,
        format=ArtifactEnvelope.model_json_schema(),
        options={"num_ctx": max(512, int(num_ctx))},
    )
    content = response.get("message", {}).get("content", "")
    return ArtifactEnvelope.model_validate_json(content)


def _build_issue_artifacts_path(settings: PipelineSettings, repo: RepositoryRef, issue_number: Any) -> str:
    return str(extraction_repo_dir(settings.output_root, repo) / f"issue_{issue_number}.json")


def _summarize_issue_counts(issue_entry: JSONDict) -> Dict[str, int]:
    comments = issue_entry.get("comments", []) if isinstance(issue_entry, dict) else []
    total_comments = len(comments)
    failed_comments = sum(
        1
        for c in comments
        if isinstance(c, dict)
        and (
            c.get("status") == "error"
            or bool(c.get("error_type"))
            or bool(c.get("error_message"))
        )
    )
    artifact_count = int(issue_entry.get("artifact_count", 0)) if isinstance(issue_entry, dict) else 0
    return {
        "total_comments": total_comments,
        "failed_comments": failed_comments,
        "artifact_count": artifact_count,
    }


def _process_single_issue(
    *,
    row: JSONDict,
    repo: RepositoryRef,
    pipeline_settings: PipelineSettings,
    extraction_settings: ExtractionSettings,
    progress: Optional[MultiIssueProgress],
) -> Dict[str, Any]:
    issue_number = row.get("issue_number")
    issue_records_path = _build_issue_artifacts_path(pipeline_settings, repo, issue_number)
    issue_output_path = Path(issue_records_path)
    slot = -1
    comment_jobs = list(iter_comment_prompt_jobs(row, include_issue_body=True))
    comment_total = len(comment_jobs)
    if progress is not None:
        slot = progress.acquire_slot(issue_number=issue_number, total=comment_total)

    if extraction_settings.skip_existing and issue_output_path.exists():
        try:
            existing_payload = read_json(issue_output_path)
            existing_issue = existing_payload.get("issue", {})
            counts = _summarize_issue_counts(existing_issue)
            if progress is not None and slot >= 0:
                progress.finish(slot=slot, issue_number=issue_number, total=comment_total, skipped=True)
            return {
                "issue_number": issue_number,
                "issue_entry": existing_issue,
                "total_comments": counts["total_comments"],
                "failed_comments": counts["failed_comments"],
                "artifact_count": counts["artifact_count"],
                "skipped": True,
            }
        except Exception as exc:  # pragma: no cover - corrupted file edge case
            LOGGER.warning("issue=%s existing file unreadable; reprocessing (%s)", issue_number, exc)

    client = Client(host=extraction_settings.model_url)
    if progress is not None and slot >= 0:
        progress.update(slot=slot, issue_number=issue_number, current=0, total=comment_total)

    issue_entry: JSONDict = {
        "issue_number": issue_number,
        "issue_title": row.get("input_vector", {}).get("title", ""),
        "issue_url": row.get("issue_url", ""),
        "comments": [],
        "artifact_count": 0,
    }

    issue_payload: JSONDict = {
        "repository": repo.slug,
        "prompt_version": PROMPT_VERSION,
        "model_id": extraction_settings.model_id,
        "model_url": extraction_settings.model_url,
        "records_path": issue_records_path,
        "status": "in_progress",
        "issue": issue_entry,
    }
    write_json(issue_output_path, issue_payload)

    issue_failed_count = 0

    for idx, prompt_job in enumerate(comment_jobs, start=1):
        comment_author = str(prompt_job.get("comment_author", ""))
        comment_link = str(prompt_job.get("comment_link", ""))

        comment_result: JSONDict = {
            "comment_author": comment_author,
            "comment_link": comment_link,
            "artifacts": [],
            "artifact_count": 0,
        }

        try:
            envelope = _extract_comment_artifacts(
                client=client,
                model_id=extraction_settings.model_id,
                messages=prompt_job["messages"],
                num_ctx=extraction_settings.num_ctx,
            )
            artifacts = [
                _normalize_artifact(a)
                for a in envelope.artifacts
            ]
            comment_result["artifacts"] = artifacts
            comment_result["artifact_count"] = len(artifacts)
            issue_entry["artifact_count"] += len(artifacts)
        except ValidationError as exc:
            issue_failed_count += 1
            comment_result["error_type"] = "ValidationError"
            comment_result["error_message"] = str(exc)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            issue_failed_count += 1
            comment_result["error_type"] = type(exc).__name__
            comment_result["error_message"] = str(exc)

        issue_entry["comments"].append(comment_result)
        if progress is not None and slot >= 0:
            progress.update(slot=slot, issue_number=issue_number, current=idx, total=comment_total)

        write_json(issue_output_path, issue_payload)

    if progress is not None and slot >= 0:
        progress.finish(slot=slot, issue_number=issue_number, total=comment_total)

    issue_payload["status"] = "completed"
    write_json(issue_output_path, issue_payload)

    return {
        "issue_number": issue_number,
        "issue_entry": issue_entry,
        "total_comments": len(comment_jobs),
        "failed_comments": issue_failed_count,
        "artifact_count": int(issue_entry["artifact_count"]),
        "skipped": False,
    }


def _build_summary(
    *,
    repo: RepositoryRef,
    extraction_settings: ExtractionSettings,
    records_path: str,
    issue_count: int,
    total_comment_count: int,
    failed_comment_count: int,
    total_artifact_count: int,
    status: str,
) -> JSONDict:
    return {
        "repository": repo.slug,
        "prompt_version": PROMPT_VERSION,
        "model_id": extraction_settings.model_id,
        "model_url": extraction_settings.model_url,
        "records_path": records_path,
        "issue_count": issue_count,
        "total_comment_count": total_comment_count,
        "failed_comment_count": failed_comment_count,
        "successful_comment_count": total_comment_count - failed_comment_count,
        "total_artifact_count": total_artifact_count,
        "issue_number_filter": extraction_settings.issue_number,
        "status": status,
    }


def extract_discussion_artifacts(
    repo: RepositoryRef,
    pipeline_settings: PipelineSettings,
    extraction_settings: ExtractionSettings,
) -> JSONDict:
    dataset_rows = read_jsonl(curated_dataset_path(pipeline_settings.output_root, repo))
    LOGGER.info("loaded curated rows=%s", len(dataset_rows))
    filtered_rows = dataset_rows
    if extraction_settings.issue_number is not None:
        filtered_rows = [
            row for row in dataset_rows if int(row.get("issue_number", -1)) == extraction_settings.issue_number
        ]
        LOGGER.info("applied issue filter issue_number=%s matched=%s", extraction_settings.issue_number, len(filtered_rows))

    if extraction_settings.limit_threads is not None:
        filtered_rows = filtered_rows[: extraction_settings.limit_threads]
        LOGGER.info("applied thread limit limit=%s selected=%s", extraction_settings.limit_threads, len(filtered_rows))

    parallel_issues = max(1, extraction_settings.parallel_issues)
    LOGGER.info(
        "starting extraction issues=%s model=%s num_ctx=%s parallel_issues=%s skip_existing=%s",
        len(filtered_rows),
        extraction_settings.model_id,
        extraction_settings.num_ctx,
        parallel_issues,
        extraction_settings.skip_existing,
    )

    records_path = ""

    total_comment_count = 0
    total_artifact_count = 0
    failed_comment_count = 0

    progress = MultiIssueProgress(slots=parallel_issues)
    try:
        with ThreadPoolExecutor(max_workers=parallel_issues) as executor:
            futures = [
                executor.submit(
                    _process_single_issue,
                    row=row,
                    repo=repo,
                    pipeline_settings=pipeline_settings,
                    extraction_settings=extraction_settings,
                    progress=progress,
                )
                for row in filtered_rows
            ]

            for future in as_completed(futures):
                result = future.result()
                total_comment_count += int(result["total_comments"])
                failed_comment_count += int(result["failed_comments"])
                total_artifact_count += int(result["artifact_count"])
    finally:
        progress.close()

    summary = _build_summary(
        repo=repo,
        extraction_settings=extraction_settings,
        records_path=records_path,
        issue_count=len(filtered_rows),
        total_comment_count=total_comment_count,
        failed_comment_count=failed_comment_count,
        total_artifact_count=total_artifact_count,
        status="completed",
    )

    LOGGER.info("completed extraction total_comments=%s total_artifacts=%s", total_comment_count, total_artifact_count)
    return summary

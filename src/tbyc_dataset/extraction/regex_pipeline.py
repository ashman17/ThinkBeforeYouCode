from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

from tbyc_dataset.config import PipelineSettings
from tbyc_dataset.extraction.pipeline import MultiIssueProgress
from tbyc_dataset.extraction.prompt import ALLOWED_TAGS, METADATA_FIELDS_BY_TYPE, iter_comment_prompt_jobs
from tbyc_dataset.models import JSONDict, RepositoryRef
from tbyc_dataset.storage import curated_dataset_path, extraction_regex_repo_dir, read_json, read_jsonl, write_json


LOGGER = logging.getLogger(__name__)
REGEX_PROMPT_VERSION = "discussion_artifacts_regex_v1"
ALLOWED_TAG_SET = set(ALLOWED_TAGS)

QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:who|what|when|where|why|how|could|can|would|should|is|are|do|does|did|will)\b",
    re.IGNORECASE,
)
STATUS_RE = re.compile(
    r"\b(?:released|landed|merged|fixed in|available in|cherry-?picked|backported|rc\d+|ready for testing)\b",
    re.IGNORECASE,
)
SUGGESTION_RE = re.compile(
    r"\b(?:suggest|recommend|try|workaround|you can|consider|enable|disable)\b",
    re.IGNORECASE,
)
SOLUTION_RE = re.compile(
    r"\b(?:fix(?:ed|es|ing)?|solution|move|add|remove|change|rename|avoid|prevent|don't trim|do not trim)\b",
    re.IGNORECASE,
)
PROBLEM_RE = re.compile(
    r"\b(?:bug|regression|error|errors|fail(?:ed|ing)?|failure|conflict|crash|hang|timeout|broken|issue)\b",
    re.IGNORECASE,
)
TASK_RE = re.compile(r"\b(?:assign(?:ed|ing)?|take this|for @[\w-]+|cc @[\w-]+)\b", re.IGNORECASE)
PRIORITY_RE = re.compile(r"\b(?:priority|prioritiz(?:e|ing)|backport|fork)\b", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]+`")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RegexExtractionSettings:
    limit_threads: Optional[int] = None
    issue_number: Optional[int] = None
    parallel_issues: int = 1
    skip_existing: bool = True


def _build_issue_artifacts_path(settings: PipelineSettings, repo: RepositoryRef, issue_number: Any) -> str:
    return str(extraction_regex_repo_dir(settings.output_root, repo) / f"issue_{issue_number}.json")


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


def _clean_summary(text: str, *, fallback: str) -> str:
    collapsed = WS_RE.sub(" ", text).strip(" -:\n\t")
    if not collapsed:
        return fallback
    if len(collapsed) <= 220:
        return collapsed
    return collapsed[:217].rstrip() + "..."


def _comment_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _first_sentence(text: str) -> str:
    stripped = WS_RE.sub(" ", text).strip()
    if not stripped:
        return ""
    match = re.search(r"(.+?[.?!])(?:\s|$)", stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _infer_tags(text: str) -> List[str]:
    lowered = text.lower()
    tags: List[str] = []
    keyword_tags = (
        ("regression", "regression"),
        ("flaky", "flaky_test"),
        ("ci", "ci_failure"),
        ("build", "build_failure"),
        ("runtime error", "runtime_error"),
        ("compile", "compile_error"),
        ("assert", "assertion_failure"),
        ("performance", "performance_issue"),
        ("memory", "memory_issue"),
        ("security", "security_issue"),
        ("corrupt", "data_corruption"),
        ("edge case", "edge_case"),
        ("compat", "compatibility_issue"),
        ("race", "race_condition"),
        ("non-determin", "non_determinism"),
        ("timestamp", "timestamp_issue"),
        ("float", "floating_point_error"),
        ("off by one", "off_by_one"),
        ("state", "state_sync_issue"),
        ("logic", "logic_bug"),
        ("api", "api"),
        ("config", "config_issue"),
        ("dependency", "dependency_issue"),
        ("version", "version_mismatch"),
        ("linux", "linux"),
        ("macos", "macos"),
        ("darwin", "macos"),
        ("windows", "windows"),
        ("arm", "arm"),
        ("x86", "x86"),
        ("docker", "docker"),
        ("gpu", "gpu"),
        ("network", "networking"),
        ("filesystem", "filesystem"),
        ("database", "database"),
        ("cli", "cli"),
        ("frontend", "frontend"),
        ("backend", "backend"),
        ("rpc", "rpc"),
        ("serialization", "serialization"),
        ("crypto", "crypto"),
        ("validation", "validation"),
        ("scheduler", "scheduler"),
        ("training", "training"),
        ("inference", "inference"),
        ("pipeline", "data_pipeline"),
        ("token", "tokenization"),
        ("crash", "crash"),
        ("hang", "hang"),
        ("timeout", "timeout"),
        ("incorrect", "incorrect_output"),
        ("missing", "missing_output"),
        ("extra output", "extra_output"),
        ("error:", "log_error"),
        ("workaround", "workaround"),
        ("fix", "fix_proposed"),
        ("merged", "fix_merged"),
        ("refactor", "refactor"),
        ("test", "unit_test"),
    )
    for needle, tag in keyword_tags:
        if needle in lowered and tag in ALLOWED_TAG_SET and tag not in tags:
            tags.append(tag)
    if "bug" in lowered and "bug" not in tags:
        tags.append("bug")
    if not tags:
        tags.append("incorrect_behavior")
    return tags


def _normalize_metadata(artifact_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: (metadata.get(key) if metadata.get(key) not in (None, "") else "unknown")
        for key in METADATA_FIELDS_BY_TYPE.get(artifact_type, ())
    }


def _make_artifact(artifact_type: str, summary: str, text: str, metadata: Dict[str, Any]) -> JSONDict:
    return {
        "type": artifact_type,
        "summary": _clean_summary(summary, fallback=artifact_type.replace("_", " ")),
        "tags": _infer_tags(text),
        "metadata": _normalize_metadata(artifact_type, metadata),
    }


def _code_artifacts(text: str) -> Iterable[JSONDict]:
    for match in CODE_FENCE_RE.finditer(text):
        language = (match.group("lang") or "plaintext").strip() or "plaintext"
        body = match.group("body").strip()
        if not body:
            continue
        yield _make_artifact(
            "code_snippet",
            f"Code block showing { _first_sentence(body) or 'technical details' }",
            match.group(0),
            {
                "language": language,
                "snippet_kind": "log" if "error" in body.lower() else "example",
                "intended_use": "illustration",
            },
        )


def _question_artifact(text: str) -> Optional[JSONDict]:
    first_sentence = _first_sentence(text)
    question_lines = [
        line for line in _comment_lines(text)
        if line.endswith("?") and len(line) <= 220 and not line.startswith("###")
    ]
    if not question_lines and not QUESTION_PREFIX_RE.search(first_sentence):
        return None
    return _make_artifact(
        "question",
        question_lines[0] if question_lines else (first_sentence or "Question raised in discussion"),
        text,
        {
            "question_kind": "clarification_request",
            "target_area": "unknown",
            "urgency": "unknown",
        },
    )


def _problem_artifact(text: str) -> Optional[JSONDict]:
    if not PROBLEM_RE.search(text):
        return None
    first_sentence = _first_sentence(text)
    return _make_artifact(
        "problem_statement",
        first_sentence or "Problem reported in discussion",
        text,
        {
            "problem_kind": "bug_report",
            "affected_area": "unknown",
            "user_or_system_impact": "unknown",
        },
    )


def _solution_artifact(text: str) -> Optional[JSONDict]:
    if not SOLUTION_RE.search(text):
        return None
    first_sentence = _first_sentence(text)
    return _make_artifact(
        "proposed_solution",
        first_sentence or "Proposed code or configuration change",
        text,
        {
            "solution_kind": "code_change",
            "target_area": "unknown",
            "expected_benefit": "unknown",
        },
    )


def _suggestion_artifact(text: str) -> Optional[JSONDict]:
    first_sentence = _first_sentence(text)
    if not SUGGESTION_RE.search(first_sentence):
        return None
    return _make_artifact(
        "suggestion",
        first_sentence or "Suggestion offered in discussion",
        text,
        {
            "target_area": "unknown",
            "expected_improvement": "unknown",
            "urgency": "unknown",
        },
    )


def _status_artifact(text: str) -> Optional[JSONDict]:
    if not STATUS_RE.search(text):
        return None
    first_sentence = _first_sentence(text)
    blocking = bool(re.search(r"\bblock(?:ed|ing|er)?\b", text, re.IGNORECASE))
    return _make_artifact(
        "status_update",
        first_sentence or "Status update shared in discussion",
        text,
        {
            "status_kind": "released" if "release" in text.lower() or "rc" in text.lower() else "update",
            "progress_state": "complete" if re.search(r"\b(?:merged|released|fixed)\b", text, re.IGNORECASE) else "in_progress",
            "blocking": blocking,
        },
    )


def _task_artifact(text: str) -> Optional[JSONDict]:
    if not TASK_RE.search(text):
        return None
    assignee_match = re.search(r"@[\w-]+", text)
    return _make_artifact(
        "task_assignment",
        _first_sentence(text) or "Task assignment noted in discussion",
        text,
        {
            "assignee": assignee_match.group(0) if assignee_match else "unknown",
            "deliverable": "unknown",
            "due_context": "unknown",
        },
    )


def _priority_artifact(text: str) -> Optional[JSONDict]:
    first_sentence = _first_sentence(text)
    if not PRIORITY_RE.search(first_sentence):
        return None
    return _make_artifact(
        "priority_discussion",
        first_sentence or "Priority or release discussion",
        text,
        {
            "priority_level": "unknown",
            "compared_against": "unknown",
            "scope": "release_planning",
        },
    )


def extract_regex_artifacts_from_comment(text: str) -> List[JSONDict]:
    if not text.strip():
        return []

    artifacts: List[JSONDict] = []
    seen = set()

    for artifact in _code_artifacts(text):
        key = (artifact["type"], artifact["summary"])
        if key not in seen:
            artifacts.append(artifact)
            seen.add(key)

    for builder in (
        _question_artifact,
        _solution_artifact,
        _suggestion_artifact,
        _status_artifact,
        _task_artifact,
        _priority_artifact,
        _problem_artifact,
    ):
        artifact = builder(text)
        if artifact is None:
            continue
        key = (artifact["type"], artifact["summary"])
        if key not in seen:
            artifacts.append(artifact)
            seen.add(key)

    if artifacts:
        return artifacts

    lines = _comment_lines(text)
    if not lines:
        return []
    return [
        _make_artifact(
            "status_update",
            lines[0],
            text,
            {
                "status_kind": "mention",
                "progress_state": "informative",
                "blocking": False,
            },
        )
    ]


def _process_single_issue(
    *,
    row: JSONDict,
    repo: RepositoryRef,
    pipeline_settings: PipelineSettings,
    extraction_settings: RegexExtractionSettings,
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
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("issue=%s existing regex file unreadable; reprocessing (%s)", issue_number, exc)

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
        "prompt_version": REGEX_PROMPT_VERSION,
        "extractor": "regex",
        "records_path": issue_records_path,
        "status": "in_progress",
        "issue": issue_entry,
    }
    write_json(issue_output_path, issue_payload)

    for idx, prompt_job in enumerate(comment_jobs, start=1):
        comment_author = str(prompt_job.get("comment_author", ""))
        comment_link = str(prompt_job.get("comment_link", ""))
        comment_text = str(prompt_job.get("comment_text", ""))

        artifacts = extract_regex_artifacts_from_comment(comment_text)
        comment_result: JSONDict = {
            "comment_author": comment_author,
            "comment_link": comment_link,
            "artifacts": artifacts,
            "artifact_count": len(artifacts),
        }
        issue_entry["comments"].append(comment_result)
        issue_entry["artifact_count"] += len(artifacts)

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
        "failed_comments": 0,
        "artifact_count": int(issue_entry["artifact_count"]),
        "skipped": False,
    }


def extract_discussion_artifacts_regex(
    repo: RepositoryRef,
    pipeline_settings: PipelineSettings,
    extraction_settings: RegexExtractionSettings,
) -> JSONDict:
    dataset_rows = read_jsonl(curated_dataset_path(pipeline_settings.output_root, repo))
    filtered_rows = dataset_rows
    if extraction_settings.issue_number is not None:
        filtered_rows = [
            row for row in dataset_rows if int(row.get("issue_number", -1)) == extraction_settings.issue_number
        ]

    if extraction_settings.limit_threads is not None:
        filtered_rows = filtered_rows[: extraction_settings.limit_threads]

    parallel_issues = max(1, extraction_settings.parallel_issues)

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

    return {
        "repository": repo.slug,
        "prompt_version": REGEX_PROMPT_VERSION,
        "extractor": "regex",
        "records_path": "",
        "issue_count": len(filtered_rows),
        "total_comment_count": total_comment_count,
        "failed_comment_count": failed_comment_count,
        "successful_comment_count": total_comment_count - failed_comment_count,
        "total_artifact_count": total_artifact_count,
        "issue_number_filter": extraction_settings.issue_number,
        "status": "completed",
    }

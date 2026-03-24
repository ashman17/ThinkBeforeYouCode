from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from tbyc_dataset.extraction.discussion_entities_prompt import (
    PROMPT_DESCRIPTION,
    PROMPT_VERSION,
    build_langextract_examples,
)
from tbyc_dataset.models import JSONDict, RepositoryRef
from tbyc_dataset.storage import (
    annotated_discussion_entities_path,
    curated_dataset_path,
    discussion_entities_records_path,
    discussion_entities_summary_path,
    read_jsonl,
    write_json,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)
PROGRESS_LOG_EVERY = 5


def extract_discussion_entities(
    repo: RepositoryRef,
    output_root: Path,
    model_id: str = "gemma3:4b",
    model_url: str = "http://localhost:11434",
    limit_threads: Optional[int] = None,
    save_annotated: bool = False,
) -> JSONDict:
    try:
        import langextract as lx
    except ImportError as exc:
        raise RuntimeError(
            "langextract is required for discussion entity extraction. "
            "Install it with `pip install langextract`."
        ) from exc

    configure_logging()

    dataset_path = curated_dataset_path(output_root, repo)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No curated dataset found for {repo.slug} at {dataset_path}. "
            "Run `curate-repo` first."
        )

    LOGGER.info("Loading curated dataset from %s", dataset_path)
    issue_records = read_jsonl(dataset_path)
    thread_jobs = build_thread_jobs(issue_records, limit_threads=limit_threads)
    LOGGER.info(
        "Prepared %s discussion threads across %s issues for extraction using model=%s prompt=%s",
        len(thread_jobs),
        len(issue_records),
        model_id,
        PROMPT_VERSION,
    )

    LOGGER.info("Building LangExtract few-shot examples")
    examples = build_langextract_examples(lx)
    records: List[JSONDict] = []
    annotated_documents: List[Any] = []
    failed_threads = 0
    total_threads = len(thread_jobs)

    for position, job in enumerate(thread_jobs, start=1):
        issue_record = job["issue_record"]
        thread_text = job["thread_text"]
        comment_spans = job["comment_spans"]
        source_comments = job["source_comments"]

        if should_log_progress(position, total_threads):
            LOGGER.info(
                "[%s/%s] extracting issue #%s (%s comments)",
                position,
                total_threads,
                issue_record["issue_number"],
                len(source_comments),
            )

        try:
            result = lx.extract(
                text_or_documents=thread_text,
                prompt_description=PROMPT_DESCRIPTION,
                examples=examples,
                model_id=model_id,
                model_url=model_url,
                fence_output=False,
                use_schema_constraints=False,
                show_progress=True,
                fetch_urls=False,
            )
        except Exception as exc:
            failed_threads += 1
            LOGGER.warning(
                "[%s/%s] LangExtract failed for issue #%s: %s: %s",
                position,
                total_threads,
                issue_record["issue_number"],
                type(exc).__name__,
                exc,
            )
            records.append(
                build_thread_record(
                    issue_record=issue_record,
                    thread_text=thread_text,
                    source_comments=source_comments,
                    model_id=model_id,
                    serialized_extractions=[],
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue

        serialized_extractions = [
            serialize_extraction(extraction, comment_spans)
            for extraction in getattr(result, "extractions", [])
        ]
        records.append(
            build_thread_record(
                issue_record=issue_record,
                thread_text=thread_text,
                source_comments=source_comments,
                model_id=model_id,
                serialized_extractions=serialized_extractions,
                status="success",
                error_type=None,
                error_message=None,
            )
        )

        if save_annotated:
            annotated_documents.append(result)

    records_path = discussion_entities_records_path(output_root, repo)
    LOGGER.info("Writing extraction records to %s", records_path)
    write_jsonl(records_path, records)

    if save_annotated and annotated_documents:
        annotated_path = annotated_discussion_entities_path(output_root, repo)
        LOGGER.info("Writing LangExtract annotated documents to %s", annotated_path)
        lx.io.save_annotated_documents(
            annotated_documents,
            output_name=annotated_path.name,
            output_dir=str(annotated_path.parent),
        )
    else:
        annotated_path = None

    summary = summarize_extraction_records(
        records=records,
        repository=repo.slug,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        records_path=records_path,
        annotated_path=annotated_path,
    )
    write_json(discussion_entities_summary_path(output_root, repo), summary)
    LOGGER.info(
        "Extraction complete: processed=%s succeeded=%s failed=%s",
        total_threads,
        total_threads - failed_threads,
        failed_threads,
    )
    return summary


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.setLevel(logging.INFO)


def build_thread_jobs(
    issue_records: Iterable[JSONDict],
    limit_threads: Optional[int],
) -> List[JSONDict]:
    jobs: List[JSONDict] = []
    for issue_record in issue_records:
        source_comments = [
            {
                "comment_index": comment_index,
                "url": comment["url"],
                "created_at": comment["created_at"],
                "author_login": comment["author_login"],
                "author_association": comment["author_association"],
                "body": comment["body"],
            }
            for comment_index, comment in enumerate(issue_record.get("deliberation_thread", []))
            if (comment.get("body") or "").strip()
        ]
        if not source_comments:
            continue
        thread_text, comment_spans = format_thread_text(source_comments)
        jobs.append(
            {
                "issue_record": issue_record,
                "source_comments": source_comments,
                "thread_text": thread_text,
                "comment_spans": comment_spans,
            }
        )
        if limit_threads is not None and len(jobs) >= limit_threads:
            return jobs
    return jobs


def format_thread_text(source_comments: List[JSONDict]) -> Tuple[str, List[JSONDict]]:
    lines: List[str] = []
    comment_spans: List[JSONDict] = []
    cursor = 0

    for comment in source_comments:
        author_login = comment["author_login"] or "unknown"
        line = f"{author_login}: {squash_whitespace(comment['body'])}"
        if lines:
            cursor += 1
        start_pos = cursor
        end_pos = start_pos + len(line)
        lines.append(line)
        comment_spans.append(
            {
                "comment_index": comment["comment_index"],
                "author_login": author_login,
                "author_association": comment["author_association"],
                "speaker_role": speaker_role_from_association(
                    comment["author_association"]
                ),
                "created_at": comment["created_at"],
                "url": comment["url"],
                "start_pos": start_pos,
                "end_pos": end_pos,
            }
        )
        cursor = end_pos

    return "\n".join(lines), comment_spans


def speaker_role_from_association(author_association: Optional[str]) -> str:
    if author_association in {"OWNER", "MEMBER", "COLLABORATOR"}:
        return "maintainer"
    if author_association in {"CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER"}:
        return "contributor"
    return "external"


def squash_whitespace(text: str) -> str:
    return " ".join(text.split())


def should_log_progress(position: int, total_threads: int) -> bool:
    return position == 1 or position == total_threads or position % PROGRESS_LOG_EVERY == 0


def build_thread_record(
    issue_record: JSONDict,
    thread_text: str,
    source_comments: List[JSONDict],
    model_id: str,
    serialized_extractions: List[JSONDict],
    status: str,
    error_type: Optional[str],
    error_message: Optional[str],
) -> JSONDict:
    return {
        "repository": issue_record["repository"],
        "issue_number": issue_record["issue_number"],
        "issue_url": issue_record["issue_url"],
        "thread_comment_count": len(source_comments),
        "thread_text": thread_text,
        "source_comments": source_comments,
        "prompt_version": PROMPT_VERSION,
        "model_id": model_id,
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        "extraction_count": len(serialized_extractions),
        "grounded_extraction_count": sum(
            1 for extraction in serialized_extractions if extraction["is_grounded"]
        ),
        "extractions": serialized_extractions,
    }


def serialize_extraction(extraction: Any, comment_spans: List[JSONDict]) -> JSONDict:
    char_interval = serialize_char_interval(getattr(extraction, "char_interval", None))
    source_context = resolve_source_context(char_interval, comment_spans)
    attributes = dict(getattr(extraction, "attributes", {}) or {})
    if source_context.get("source_author_login") and not attributes.get("speaker"):
        attributes["speaker"] = source_context["source_author_login"]
    if source_context.get("speaker_role"):
        attributes["speaker_role"] = source_context["speaker_role"]
    return {
        "extraction_class": getattr(extraction, "extraction_class", None),
        "extraction_text": getattr(extraction, "extraction_text", None),
        "attributes": attributes,
        "char_interval": char_interval,
        "is_grounded": char_interval is not None,
        **source_context,
    }


def serialize_char_interval(char_interval: Any) -> Optional[JSONDict]:
    if char_interval is None:
        return None
    if isinstance(char_interval, dict):
        return dict(char_interval)
    start_pos = getattr(char_interval, "start_pos", None)
    end_pos = getattr(char_interval, "end_pos", None)
    if start_pos is not None or end_pos is not None:
        return {
            "start_pos": start_pos,
            "end_pos": end_pos,
        }
    if isinstance(char_interval, (list, tuple)) and len(char_interval) == 2:
        return {
            "start_pos": char_interval[0],
            "end_pos": char_interval[1],
        }
    return {"repr": repr(char_interval)}


def resolve_source_context(
    char_interval: Optional[JSONDict],
    comment_spans: List[JSONDict],
) -> JSONDict:
    if char_interval is None:
        return {}
    start_pos = char_interval.get("start_pos")
    end_pos = char_interval.get("end_pos")
    if start_pos is None and end_pos is None:
        return {}

    overlaps = [
        comment_span
        for comment_span in comment_spans
        if intervals_overlap(
            start_pos=start_pos,
            end_pos=end_pos,
            span_start=comment_span["start_pos"],
            span_end=comment_span["end_pos"],
        )
    ]
    if not overlaps and start_pos is not None:
        overlaps = [
            comment_span
            for comment_span in comment_spans
            if comment_span["start_pos"] <= start_pos < comment_span["end_pos"]
        ]
    if not overlaps:
        return {}

    context: JSONDict = {
        "source_comment_indices": [
            comment_span["comment_index"] for comment_span in overlaps
        ],
        "source_author_logins": [
            comment_span["author_login"] for comment_span in overlaps
        ],
        "source_author_associations": [
            comment_span["author_association"] for comment_span in overlaps
        ],
        "source_speaker_roles": [
            comment_span["speaker_role"] for comment_span in overlaps
        ],
    }
    if len(overlaps) == 1:
        comment_span = overlaps[0]
        context.update(
            {
                "source_comment_index": comment_span["comment_index"],
                "source_author_login": comment_span["author_login"],
                "source_author_association": comment_span["author_association"],
                "speaker_role": comment_span["speaker_role"],
                "source_comment_url": comment_span["url"],
                "source_comment_created_at": comment_span["created_at"],
            }
        )
    return context


def intervals_overlap(
    start_pos: Optional[int],
    end_pos: Optional[int],
    span_start: int,
    span_end: int,
) -> bool:
    effective_start = start_pos if start_pos is not None else end_pos
    effective_end = end_pos if end_pos is not None else start_pos
    if effective_start is None or effective_end is None:
        return False
    return not (effective_end <= span_start or span_end <= effective_start)


def summarize_extraction_records(
    records: Iterable[JSONDict],
    repository: str,
    model_id: str,
    prompt_version: str,
    records_path: Path,
    annotated_path: Optional[Path],
) -> JSONDict:
    rows = list(records)
    class_counts = Counter(
        extraction["extraction_class"]
        for row in rows
        for extraction in row["extractions"]
        if extraction.get("extraction_class")
    )
    return {
        "repository": repository,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "thread_record_count": len(rows),
        "successful_thread_count": sum(1 for row in rows if row["status"] == "success"),
        "failed_thread_count": sum(1 for row in rows if row["status"] == "error"),
        "threads_with_extractions": sum(1 for row in rows if row["extraction_count"] > 0),
        "total_extraction_count": sum(row["extraction_count"] for row in rows),
        "grounded_extraction_count": sum(row["grounded_extraction_count"] for row in rows),
        "extraction_class_distribution": dict(sorted(class_counts.items())),
        "records_path": str(records_path),
        "annotated_path": str(annotated_path) if annotated_path is not None else None,
    }

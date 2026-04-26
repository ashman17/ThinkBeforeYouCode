"""Extraction helpers for discussion artifacts."""

from tbyc_dataset.extraction.prompt import (
    ARTIFACT_TYPES,
    METADATA_FIELDS_BY_TYPE,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_comment_prompt_job,
    iter_comment_prompt_jobs,
)
from tbyc_dataset.extraction.pipeline import ExtractionSettings, extract_discussion_artifacts
from tbyc_dataset.extraction.regex_pipeline import (
    REGEX_PROMPT_VERSION,
    RegexExtractionSettings,
    extract_discussion_artifacts_regex,
    extract_regex_artifacts_from_comment,
)

__all__ = [
    "ARTIFACT_TYPES",
    "METADATA_FIELDS_BY_TYPE",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "build_comment_prompt_job",
    "iter_comment_prompt_jobs",
    "ExtractionSettings",
    "extract_discussion_artifacts",
    "REGEX_PROMPT_VERSION",
    "RegexExtractionSettings",
    "extract_discussion_artifacts_regex",
    "extract_regex_artifacts_from_comment",
]

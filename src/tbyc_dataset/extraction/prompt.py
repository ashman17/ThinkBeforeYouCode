from __future__ import annotations

import json
from typing import Iterator, Mapping, Sequence

from tbyc_dataset.models import JSONDict

PROMPT_VERSION = "discussion_artifacts_v7"

ARTIFACT_TYPES: Sequence[str] = (
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
)

ALLOWED_TAGS: Sequence[str] = (
    "bug",
    "regression",
    "flaky_test",
    "ci_failure",
    "build_failure",
    "runtime_error",
    "compile_error",
    "assertion_failure",
    "performance_issue",
    "memory_issue",
    "security_issue",
    "data_corruption",
    "incorrect_behavior",
    "edge_case",
    "compatibility_issue",
    "race_condition",
    "non_determinism",
    "timestamp_issue",
    "floating_point_error",
    "off_by_one",
    "null_pointer",
    "uninitialized_state",
    "state_sync_issue",
    "logic_bug",
    "api_misuse",
    "config_issue",
    "dependency_issue",
    "version_mismatch",
    "platform_specific",
    "overflow_underflow",
    "unit_test",
    "integration_test",
    "e2e_test",
    "functional_test",
    "test_framework_issue",
    "test_flakiness",
    "test_timeout",
    "test_infra_issue",
    "networking",
    "filesystem",
    "database",
    "api",
    "cli",
    "ui",
    "backend",
    "frontend",
    "distributed_system",
    "p2p",
    "consensus",
    "mempool",
    "rpc",
    "serialization",
    "crypto",
    "validation",
    "scheduler",
    "training",
    "inference",
    "data_pipeline",
    "tokenization",
    "linux",
    "macos",
    "windows",
    "arm",
    "x86",
    "docker",
    "ci_environment",
    "gpu",
    "low_memory",
    "build_time",
    "startup",
    "runtime",
    "shutdown",
    "deployment",
    "crash",
    "hang",
    "timeout",
    "incorrect_output",
    "missing_output",
    "extra_output",
    "log_error",
    "performance_degradation",
    "workaround",
    "fix_proposed",
    "fix_merged",
    "refactor",
    "test_adjustment",
    "relax_assertion",
)

METADATA_FIELDS_BY_TYPE: Mapping[str, Sequence[str]] = {
    "problem_statement": ("problem_kind", "affected_area", "user_or_system_impact"),
    "proposed_solution": ("solution_kind", "target_area", "expected_benefit"),
    "alternative_solution": ("alternative_to", "differentiator", "comparison_focus"),
    "design_decision": ("decision_status", "chosen_option", "rejected_options"),
    "trade_off_argument": ("options_compared", "comparison_axes", "preferred_option"),
    "rationale": ("supports_artifact_type", "reasoning_kind", "impact_scope"),
    "constraint": ("constraint_type", "scope", "severity"),
    "assumption": ("assumption_scope", "confidence", "failure_mode"),
    "implementation_detail": ("component", "lifecycle_stage", "implementation_action"),
    "code_snippet": ("language", "snippet_kind", "intended_use"),
    "algorithm_approach": ("approach_name", "target_problem", "scale_or_complexity_notes"),
    "api_design": ("interface_kind", "api_surface", "compatibility_impact"),
    "data_structure_choice": ("data_structure", "workload", "motivation"),
    "configuration_choice": ("parameter", "selected_value", "config_scope"),
    "benchmark_result": ("metric", "baseline", "candidate", "environment"),
    "performance_claim": ("metric_focus", "claim_direction", "workload"),
    "test_case": ("test_level", "coverage_target", "expected_outcome"),
    "bug_reproduction_steps": ("environment", "prerequisites", "determinism"),
    "edge_case": ("trigger", "impact", "rarity"),
    "empirical_evidence": ("evidence_type", "sample_source", "reliability"),
    "question": ("question_kind", "target_area", "urgency"),
    "answer_clarification": ("target_question_topic", "certainty", "resolution_status"),
    "agreement": ("target", "strength", "justification"),
    "disagreement": ("target", "contention_point", "severity"),
    "suggestion": ("target_area", "expected_improvement", "urgency"),
    "critique": ("target", "critique_dimension", "severity"),
    "task_assignment": ("assignee", "deliverable", "due_context"),
    "status_update": ("status_kind", "progress_state", "blocking"),
    "priority_discussion": ("priority_level", "compared_against", "scope"),
    "blocking_issue": ("blocker_type", "blocking_target", "unblock_condition"),
    "dependency": ("dependency_type", "dependency_target", "relationship"),
}

SYSTEM_PROMPT = """Extract discussion artifacts from one GitHub comment.
Return strict JSON only: {{"artifacts": [...]}}.

Rules:
1) type: required, one of [{allowed_types}].
2) summary: required, specific key takeaway (not generic). Include
   - subject (component/file/feature)
   - event (bug/change/question/suggestion)
   - key technical detail (error/behavior/number/condition/cause if known)
3) tags: required, at least 1 tag, use as many as needed, unique, lowercase snake_case, only from [{allowed_tags}].
    Be generous with relevant tags: include all tags clearly supported by the comment.
    For each candidate tag, explicitly check if evidence in the comment supports it; add it when supported, skip when not supported.
    Prefer broader correct coverage over minimal tagging, but do not add speculative tags.
4) metadata: required; include exactly the fixed keys for the chosen type, never use null, no extra keys.
5) Use only the provided comment text; no outside knowledge.
6) If no valid artifact exists, return {{"artifacts": []}}.
""".format(
    allowed_types=", ".join(ARTIFACT_TYPES),
    allowed_tags=", ".join(ALLOWED_TAGS),
)


def _build_metadata_rules_text() -> str:
    lines = ["Metadata keys by type (use exact keys; never null; use 'unknown' when needed):"]
    for artifact_type in ARTIFACT_TYPES:
        fields = ", ".join(METADATA_FIELDS_BY_TYPE[artifact_type])
        lines.append(f"- {artifact_type}: {fields}")
    return "\n".join(lines)


METADATA_RULES_TEXT = _build_metadata_rules_text()


def _comment_link(comment: JSONDict) -> str:
    for key in ("url", "html_url", "web_url"):
        value = comment.get(key)
        if value:
            return str(value)
    return ""


def build_comment_prompt_job(
    record: JSONDict,
    comment: JSONDict,
    index: int,
) -> JSONDict:
    """Builds one model request payload for a single comment."""
    repository = str(record.get("repository", ""))
    issue_number = record.get("issue_number")
    issue_url = str(record.get("issue_url", ""))

    comment_payload = {
        "author": str(comment.get("author_login", "")),
        "link": _comment_link(comment),
        "text": str(comment.get("body", "")),
    }

    user_prompt = "\n".join(
        (
            f"Repo: {repository}",
            f"Issue: {issue_number}",
            "Comment JSON:",
            json.dumps(comment_payload, ensure_ascii=True, separators=(",", ":")),
            METADATA_RULES_TEXT,
        )
    )

    return {
        "prompt_version": PROMPT_VERSION,
        "repository": repository,
        "issue_number": issue_number,
        "issue_url": issue_url,
        "comment_index": index,
        "comment_author": comment_payload["author"],
        "comment_link": comment_payload["link"],
        "comment_text": comment_payload["text"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def iter_comment_prompt_jobs(
    record: JSONDict,
    include_issue_body: bool = False,
) -> Iterator[JSONDict]:
    """Yields one prompt job per discussion comment for extraction."""
    thread = record.get("deliberation_thread", [])
    for idx, comment in enumerate(thread):
        if not isinstance(comment, dict):
            continue
        if not include_issue_body and comment.get("is_issue_body"):
            continue
        yield build_comment_prompt_job(record=record, comment=comment, index=idx)

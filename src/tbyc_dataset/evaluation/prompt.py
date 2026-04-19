from __future__ import annotations

from typing import Iterable, Mapping, Sequence


THOUGHT_TOPICS: Sequence[str] = (
    "Problem Statement",
    "Proposed Solution",
    "Alternative Solution",
    "Design Decision",
    "Trade-off Argument",
    "Rationale",
    "Constraint",
    "Assumption",
    "Implementation Detail",
    "Code Snippet",
    "Algorithm / Approach",
    "API Design",
    "Data Structure Choice",
    "Configuration Choice",
    "Benchmark Result",
    "Performance Claim",
    "Test Case",
    "Bug Reproduction Steps",
    "Edge Case",
    "Empirical Evidence",
    "Question",
    "Answer / Clarification",
    "Agreement",
    "Disagreement",
    "Suggestion",
    "Critique",
    "Task Assignment",
    "Status Update",
    "Priority Discussion",
    "Blocking Issue",
    "Dependency",
)


def build_issue_thought_prompt(
    issue: Mapping[str, object],
    *,
    include_context: bool,
    context_blocks: Iterable[Mapping[str, object]],
) -> str:
    title = str(issue.get("title") or "").strip()
    body = str(issue.get("body") or "").strip()
    issue_number = issue.get("number")

    sections = [
        "Read the issue and think like a human engineer.",
        "Write at least 20 short comments. Conflicting or unrelated comments are allowed.",
        "Each line must follow: [Topic] comment",
        "Use only these topics:",
        "; ".join(THOUGHT_TOPICS),
        "",
        f"Issue #{issue_number}: {title}" if issue_number is not None else f"Issue: {title}",
        body,
    ]

    if include_context:
        rendered_blocks = list(_render_context_blocks(context_blocks))
        if rendered_blocks:
            sections.extend(
                [
                    "",
                    "Code context:",
                    "\n\n".join(rendered_blocks),
                ]
            )
        else:
            sections.extend(["", "Code context: [none]"])
    else:
        sections.extend(["", "Code context omitted by configuration."])

    return "\n".join(sections).strip() + "\n"


def _render_context_blocks(blocks: Iterable[Mapping[str, object]]) -> Iterable[str]:
    for block in blocks:
        path = str(block.get("file_path") or "unknown")
        start_line = block.get("start_line")
        end_line = block.get("end_line")
        symbol = str(block.get("symbol_name") or "").strip()
        text = str(block.get("text") or "").strip()
        if not text:
            continue

        location = path
        if isinstance(start_line, int) and isinstance(end_line, int):
            location = f"{path}:{start_line}-{end_line}"
        symbol_segment = f" ({symbol})" if symbol else ""
        yield f"[{location}{symbol_segment}]\n{text}"
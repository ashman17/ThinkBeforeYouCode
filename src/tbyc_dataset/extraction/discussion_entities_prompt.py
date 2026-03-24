from __future__ import annotations

import textwrap
from typing import Any, Dict, List


PROMPT_VERSION = "discussion_entities_v4"

ENTITY_CLASSES = {
    "Problem Statement": "Description of a bug, feature request, or user need.",
    "Proposed Solution": "Suggested fix, implementation, or approach.",
    "Alternative Solution": "Competing or different approach to solve the same problem.",
    "Design Decision": "Final or tentative choice among alternatives.",
    "Trade-off Argument": "Comparison between options highlighting pros and cons.",
    "Rationale": "Explanation or justification for a decision.",
    "Constraint": "Limitation or requirement such as performance or compatibility.",
    "Assumption": "Stated or implicit premise about the system or inputs.",
    "Implementation Detail": "Low-level description of how something should be built.",
    "Code Snippet": "Concrete code example or patch.",
    "Algorithm / Approach": "High-level method or technique.",
    "API Design": "Discussion of interfaces, endpoints, or function signatures.",
    "Data Structure Choice": "Selection of data structures and their implications.",
    "Configuration Choice": "Decisions about parameters or settings.",
    "Benchmark Result": "Measured performance or experimental result.",
    "Performance Claim": "Assertion about speed, memory, or efficiency.",
    "Test Case": "Example used to validate correctness.",
    "Bug Reproduction Steps": "Steps to reproduce an issue.",
    "Edge Case": "Rare or boundary scenario that needs handling.",
    "Empirical Evidence": "Data or experiments supporting a claim.",
    "Question": "Inquiry asking for clarification or information.",
    "Answer / Clarification": "Response resolving a question.",
    "Agreement": "Expression of alignment or approval.",
    "Disagreement": "Expression of conflict or opposing view.",
    "Suggestion": "Recommendation for improvement.",
    "Critique": "Evaluation or review of a proposal.",
    "Task Assignment": "Delegation of work to a contributor.",
    "Status Update": "Progress or current state of work.",
    "Priority Discussion": "Conversation about importance or urgency.",
    "Blocking Issue": "Issue preventing further progress.",
    "Dependency": "Relationship to another issue, module, or system.",
    "Reference (Code)": "Pointer to a file, function, or code location.",
    "Reference (Pull Request)": "Link or mention of a PR.",
    "Reference (External)": "Citation of external resource such as a paper, blog, or tool.",
}

ALLOWED_CLASSES_BLOCK = "\n".join(
    f"- {class_name}: {description}"
    for class_name, description in ENTITY_CLASSES.items()
)

PROMPT_DESCRIPTION = textwrap.dedent(
    f"""\
    You are extracting strategic reasoning entities from an entire GitHub issue discussion.

    The benchmark goal is not generic NER. It is to capture the kinds of engineering judgments
    that reveal whether a model reasons like experienced developers before writing code:
    problem framing, risk surfacing, alternatives, constraints, evidence, decisions, blockers,
    and work coordination that changes what should be built.

    The input is a multi-speaker thread formatted as one utterance per line:
    author: comment
    author: comment
    ...

    Allowed extraction classes:
    {ALLOWED_CLASSES_BLOCK}

    Thread-level extraction rules:
    - Use exact contiguous text from the thread. Do not paraphrase.
    - Preserve order of appearance across the full thread.
    - Prefer one extraction per atomic engineering point.
    - Use the smallest span that still preserves the full meaning.
    - Ignore greetings, thanks, social chatter, and empty coordination unless they express one of the allowed classes.
    - Do not invent missing facts. If a field is not supported by the text, omit it.
    - If a line contains multiple distinct reasoning points, split them into separate extractions.

    Attribute rules:
    - Always include `speaker`, `topic`, and `factors`.
    - `speaker` must be the thread author name from the line prefix.
    - `topic` must be a short normalized engineering topic such as `wallet debug endpoint`,
      `peer lookup structure`, or `message ordering`.
    - `factors` must contain 1-4 short, atomic, engineering-specific considerations.
    - Do not use vague factors such as `better design`, `more robust`, or `cleaner`.
    - Prefer factors such as `data leak risk`, `MITM attack surface`, `O(log n) lookup cost`,
      `startup latency regression`, `ABI compatibility`, or `single-machine benchmark only`.
    - Include `stance` when the speaker is proposing, preferring, approving, questioning,
      disagreeing, blocking, or expressing concern.
    - When the span contains quantitative evidence, include normalized fields such as
      `metric`, `baseline`, `candidate`, `delta`, `sample_size`, `environment`,
      or `config_scope`.
    - When the span contains a decision, include normalized fields such as
      `decision_status`, `chosen_option`, `rejected_option`, `comparison_axis`,
      `decision_scope`, or `impact_area`.
    - When the span contains process constraints, blockers, or prioritization, include
      fields such as `constraint_type`, `dependency_target`, `blocking_target`,
      `priority_level`, `assignee`, or `deliverable`.
    - Keep attributes flat. Use strings, numbers, booleans, or lists of short strings.

    Output rules:
    - Always return valid JSON with a top-level `extractions` key, even if the list is empty.
    - Each extraction must contain `extraction_class`, `extraction_text`, and `attributes`.
    - Return ONLY a valid JSON object.
    - `extraction_text` must be a verbatim span from the input thread.
    """
)


EXAMPLE_SPECS: List[Dict[str, Any]] = [
    {
        "name": "security_tradeoff_thread",
        "text": (
            "alice: The current debug endpoint prints wallet addresses in plaintext.\n"
            "bob: We should not ship that as-is because it creates a data leak risk on shared machines.\n"
            "carol: An alternative is to gate it behind -debug=wallet, but that still increases the MITM attack surface for remote setups.\n"
            "bob: Design decision: remove the address payload entirely and keep only counts.\n"
            "bob: The encryption overhead is negligible compared with the privacy risk."
        ),
        "extractions": [
            {
                "extraction_class": "Problem Statement",
                "extraction_text": "The current debug endpoint prints wallet addresses in plaintext.",
                "attributes": {
                    "speaker": "alice",
                    "topic": "wallet debug endpoint",
                    "impact_area": "debug payload privacy",
                    "severity": "high",
                    "factors": [
                        "wallet addresses exposed",
                        "plaintext output",
                        "privacy-sensitive data",
                    ],
                },
            },
            {
                "extraction_class": "Critique",
                "extraction_text": "We should not ship that as-is because it creates a data leak risk on shared machines.",
                "attributes": {
                    "speaker": "bob",
                    "topic": "shipping plaintext addresses",
                    "stance": "concern",
                    "impact_area": "shared-machine deployments",
                    "risk_level": "high",
                    "factors": [
                        "data leak risk",
                        "shared-machine exposure",
                        "unsafe default behavior",
                    ],
                },
            },
            {
                "extraction_class": "Alternative Solution",
                "extraction_text": "An alternative is to gate it behind -debug=wallet,",
                "attributes": {
                    "speaker": "carol",
                    "topic": "debug payload exposure control",
                    "stance": "propose",
                    "chosen_option": "gate behind -debug=wallet",
                    "impact_area": "debug endpoint access path",
                    "factors": [
                        "opt-in visibility",
                        "preserve debugging workflow",
                        "reduce default exposure",
                    ],
                },
            },
            {
                "extraction_class": "Trade-off Argument",
                "extraction_text": "that still increases the MITM attack surface for remote setups.",
                "attributes": {
                    "speaker": "carol",
                    "topic": "debug payload exposure control",
                    "stance": "concern",
                    "comparison_axis": "privacy protection vs remote debugging observability",
                    "preferred_option": "remove address payload",
                    "rejected_option": "gate behind debug flag",
                    "risk_level": "high",
                    "factors": [
                        "MITM attack surface",
                        "remote setup exposure",
                        "partial mitigation only",
                    ],
                },
            },
            {
                "extraction_class": "Design Decision",
                "extraction_text": "Design decision: remove the address payload entirely and keep only counts.",
                "attributes": {
                    "speaker": "bob",
                    "topic": "debug endpoint payload",
                    "stance": "approve",
                    "decision_status": "final",
                    "chosen_option": "counts only",
                    "decision_scope": "debug endpoint response",
                    "impact_area": "privacy-by-default behavior",
                    "factors": [
                        "eliminate plaintext address exposure",
                        "retain coarse debugging signal",
                        "simpler security posture",
                    ],
                },
            },
            {
                "extraction_class": "Rationale",
                "extraction_text": "The encryption overhead is negligible compared with the privacy risk.",
                "attributes": {
                    "speaker": "bob",
                    "topic": "security overhead comparison",
                    "stance": "justify",
                    "comparison_axis": "privacy risk vs encryption overhead",
                    "supports_decision": "remove address payload",
                    "factors": [
                        "privacy risk dominates",
                        "encryption overhead low",
                    ],
                },
            },
        ],
    },
    {
        "name": "api_refactor_thread",
        "text": (
            "dave: Proposed solution: add a ScanOptions struct so we stop threading three booleans through the wallet scan path.\n"
            "erin: The API should be `ScanWallet(const ScanOptions&)` instead of another overload.\n"
            "erin: Concrete snippet: `struct ScanOptions { bool rescan; bool include_watchonly; bool skip_reused; };`\n"
            "frank: The current handoff starts in wallet/rpc/import.cpp.\n"
            "frank: Implementation detail: populate the struct before validation, then pass it through unchanged.\n"
            "erin: I agree with this direction because it reduces call-site ambiguity."
        ),
        "extractions": [
            {
                "extraction_class": "Proposed Solution",
                "extraction_text": "Proposed solution: add a ScanOptions struct so we stop threading three booleans through the wallet scan path.",
                "attributes": {
                    "speaker": "dave",
                    "topic": "wallet scan options",
                    "stance": "propose",
                    "solution_type": "parameter object",
                    "target_component": "wallet scan path",
                    "factors": [
                        "remove boolean fan-out",
                        "centralize scan parameters",
                        "improve call-site readability",
                    ],
                },
            },
            {
                "extraction_class": "API Design",
                "extraction_text": "The API should be `ScanWallet(const ScanOptions&)` instead of another overload.",
                "attributes": {
                    "speaker": "erin",
                    "topic": "scan function signature",
                    "stance": "prefer",
                    "chosen_option": "ScanWallet(const ScanOptions&)",
                    "rejected_option": "additional overload",
                    "impact_area": "wallet scan API surface",
                    "factors": [
                        "single canonical entry point",
                        "avoid overload sprawl",
                        "clearer call sites",
                    ],
                },
            },
            {
                "extraction_class": "Code Snippet",
                "extraction_text": "`struct ScanOptions { bool rescan; bool include_watchonly; bool skip_reused; };`",
                "attributes": {
                    "speaker": "erin",
                    "topic": "scan options struct",
                    "artifact_type": "code example",
                    "field_count": 3,
                    "target_component": "scan options struct",
                    "factors": [
                        "explicit flags",
                        "grouped configuration",
                        "named fields over positional booleans",
                    ],
                },
            },
            {
                "extraction_class": "Reference (Code)",
                "extraction_text": "wallet/rpc/import.cpp",
                "attributes": {
                    "speaker": "frank",
                    "topic": "import rpc handoff",
                    "reference_target": "wallet/rpc/import.cpp",
                    "reference_kind": "file path",
                    "factors": [
                        "current option population site",
                    ],
                },
            },
            {
                "extraction_class": "Implementation Detail",
                "extraction_text": "Implementation detail: populate the struct before validation, then pass it through unchanged.",
                "attributes": {
                    "speaker": "frank",
                    "topic": "scan options flow",
                    "stance": "specify",
                    "phase": "pre-validation",
                    "mutation_policy": "immutable-after-population",
                    "target_component": "ScanOptions lifecycle",
                    "factors": [
                        "populate before validation",
                        "single source of truth",
                        "avoid mid-pipeline mutation",
                    ],
                },
            },
            {
                "extraction_class": "Agreement",
                "extraction_text": "I agree with this direction because it reduces call-site ambiguity.",
                "attributes": {
                    "speaker": "erin",
                    "topic": "scan options direction",
                    "stance": "agree",
                    "supports_decision": "parameter object API",
                    "factors": [
                        "reduces call-site ambiguity",
                        "improves maintainability",
                    ],
                },
            },
        ],
    },
    {
        "name": "benchmark_configuration_thread",
        "text": (
            "gina: A sorted vector is the data structure choice I would try first, and binary search is the lookup approach.\n"
            "gina: Performance claim: it should be faster than the bloom filter for the small peer sets we actually have.\n"
            "gina: Benchmark result: median lookup time dropped from 2.8 ms to 1.9 ms with 512 peers.\n"
            "henry: Empirical evidence: that measurement was collected on my laptop with `-dbcache=512` and the default scheduler settings.\n"
            "henry: See BIP157 for the filter semantics."
        ),
        "extractions": [
            {
                "extraction_class": "Data Structure Choice",
                "extraction_text": "A sorted vector is the data structure choice I would try first,",
                "attributes": {
                    "speaker": "gina",
                    "topic": "peer lookup representation",
                    "stance": "prefer",
                    "chosen_option": "sorted vector",
                    "impact_area": "peer filter lookup",
                    "factors": [
                        "small working set",
                        "cache-friendly layout",
                        "low per-entry overhead",
                    ],
                },
            },
            {
                "extraction_class": "Algorithm / Approach",
                "extraction_text": "binary search is the lookup approach.",
                "attributes": {
                    "speaker": "gina",
                    "topic": "peer lookup algorithm",
                    "stance": "prefer",
                    "chosen_option": "binary search",
                    "complexity": "O(log n)",
                    "factors": [
                        "ordered search",
                        "predictable lookup cost",
                    ],
                },
            },
            {
                "extraction_class": "Performance Claim",
                "extraction_text": "Performance claim: it should be faster than the bloom filter for the small peer sets we actually have.",
                "attributes": {
                    "speaker": "gina",
                    "topic": "peer lookup performance",
                    "stance": "claim",
                    "comparison_target": "bloom filter",
                    "workload": "small peer sets",
                    "factors": [
                        "less hashing overhead",
                        "small peer population",
                        "cache-friendly access",
                    ],
                },
            },
            {
                "extraction_class": "Benchmark Result",
                "extraction_text": "Benchmark result: median lookup time dropped from 2.8 ms to 1.9 ms with 512 peers.",
                "attributes": {
                    "speaker": "gina",
                    "topic": "peer lookup benchmark",
                    "metric": "median lookup time",
                    "baseline": "2.8 ms",
                    "candidate": "1.9 ms",
                    "delta": "-0.9 ms",
                    "sample_size": 512,
                    "factors": [
                        "512-peer workload",
                        "measured latency reduction",
                    ],
                },
            },
            {
                "extraction_class": "Empirical Evidence",
                "extraction_text": "Empirical evidence: that measurement was collected on my laptop",
                "attributes": {
                    "speaker": "henry",
                    "topic": "benchmark measurement conditions",
                    "evidence_type": "single-machine benchmark",
                    "environment": "local laptop",
                    "factors": [
                        "single machine",
                        "configuration dependent",
                        "not production traffic",
                    ],
                },
            },
            {
                "extraction_class": "Configuration Choice",
                "extraction_text": "with `-dbcache=512` and the default scheduler settings.",
                "attributes": {
                    "speaker": "henry",
                    "topic": "benchmark configuration",
                    "config_scope": "measurement setup",
                    "parameters": [
                        "-dbcache=512",
                        "default scheduler settings",
                    ],
                    "factors": [
                        "fixed dbcache",
                        "default scheduler",
                    ],
                },
            },
            {
                "extraction_class": "Reference (External)",
                "extraction_text": "BIP157",
                "attributes": {
                    "speaker": "henry",
                    "topic": "filter semantics reference",
                    "reference_target": "BIP157",
                    "reference_purpose": "filter semantics",
                    "factors": [
                        "protocol semantics source",
                    ],
                },
            },
        ],
    },
    {
        "name": "ordering_blocker_thread",
        "text": (
            "ivy: Question: do we assume compact blocks always arrive before headers?\n"
            "jane: Answer: no, the node has to work when compact blocks arrive first.\n"
            "jane: The old assumption in the reconnect logic was that the peer always sends compact blocks before headers.\n"
            "jane: Constraint: we cannot reorder messages across network threads.\n"
            "kate: The remaining blocker is PR #31210, and this patch depends on the descriptor cache landing first.\n"
            "kate: Track the integration in PR #31210.\n"
            "jane: Status update: tests are green locally.\n"
            "jane: This is lower priority than the startup regression.\n"
            "jane: Alice, can you take the wallet-side tests?"
        ),
        "extractions": [
            {
                "extraction_class": "Question",
                "extraction_text": "Question: do we assume compact blocks always arrive before headers?",
                "attributes": {
                    "speaker": "ivy",
                    "topic": "message ordering",
                    "question_type": "protocol ordering",
                    "factors": [
                        "compact blocks",
                        "headers arrival order",
                    ],
                },
            },
            {
                "extraction_class": "Answer / Clarification",
                "extraction_text": "Answer: no, the node has to work when compact blocks arrive first.",
                "attributes": {
                    "speaker": "jane",
                    "topic": "message ordering",
                    "stance": "clarify",
                    "resolution_status": "resolved",
                    "factors": [
                        "must support compact blocks first",
                        "ordering cannot be assumed",
                    ],
                },
            },
            {
                "extraction_class": "Assumption",
                "extraction_text": "The old assumption in the reconnect logic was that the peer always sends compact blocks before headers.",
                "attributes": {
                    "speaker": "jane",
                    "topic": "reconnect ordering assumption",
                    "assumption_scope": "reconnect logic",
                    "assumption_state": "invalidated",
                    "factors": [
                        "peer ordering assumption",
                        "legacy reconnect path",
                    ],
                },
            },
            {
                "extraction_class": "Constraint",
                "extraction_text": "Constraint: we cannot reorder messages across network threads.",
                "attributes": {
                    "speaker": "jane",
                    "topic": "network thread ordering",
                    "constraint_type": "concurrency",
                    "hardness": "hard",
                    "impact_area": "message scheduling",
                    "factors": [
                        "message order preserved",
                        "network threads independent",
                    ],
                },
            },
            {
                "extraction_class": "Blocking Issue",
                "extraction_text": "The remaining blocker is PR #31210,",
                "attributes": {
                    "speaker": "kate",
                    "topic": "merge blocker",
                    "stance": "blocked",
                    "blocking_target": "PR #31210",
                    "blocking_severity": "hard",
                    "factors": [
                        "pending pull request",
                        "cannot finish integration",
                    ],
                },
            },
            {
                "extraction_class": "Dependency",
                "extraction_text": "this patch depends on the descriptor cache landing first.",
                "attributes": {
                    "speaker": "kate",
                    "topic": "descriptor cache dependency",
                    "dependency_target": "descriptor cache",
                    "dependency_type": "upstream merge prerequisite",
                    "factors": [
                        "ordering dependency",
                        "upstream prerequisite",
                    ],
                },
            },
            {
                "extraction_class": "Reference (Pull Request)",
                "extraction_text": "PR #31210",
                "attributes": {
                    "speaker": "kate",
                    "topic": "blocking pull request",
                    "reference_target": "PR #31210",
                    "reference_purpose": "blocker",
                    "factors": [
                        "pending merge dependency",
                    ],
                },
            },
            {
                "extraction_class": "Status Update",
                "extraction_text": "Status update: tests are green locally.",
                "attributes": {
                    "speaker": "jane",
                    "topic": "patch status",
                    "stance": "status",
                    "validation_state": "local tests green",
                    "factors": [
                        "local tests green",
                        "ready for review signal",
                    ],
                },
            },
            {
                "extraction_class": "Priority Discussion",
                "extraction_text": "This is lower priority than the startup regression.",
                "attributes": {
                    "speaker": "jane",
                    "topic": "release prioritization",
                    "priority_level": "lower",
                    "higher_priority_item": "startup regression",
                    "factors": [
                        "startup regression more urgent",
                        "limited release bandwidth",
                    ],
                },
            },
            {
                "extraction_class": "Task Assignment",
                "extraction_text": "Alice, can you take the wallet-side tests?",
                "attributes": {
                    "speaker": "jane",
                    "topic": "wallet-side tests",
                    "stance": "assign",
                    "assignee": "Alice",
                    "deliverable": "wallet-side tests",
                    "factors": [
                        "delegate validation work",
                        "unblock review",
                    ],
                },
            },
        ],
    },
    {
        "name": "reproduction_and_review_thread",
        "text": (
            "leo: Bug reproduction steps: start with `-blocksonly`, connect two outbound peers, then trigger a mempool rebroadcast.\n"
            "maya: Edge case: the failure only shows up when one peer disconnects during the rebroadcast window.\n"
            "maya: Test case: add coverage where the second peer drops after INV relay but before GETDATA.\n"
            "nick: I disagree with retrying indefinitely because it can hide the liveness bug and grow the outbound queue.\n"
            "nick: Suggestion: cap retries at 2 and surface a debug log when the queue is drained."
        ),
        "extractions": [
            {
                "extraction_class": "Bug Reproduction Steps",
                "extraction_text": "Bug reproduction steps: start with `-blocksonly`, connect two outbound peers, then trigger a mempool rebroadcast.",
                "attributes": {
                    "speaker": "leo",
                    "topic": "rebroadcast failure reproduction",
                    "environment": "blocksonly node",
                    "step_count": 3,
                    "factors": [
                        "blocksonly mode",
                        "two outbound peers",
                        "mempool rebroadcast trigger",
                    ],
                },
            },
            {
                "extraction_class": "Edge Case",
                "extraction_text": "Edge case: the failure only shows up when one peer disconnects during the rebroadcast window.",
                "attributes": {
                    "speaker": "maya",
                    "topic": "rebroadcast disconnect edge case",
                    "rarity": "timing-sensitive",
                    "impact_area": "peer disconnect handling",
                    "factors": [
                        "peer disconnect timing",
                        "rebroadcast window",
                        "non-steady-state failure",
                    ],
                },
            },
            {
                "extraction_class": "Test Case",
                "extraction_text": "Test case: add coverage where the second peer drops after INV relay but before GETDATA.",
                "attributes": {
                    "speaker": "maya",
                    "topic": "rebroadcast disconnect test",
                    "target_component": "peer relay sequencing",
                    "validation_goal": "capture disconnect timing bug",
                    "factors": [
                        "second peer drops",
                        "after INV relay",
                        "before GETDATA",
                    ],
                },
            },
            {
                "extraction_class": "Disagreement",
                "extraction_text": "I disagree with retrying indefinitely because it can hide the liveness bug and grow the outbound queue.",
                "attributes": {
                    "speaker": "nick",
                    "topic": "retry policy",
                    "stance": "disagree",
                    "rejected_option": "indefinite retries",
                    "comparison_axis": "recovery persistence vs bug visibility",
                    "factors": [
                        "liveness bug hidden",
                        "outbound queue growth",
                        "failure masked instead of exposed",
                    ],
                },
            },
            {
                "extraction_class": "Suggestion",
                "extraction_text": "Suggestion: cap retries at 2 and surface a debug log when the queue is drained.",
                "attributes": {
                    "speaker": "nick",
                    "topic": "retry policy",
                    "stance": "suggest",
                    "chosen_option": "cap retries at 2 with debug log",
                    "target_component": "rebroadcast retry handling",
                    "factors": [
                        "bounded retry cost",
                        "explicit failure visibility",
                        "queue-drain observability",
                    ],
                },
            },
        ],
    },
]


def build_langextract_examples(lx: Any) -> List[Any]:
    return [
        lx.data.ExampleData(
            text=spec["text"],
            extractions=[
                lx.data.Extraction(
                    extraction_class=extraction["extraction_class"],
                    extraction_text=extraction["extraction_text"],
                    attributes=extraction["attributes"],
                )
                for extraction in spec["extractions"]
            ],
        )
        for spec in EXAMPLE_SPECS
    ]

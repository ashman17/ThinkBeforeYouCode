from __future__ import annotations

from tbyc_dataset.evaluation.derived import split_response_into_comments


def test_split_response_into_comments_splits_multiple_bracket_sections() -> None:
    text = "[Problem Statement] A fails in env X. [Proposed Solution] Add guard in loader."

    comments = split_response_into_comments(text)

    assert comments == [
        "[Problem Statement] A fails in env X.",
        "[Proposed Solution] Add guard in loader.",
    ]


def test_split_response_into_comments_keeps_single_text_without_topics() -> None:
    text = "No explicit topic markers here."

    comments = split_response_into_comments(text)

    assert comments == ["No explicit topic markers here."]
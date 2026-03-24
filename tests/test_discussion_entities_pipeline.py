import unittest

from tbyc_dataset.extraction.discussion_entities_pipeline import (
    format_thread_text,
    serialize_extraction,
)


class DiscussionEntityPipelineTests(unittest.TestCase):
    def test_format_thread_text_uses_author_prefix_and_tracks_spans(self) -> None:
        source_comments = [
            {
                "comment_index": 0,
                "author_login": "alice",
                "author_association": "MEMBER",
                "created_at": "2026-03-21T10:00:00Z",
                "url": "https://example.test/c1",
                "body": "First line with  extra   spaces.",
            },
            {
                "comment_index": 1,
                "author_login": "bob",
                "author_association": "CONTRIBUTOR",
                "created_at": "2026-03-21T10:05:00Z",
                "url": "https://example.test/c2",
                "body": "Second line.",
            },
        ]

        thread_text, comment_spans = format_thread_text(source_comments)

        self.assertEqual(
            thread_text,
            "alice: First line with extra spaces.\n"
            "bob: Second line.",
        )
        self.assertEqual(comment_spans[0]["start_pos"], 0)
        self.assertEqual(
            comment_spans[1]["start_pos"],
            len("alice: First line with extra spaces.\n"),
        )
        self.assertEqual(comment_spans[0]["speaker_role"], "maintainer")
        self.assertEqual(comment_spans[1]["speaker_role"], "contributor")

    def test_serialize_extraction_attaches_source_metadata(self) -> None:
        source_comments = [
            {
                "comment_index": 0,
                "author_login": "alice",
                "author_association": "COLLABORATOR",
                "created_at": "2026-03-21T10:00:00Z",
                "url": "https://example.test/c1",
                "body": "We should not ship that as-is because it creates a data leak risk.",
            }
        ]
        thread_text, comment_spans = format_thread_text(source_comments)
        extraction_text = "We should not ship that as-is because it creates a data leak risk."
        start_pos = thread_text.index(extraction_text)
        end_pos = start_pos + len(extraction_text)

        class FakeExtraction:
            def __init__(self) -> None:
                self.extraction_class = "Critique"
                self.attributes = {
                    "topic": "shipping risk",
                    "factors": ["data leak risk"],
                }
                self.char_interval = {
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                }
                self.extraction_text = extraction_text

        serialized = serialize_extraction(FakeExtraction(), comment_spans)

        self.assertEqual(serialized["source_author_login"], "alice")
        self.assertEqual(serialized["speaker_role"], "maintainer")
        self.assertEqual(serialized["source_comment_index"], 0)
        self.assertEqual(serialized["attributes"]["speaker"], "alice")
        self.assertEqual(serialized["attributes"]["speaker_role"], "maintainer")


if __name__ == "__main__":
    unittest.main()

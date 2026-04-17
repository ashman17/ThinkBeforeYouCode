import unittest

from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.dataset.normalize import normalize_issue


class NormalizeIssueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = RepositoryRef(owner="octo", name="demo")

    def test_preserves_linked_pull_request_facts(self) -> None:
        raw_issue = {
            "number": 10,
            "url": "https://github.com/octo/demo/issues/10",
            "title": "Add feature",
            "body": "Please add it.",
            "state": "CLOSED",
            "stateReason": "COMPLETED",
            "createdAt": "2025-01-01T00:00:00Z",
            "closedAt": "2025-01-02T00:00:00Z",
            "authorAssociation": "NONE",
            "author": {"login": "reporter"},
            "labels": [{"name": "feature"}],
            "comments": [],
            "timelineItems": [
                {
                    "__typename": "CrossReferencedEvent",
                    "createdAt": "2025-01-02T00:00:00Z",
                    "willCloseTarget": True,
                    "source": {
                        "__typename": "PullRequest",
                        "number": 99,
                        "url": "https://github.com/octo/demo/pull/99",
                        "title": "Implement feature",
                        "state": "MERGED",
                        "merged": True,
                        "mergedAt": "2025-01-02T00:00:00Z",
                    },
                }
            ],
        }

        record = normalize_issue(raw_issue, self.repo)

        self.assertNotIn("ground_truth", record)
        self.assertNotIn("triage_signals", record)
        self.assertEqual(record["resolution_artifacts"]["linked_pull_requests"][0]["number"], 99)
        self.assertEqual(record["timeline_events"][0]["event_type"], "CrossReferencedEvent")
        self.assertEqual(record["deliberation_thread"][0]["author_login"], "reporter")
        self.assertTrue(record["deliberation_thread"][0]["is_issue_body"])
        self.assertEqual(
            record["formatted_discussion"],
            "reporter (external):Add feature Please add it.",
        )

    def test_preserves_comments_without_inferred_labels(self) -> None:
        raw_issue = {
            "number": 20,
            "url": "https://github.com/octo/demo/issues/20",
            "title": "Repeated bug",
            "body": "This happened again.",
            "state": "CLOSED",
            "stateReason": "NOT_PLANNED",
            "createdAt": "2025-01-01T00:00:00Z",
            "closedAt": "2025-01-02T00:00:00Z",
            "authorAssociation": "NONE",
            "author": {"login": "reporter"},
            "labels": [{"name": "bug"}],
            "comments": [
                {
                    "id": "c1",
                    "url": "https://github.com/octo/demo/issues/20#issuecomment-1",
                    "body": "Closing as duplicate of #14.",
                    "createdAt": "2025-01-02T00:00:00Z",
                    "authorAssociation": "MEMBER",
                    "author": {"login": "maintainer"},
                }
            ],
            "timelineItems": [],
        }

        record = normalize_issue(raw_issue, self.repo)

        self.assertEqual(record["deliberation_thread"][1]["author_login"], "maintainer")
        self.assertNotIn("ground_truth", record)
        self.assertNotIn("triage_signals", record)
        self.assertEqual(
            record["formatted_discussion"],
            "reporter (external):Repeated bug This happened again.\n"
            "maintainer (member):Closing as duplicate of #14.",
        )

    def test_tracks_actor_associations_as_facts(self) -> None:
        raw_issue = {
            "number": 30,
            "url": "https://github.com/octo/demo/issues/30",
            "title": "Odd request",
            "body": "Could you support this?",
            "state": "CLOSED",
            "stateReason": "NOT_PLANNED",
            "createdAt": "2025-01-01T00:00:00Z",
            "closedAt": "2025-01-02T00:00:00Z",
            "authorAssociation": "NONE",
            "author": {"login": "reporter"},
            "labels": [{"name": "feature"}],
            "comments": [],
            "timelineItems": [],
        }

        record = normalize_issue(raw_issue, self.repo)

        self.assertEqual(record["issue_author"]["login"], "reporter")
        self.assertEqual(record["actor_typology"][0]["author_associations"], ["NONE"])
        self.assertEqual(record["actor_typology"][0]["roles"], ["issue_author"])

    def test_extracts_explicit_file_references_or_empty_list(self) -> None:
        raw_issue = {
            "number": 40,
            "url": "https://github.com/octo/demo/issues/40",
            "title": "Crash in parser",
            "body": "The failure starts in src/parser/main.py and later touches config.yaml.",
            "state": "OPEN",
            "stateReason": None,
            "createdAt": "2025-01-01T00:00:00Z",
            "closedAt": None,
            "authorAssociation": "CONTRIBUTOR",
            "author": {"login": "reporter"},
            "labels": [],
            "comments": [
                {
                    "id": "c1",
                    "url": "https://github.com/octo/demo/issues/40#issuecomment-1",
                    "body": "I also saw this in tests/test_parser.py.",
                    "createdAt": "2025-01-02T00:00:00Z",
                    "authorAssociation": "MEMBER",
                    "author": {"login": "maintainer"},
                }
            ],
            "timelineItems": [],
        }

        record = normalize_issue(raw_issue, self.repo)

        self.assertEqual(
            record["files"],
            ["src/parser/main.py", "config.yaml", "tests/test_parser.py"],
        )

        raw_issue["body"] = "No file names mentioned here."
        raw_issue["comments"][0]["body"] = "Still no explicit files."

        record_without_files = normalize_issue(raw_issue, self.repo)

        self.assertEqual(record_without_files["files"], [])


if __name__ == "__main__":
    unittest.main()

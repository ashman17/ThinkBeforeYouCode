import json
import tempfile
import unittest
from pathlib import Path

from tbyc_dataset.storage import write_jsonl
from tbyc_dataset.viewer import build_processed_viewer, build_viewer_payload


class ViewerTests(unittest.TestCase):
    def test_build_viewer_payload_collects_repositories_and_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            repo_dir = output_root / "processed" / "octo__demo"
            write_jsonl(
                repo_dir / "curated.jsonl",
                [
                    {
                        "repository": "octo/demo",
                        "issue_number": 12,
                        "issue_url": "https://example.test/issues/12",
                        "files": ["src/app.py"],
                        "formatted_discussion": "alice (maintainer):Looks good",
                        "deliberation_thread": [
                            {
                                "author_login": "alice",
                                "author_association": "MEMBER",
                                "body": "Looks good",
                            }
                        ],
                        "input_vector": {"title": "Fix app", "body": "Issue body"},
                        "taxonomic_metadata": {
                            "labels": ["bug"],
                            "issue_state": "OPEN",
                            "created_at": "2026-01-01T00:00:00Z",
                            "closed_at": None,
                        },
                    }
                ],
            )

            payload = build_viewer_payload(output_root)

            self.assertEqual(len(payload["repositories"]), 1)
            repo = payload["repositories"][0]
            self.assertEqual(repo["repository"], "octo/demo")
            self.assertEqual(repo["issue_count"], 1)
            self.assertEqual(repo["issues"][0]["files"], ["src/app.py"])
            self.assertEqual(
                repo["issues"][0]["discussion_entries"],
                [
                    {
                        "author_id": "alice",
                        "author_role": "member",
                        "content": "Looks good",
                    }
                ],
            )
            self.assertEqual(
                repo["issues"][0]["formatted_discussion"],
                "alice (member):Looks good",
            )

    def test_build_processed_viewer_writes_html_with_embedded_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            repo_dir = output_root / "processed" / "octo__demo"
            write_jsonl(
                repo_dir / "curated.jsonl",
                [
                    {
                        "repository": "octo/demo",
                        "issue_number": 7,
                        "issue_url": "https://example.test/issues/7",
                        "files": [],
                        "formatted_discussion": "reporter (external):Problem statement",
                        "deliberation_thread": [
                            {
                                "author_login": "reporter",
                                "author_association": "NONE",
                                "body": "Problem statement",
                            }
                        ],
                        "input_vector": {"title": "Viewer issue", "body": "Body"},
                        "taxonomic_metadata": {
                            "labels": [],
                            "issue_state": "CLOSED",
                            "created_at": "2026-01-02T00:00:00Z",
                            "closed_at": "2026-01-03T00:00:00Z",
                        },
                    }
                ],
            )

            result = build_processed_viewer(output_root)

            viewer_path = Path(result["viewer_path"])
            self.assertTrue(viewer_path.exists())
            html = viewer_path.read_text(encoding="utf-8")
            self.assertIn("Processed Dataset Viewer", html)
            self.assertIn('"author_id": "reporter"', html)
            self.assertIn('"author_role": "external"', html)
            self.assertIn('"content": "Problem statement"', html)
            self.assertIn("discussion-speaker", html)
            self.assertIn("octo/demo", html)
            self.assertIn(json.dumps("https://example.test/issues/7")[1:-1], html)

    def test_viewer_preserves_member_and_collaborator_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            repo_dir = output_root / "processed" / "octo__demo"
            write_jsonl(
                repo_dir / "curated.jsonl",
                [
                    {
                        "repository": "octo/demo",
                        "issue_number": 9,
                        "issue_url": "https://example.test/issues/9",
                        "files": [],
                        "formatted_discussion": (
                            "wyverald (member):Could you diff the two configs?\n"
                            "fmeum (collaborator):Just a guess at this point though."
                        ),
                        "deliberation_thread": [
                            {
                                "author_login": "wyverald",
                                "author_association": "MEMBER",
                                "body": "Could you diff the two configs?",
                            },
                            {
                                "author_login": "fmeum",
                                "author_association": "COLLABORATOR",
                                "body": "Just a guess at this point though.",
                            },
                        ],
                        "input_vector": {"title": "Role test", "body": "Body"},
                        "taxonomic_metadata": {
                            "labels": [],
                            "issue_state": "OPEN",
                            "created_at": "2026-01-02T00:00:00Z",
                            "closed_at": None,
                        },
                    }
                ],
            )

            payload = build_viewer_payload(output_root)

            self.assertEqual(payload["repositories"][0]["issues"][0]["discussion_entries"][0]["author_role"], "member")
            self.assertEqual(payload["repositories"][0]["issues"][0]["discussion_entries"][1]["author_role"], "collaborator")


if __name__ == "__main__":
    unittest.main()

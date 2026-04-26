import json
import tempfile
import unittest
from pathlib import Path

from tbyc_dataset.viewer import build_processed_viewer, build_viewer_payload


class ViewerTests(unittest.TestCase):
    def test_build_viewer_payload_collects_raw_extractions_derived_and_leaderboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)

            raw_issue_path = output_root / "raw" / "octo__demo" / "issues" / "issue_12.json"
            raw_issue_path.parent.mkdir(parents=True, exist_ok=True)
            raw_issue_path.write_text(
                json.dumps(
                    {
                        "number": 12,
                        "title": "Fix app",
                        "body": "Issue body",
                        "url": "https://example.test/issues/12",
                        "state": "OPEN",
                        "stateReason": "",
                        "createdAt": "2026-01-01T00:00:00Z",
                        "closedAt": None,
                        "labels": ["bug"],
                        "comments": [
                            {
                                "author": {"login": "alice"},
                                "authorAssociation": "MEMBER",
                                "createdAt": "2026-01-01T01:00:00Z",
                                "url": "https://example.test/issues/12#issuecomment-1",
                                "body": "Looks good",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            extraction_path = output_root / "extractions" / "octo__demo" / "issue_12.json"
            extraction_path.parent.mkdir(parents=True, exist_ok=True)
            extraction_path.write_text(
                json.dumps(
                    {
                        "issue": {
                            "artifact_count": 1,
                            "comments": [
                                {
                                    "comment_author": "alice",
                                    "comment_link": "https://example.test/issues/12#issuecomment-1",
                                    "artifact_count": 1,
                                    "artifacts": [
                                        {
                                            "type": "problem_statement",
                                            "summary": "A reproducible issue",
                                            "tags": ["bug"],
                                            "metadata": {"affected_area": "app"},
                                        }
                                    ],
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            derived_path = output_root / "derived" / "gpt-4.1-mini" / "octo__demo" / "issue_12.json"
            derived_path.parent.mkdir(parents=True, exist_ok=True)
            derived_path.write_text(
                json.dumps(
                    {
                        "issue": {
                            "artifact_count": 1,
                            "comments": [
                                {
                                    "comment_author": "llm",
                                    "comment_link": "",
                                    "artifact_count": 1,
                                    "artifacts": [
                                        {
                                            "type": "suggestion",
                                            "summary": "Try patch X",
                                            "tags": ["idea"],
                                            "metadata": {},
                                        }
                                    ],
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            leaderboard_path = output_root / "metrics" / "leaderboard_rank_fusion.json"
            leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
            leaderboard_path.write_text(
                json.dumps(
                    {
                        "repo_count": 1,
                        "model_count": 1,
                        "all_repos_combined": {
                            "leaderboard": [
                                {
                                    "rank": 1,
                                    "model_id": "gpt-4.1-mini",
                                    "normalized_score": 91.2,
                                    "points": 92,
                                    "component_values": {
                                        "metadata_f1": 0.45,
                                        "metadata_soft_f1": 0.5,
                                        "tag_f1": 0.4,
                                        "type_f1": 0.6,
                                        "summary_bleurt": 0.3,
                                        "summary_bertscore_f1": 0.7,
                                        "summary_codebert": 0.8,
                                    },
                                }
                            ]
                        },
                        "per_repo": [
                            {
                                "repository": "octo/demo",
                                "leaderboard": [
                                    {
                                        "rank": 1,
                                        "model_id": "gpt-4.1-mini",
                                        "normalized_score": 91.2,
                                        "points": 92,
                                        "component_values": {"metadata_f1": 0.45},
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_viewer_payload(output_root)

            self.assertEqual(len(payload["repositories"]), 1)
            repo = payload["repositories"][0]
            self.assertEqual(repo["repository"], "octo/demo")
            self.assertEqual(repo["issue_count"], 1)
            self.assertEqual(repo["issues"][0]["comments"][0]["author_id"], "alice")
            self.assertEqual(payload["raw_extractions"]["entries"][0]["artifact_count"], 1)
            self.assertEqual(payload["response_extractions"]["entries"][0]["model_id"], "gpt-4.1-mini")
            self.assertTrue(payload["leaderboard"]["available"])
            self.assertEqual(payload["leaderboard"]["global"][0]["model_id"], "gpt-4.1-mini")

    def test_build_processed_viewer_writes_html_with_embedded_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            raw_issue_path = output_root / "raw" / "octo__demo" / "issues" / "issue_7.json"
            raw_issue_path.parent.mkdir(parents=True, exist_ok=True)
            raw_issue_path.write_text(
                json.dumps(
                    {
                        "number": 7,
                        "title": "Viewer issue",
                        "body": "Body",
                        "url": "https://example.test/issues/7",
                        "state": "CLOSED",
                        "createdAt": "2026-01-02T00:00:00Z",
                        "closedAt": "2026-01-03T00:00:00Z",
                        "labels": [],
                        "comments": [
                            {
                                "author": {"login": "reporter"},
                                "authorAssociation": "NONE",
                                "body": "Problem statement",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = build_processed_viewer(output_root)

            viewer_path = Path(result["viewer_path"])
            self.assertTrue(viewer_path.exists())
            html = viewer_path.read_text(encoding="utf-8")
            self.assertIn("TBYC Arena Viewer", html)
            self.assertIn('"author_id": "reporter"', html)
            self.assertIn('"author_role": "external"', html)
            self.assertIn('"body": "Problem statement"', html)
            self.assertIn("tab-btn", html)
            self.assertIn("octo/demo", html)
            self.assertIn(json.dumps("https://example.test/issues/7")[1:-1], html)

    def test_viewer_preserves_member_and_collaborator_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            raw_issue_path = output_root / "raw" / "octo__demo" / "issues" / "issue_9.json"
            raw_issue_path.parent.mkdir(parents=True, exist_ok=True)
            raw_issue_path.write_text(
                json.dumps(
                    {
                        "number": 9,
                        "title": "Role test",
                        "body": "Body",
                        "url": "https://example.test/issues/9",
                        "state": "OPEN",
                        "labels": [],
                        "comments": [
                            {
                                "author": {"login": "wyverald"},
                                "authorAssociation": "MEMBER",
                                "body": "Could you diff the two configs?",
                            },
                            {
                                "author": {"login": "fmeum"},
                                "authorAssociation": "COLLABORATOR",
                                "body": "Just a guess at this point though.",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_viewer_payload(output_root)

            self.assertEqual(payload["repositories"][0]["issues"][0]["comments"][0]["author_role"], "member")
            self.assertEqual(payload["repositories"][0]["issues"][0]["comments"][1]["author_role"], "collaborator")


if __name__ == "__main__":
    unittest.main()

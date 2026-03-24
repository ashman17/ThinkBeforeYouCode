import unittest

from tbyc_dataset.config import GitHubSettings
from tbyc_dataset.dataset.github import GitHubGraphQLClient
from tbyc_dataset.models import RepositoryRef


class StubGitHubGraphQLClient(GitHubGraphQLClient):
    def __init__(self, payloads):
        super().__init__(GitHubSettings(token="test-token"))
        self.payloads = list(payloads)

    def execute(self, query, variables):  # type: ignore[override]
        return self.payloads.pop(0)


class GitHubClientTests(unittest.TestCase):
    def test_list_issue_numbers_applies_min_and_max_comment_filters(self) -> None:
        client = StubGitHubGraphQLClient(
            [
                {
                    "data": {
                        "repository": {
                            "issues": {
                                "nodes": [
                                    {"number": 101, "comments": {"totalCount": 0}},
                                    {"number": 102, "comments": {"totalCount": 9}},
                                    {"number": 103, "comments": {"totalCount": 10}},
                                    {"number": 104, "comments": {"totalCount": 25}},
                                    {"number": 105, "comments": {"totalCount": 26}},
                                ],
                                "pageInfo": {
                                    "hasNextPage": False,
                                    "endCursor": None,
                                },
                            }
                        }
                    }
                }
            ]
        )

        issue_numbers = client.list_issue_numbers(
            repo=RepositoryRef(owner="octo", name="demo"),
            states=["OPEN", "CLOSED"],
            min_comments=10,
            max_comments=25,
        )

        self.assertEqual(issue_numbers, [103, 104])


if __name__ == "__main__":
    unittest.main()

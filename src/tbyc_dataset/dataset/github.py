from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from tbyc_dataset.config import GitHubSettings
from tbyc_dataset.dataset.queries import ISSUE_DETAIL_QUERY, ISSUE_LIST_QUERY
from tbyc_dataset.models import JSONDict, RepositoryRef


class GitHubGraphQLError(RuntimeError):
    """Raised when GitHub GraphQL returns an error payload."""


class GitHubGraphQLClient:
    def __init__(self, settings: GitHubSettings) -> None:
        self.settings = settings

    def execute(self, query: str, variables: Dict[str, Any]) -> JSONDict:
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.settings.token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "User-Agent": "tbyc-dataset-pipeline",
        }

        for attempt in range(self.settings.max_retries + 1):
            request = urllib.request.Request(
                self.settings.endpoint,
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.settings.request_timeout_seconds,
                ) as response:
                    raw = response.read().decode("utf-8")
                body = json.loads(raw)
                if body.get("errors"):
                    raise GitHubGraphQLError(json.dumps(body["errors"], indent=2))
                return body
            except GitHubGraphQLError:
                raise
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.settings.max_retries:
                    raise RuntimeError(f"GitHub GraphQL request failed: {exc}") from exc
                time.sleep(2 ** attempt)
            finally:
                if self.settings.sleep_between_requests_seconds > 0:
                    time.sleep(self.settings.sleep_between_requests_seconds)
        raise RuntimeError("GitHub GraphQL request exhausted retries.")

    def list_issue_numbers(
        self,
        repo: RepositoryRef,
        states: List[str],
        max_issues: Optional[int] = None,
        min_comments: int = 0,
        max_comments: Optional[int] = None,
    ) -> List[int]:
        numbers: List[int] = []
        cursor: Optional[str] = None

        while True:
            variables = {
                "owner": repo.owner,
                "name": repo.name,
                "cursor": cursor,
                "pageSize": self.settings.issue_page_size,
                "states": states,
            }
            payload = self.execute(ISSUE_LIST_QUERY, variables)
            repository = payload["data"]["repository"]
            if repository is None:
                raise ValueError(f"Repository not found: {repo.slug}")

            issues = repository["issues"]
            nodes = issues["nodes"] or []
            for node in nodes:
                comment_count = (node.get("comments") or {}).get("totalCount", 0)
                if comment_count < min_comments:
                    continue
                if max_comments is not None and comment_count > max_comments:
                    continue
                numbers.append(node["number"])
                if max_issues is not None and len(numbers) >= max_issues:
                    return numbers

            page_info = issues["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            cursor = page_info["endCursor"]

        return numbers

    def fetch_issue(self, repo: RepositoryRef, number: int) -> JSONDict:
        comments_cursor: Optional[str] = None
        timeline_cursor: Optional[str] = None
        comments: List[JSONDict] = []
        timeline_items: List[JSONDict] = []
        issue_stub: Optional[JSONDict] = None
        seen_comment_ids = set()
        seen_timeline_signatures = set()

        while True:
            variables = {
                "owner": repo.owner,
                "name": repo.name,
                "number": number,
                "commentsPageSize": self.settings.comment_page_size,
                "commentsCursor": comments_cursor,
                "timelinePageSize": self.settings.timeline_page_size,
                "timelineCursor": timeline_cursor,
            }
            payload = self.execute(ISSUE_DETAIL_QUERY, variables)
            repository = payload["data"]["repository"]
            if repository is None or repository["issue"] is None:
                raise ValueError(f"Issue #{number} not found in {repo.slug}")

            issue = repository["issue"]
            if issue_stub is None:
                issue_stub = {
                    "id": issue["id"],
                    "number": issue["number"],
                    "title": issue["title"],
                    "body": issue["body"],
                    "url": issue["url"],
                    "state": issue["state"],
                    "stateReason": issue["stateReason"],
                    "createdAt": issue["createdAt"],
                    "closedAt": issue["closedAt"],
                    "authorAssociation": issue["authorAssociation"],
                    "author": issue["author"],
                    "labels": issue["labels"]["nodes"] if issue["labels"] else [],
                }

            comment_connection = issue["comments"]
            for comment in comment_connection["nodes"] or []:
                comment_id = comment["id"]
                if comment_id in seen_comment_ids:
                    continue
                seen_comment_ids.add(comment_id)
                comments.append(comment)

            timeline_connection = issue["timelineItems"]
            for item in timeline_connection["nodes"] or []:
                signature = (
                    item.get("__typename"),
                    item.get("createdAt"),
                    json.dumps(item, sort_keys=True),
                )
                if signature in seen_timeline_signatures:
                    continue
                seen_timeline_signatures.add(signature)
                timeline_items.append(item)

            comments_page = comment_connection["pageInfo"]
            timeline_page = timeline_connection["pageInfo"]
            comments_done = not comments_page["hasNextPage"]
            timeline_done = not timeline_page["hasNextPage"]

            if comments_done and timeline_done:
                break

            comments_cursor = comments_page["endCursor"] if not comments_done else comments_page["endCursor"]
            timeline_cursor = timeline_page["endCursor"] if not timeline_done else timeline_page["endCursor"]

        assert issue_stub is not None
        issue_stub["comments"] = comments
        issue_stub["timelineItems"] = timeline_items
        return issue_stub

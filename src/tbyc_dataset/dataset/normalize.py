from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tbyc_dataset.models import JSONDict, RepositoryRef


def normalize_issue(raw_issue: JSONDict, repo: RepositoryRef) -> JSONDict:
    labels = sorted({label["name"] for label in raw_issue.get("labels", []) if label.get("name")})
    comments = sorted(raw_issue.get("comments", []), key=lambda item: item["createdAt"])
    timeline_items = sorted(
        raw_issue.get("timelineItems", []),
        key=lambda item: item.get("createdAt") or "",
    )

    return {
        "repository": repo.slug,
        "issue_number": raw_issue["number"],
        "issue_url": raw_issue["url"],
        "issue_author": {
            "login": nested_login(raw_issue.get("author")),
            "author_association": raw_issue.get("authorAssociation"),
        },
        "input_vector": {
            "title": raw_issue["title"],
            "body": raw_issue.get("body") or "",
        },
        "taxonomic_metadata": {
            "labels": labels,
            "issue_state": raw_issue["state"],
            "state_reason": raw_issue.get("stateReason"),
            "created_at": raw_issue["createdAt"],
            "closed_at": raw_issue.get("closedAt"),
        },
        "deliberation_thread": [
            {
                "url": comment["url"],
                "created_at": comment["createdAt"],
                "author_login": nested_login(comment.get("author")),
                "author_association": comment.get("authorAssociation"),
                "body": comment.get("body") or "",
            }
            for comment in comments
        ],
        "timeline_events": normalize_timeline_items(timeline_items),
        "resolution_artifacts": {
            "linked_pull_requests": extract_linked_pull_requests(timeline_items),
            "issue_state": raw_issue["state"],
            "issue_state_reason": raw_issue.get("stateReason"),
            "closed_at": raw_issue.get("closedAt"),
        },
        "actor_typology": build_actor_typology(raw_issue, comments, timeline_items),
    }


def build_actor_typology(
    raw_issue: JSONDict,
    comments: Sequence[JSONDict],
    timeline_items: Sequence[JSONDict],
) -> List[JSONDict]:
    actors: Dict[str, Dict[str, Any]] = {}

    def register(login: Optional[str], association: Optional[str], role: str) -> None:
        if not login:
            return
        current = actors.setdefault(
            login,
            {
                "login": login,
                "author_associations": set(),
                "roles": set(),
                "comment_count": 0,
            },
        )
        if association:
            current["author_associations"].add(association)
        current["roles"].add(role)

    issue_author = nested_login(raw_issue.get("author"))
    register(issue_author, raw_issue.get("authorAssociation"), "issue_author")

    for comment in comments:
        login = nested_login(comment.get("author"))
        register(login, comment.get("authorAssociation"), "commenter")
        if login and login in actors:
            actors[login]["comment_count"] += 1

    for item in timeline_items:
        actor = nested_login(item.get("actor"))
        register(actor, None, item.get("__typename", "timeline_actor"))

    rows = []
    for actor in actors.values():
        rows.append(
            {
                "login": actor["login"],
                "author_associations": sorted(actor["author_associations"]),
                "roles": sorted(actor["roles"]),
                "comment_count": actor["comment_count"],
            }
        )
    rows.sort(key=lambda item: item["login"])
    return rows


def normalize_timeline_items(timeline_items: Sequence[JSONDict]) -> List[JSONDict]:
    normalized = []
    for item in timeline_items:
        event = {
            "event_type": item.get("__typename"),
            "created_at": item.get("createdAt"),
            "actor_login": nested_login(item.get("actor")),
        }

        if item.get("__typename") in {"LabeledEvent", "UnlabeledEvent"}:
            label = item.get("label") or {}
            event["label_name"] = label.get("name")

        if item.get("__typename") == "ClosedEvent":
            event["state_reason"] = item.get("stateReason")

        if item.get("__typename") == "CrossReferencedEvent":
            source = item.get("source") or {}
            event["will_close_target"] = bool(item.get("willCloseTarget"))
            event["source"] = {
                "type": source.get("__typename"),
                "number": source.get("number"),
                "url": source.get("url"),
                "title": source.get("title"),
                "state": source.get("state"),
                "merged": source.get("merged"),
                "merged_at": source.get("mergedAt"),
            }

        normalized.append(event)

    return normalized


def extract_linked_pull_requests(timeline_items: Sequence[JSONDict]) -> List[JSONDict]:
    linked = []
    for item in timeline_items:
        if item.get("__typename") != "CrossReferencedEvent":
            continue
        source = item.get("source") or {}
        if source.get("__typename") != "PullRequest":
            continue
        linked.append(
            {
                "number": source.get("number"),
                "url": source.get("url"),
                "title": source.get("title"),
                "state": source.get("state"),
                "merged": source.get("merged"),
                "merged_at": source.get("mergedAt"),
                "will_close_target": bool(item.get("willCloseTarget")),
                "referenced_at": item.get("createdAt"),
            }
        )
    linked.sort(key=lambda item: item.get("referenced_at") or "")
    return linked


def summarize_dataset(records: Iterable[JSONDict]) -> JSONDict:
    records = list(records)
    issue_states = Counter(
        record["taxonomic_metadata"]["issue_state"]
        for record in records
        if record.get("taxonomic_metadata")
    )
    return {
        "record_count": len(records),
        "records_with_comments": sum(1 for record in records if record["deliberation_thread"]),
        "records_with_timeline_events": sum(1 for record in records if record["timeline_events"]),
        "linked_pull_request_count": sum(
            len(record["resolution_artifacts"]["linked_pull_requests"])
            for record in records
        ),
        "merged_linked_pull_request_count": sum(
            1
            for record in records
            for pull_request in record["resolution_artifacts"]["linked_pull_requests"]
            if pull_request.get("merged")
        ),
        "issue_state_distribution": dict(sorted(issue_states.items())),
    }


def nested_login(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    login = value.get("login")
    if isinstance(login, str) and login.strip():
        return login
    return None

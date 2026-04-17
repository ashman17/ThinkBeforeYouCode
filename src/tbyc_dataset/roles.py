from __future__ import annotations

from typing import Optional


def display_role_from_association(author_association: Optional[str]) -> str:
    if author_association == "OWNER":
        return "owner"
    if author_association == "MEMBER":
        return "member"
    if author_association == "COLLABORATOR":
        return "collaborator"
    if author_association == "CONTRIBUTOR":
        return "contributor"
    if author_association == "FIRST_TIME_CONTRIBUTOR":
        return "first-time contributor"
    if author_association == "FIRST_TIMER":
        return "first-timer"
    return "external"

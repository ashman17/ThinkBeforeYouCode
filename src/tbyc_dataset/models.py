from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

JSONDict = Dict[str, Any]
JSONList = List[JSONDict]


@dataclass(frozen=True)
class RepositoryRef:
    owner: str
    name: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def fs_slug(self) -> str:
        return f"{self.owner}__{self.name}"

    @classmethod
    def parse(cls, value: str) -> "RepositoryRef":
        if "/" not in value:
            raise ValueError("Repository must be formatted as owner/name.")
        owner, name = value.split("/", 1)
        owner = owner.strip()
        name = name.strip()
        if not owner or not name:
            raise ValueError("Repository must be formatted as owner/name.")
        return cls(owner=owner, name=name)

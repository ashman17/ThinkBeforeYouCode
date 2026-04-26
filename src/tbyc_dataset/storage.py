from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping

from tbyc_dataset.models import JSONDict, RepositoryRef


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def raw_repo_dir(output_root: Path, repo: RepositoryRef) -> Path:
    return output_root / "raw" / repo.fs_slug


def processed_repo_dir(output_root: Path, repo: RepositoryRef) -> Path:
    return output_root / "processed" / repo.fs_slug


def extraction_repo_dir(output_root: Path, repo: RepositoryRef) -> Path:
    return output_root / "extractions" / repo.fs_slug


def extraction_regex_repo_dir(output_root: Path, repo: RepositoryRef) -> Path:
    return output_root / "extractions_regex" / repo.fs_slug


def issue_snapshot_path(output_root: Path, repo: RepositoryRef, number: int) -> Path:
    return raw_repo_dir(output_root, repo) / "issues" / f"issue_{number}.json"


def curated_dataset_path(output_root: Path, repo: RepositoryRef) -> Path:
    return processed_repo_dir(output_root, repo) / "curated.jsonl"


def dataset_summary_path(output_root: Path, repo: RepositoryRef) -> Path:
    return processed_repo_dir(output_root, repo) / "summary.json"


def discussion_entities_records_path(output_root: Path, repo: RepositoryRef) -> Path:
    return extraction_repo_dir(output_root, repo) / "discussion_entities.jsonl"


def discussion_entities_summary_path(output_root: Path, repo: RepositoryRef) -> Path:
    return extraction_repo_dir(output_root, repo) / "summary.json"


def annotated_discussion_entities_path(output_root: Path, repo: RepositoryRef) -> Path:
    return extraction_repo_dir(output_root, repo) / "langextract_annotated.jsonl"


def viewer_index_path(output_root: Path) -> Path:
    return output_root / "viewer" / "index.html"


def discussion_artifacts_records_path(output_root: Path, repo: RepositoryRef) -> Path:
    return extraction_repo_dir(output_root, repo) / "discussion_artifacts.jsonl"


def discussion_artifacts_summary_path(output_root: Path, repo: RepositoryRef) -> Path:
    return extraction_repo_dir(output_root, repo) / "discussion_artifacts_summary.json"


def discussion_artifacts_flat_path(output_root: Path, repo: RepositoryRef) -> Path:
    return extraction_repo_dir(output_root, repo) / "discussion_artifacts_flat.jsonl"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[JSONDict]:
    rows: List[JSONDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

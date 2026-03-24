from __future__ import annotations

from typing import List, Optional, Sequence

from tbyc_dataset.config import GitHubSettings, PipelineSettings
from tbyc_dataset.dataset.github import GitHubGraphQLClient
from tbyc_dataset.dataset.normalize import normalize_issue, summarize_dataset
from tbyc_dataset.models import JSONDict, RepositoryRef
from tbyc_dataset.storage import (
    curated_dataset_path,
    dataset_summary_path,
    issue_snapshot_path,
    raw_repo_dir,
    read_json,
    write_json,
    write_jsonl,
)


def fetch_repository(
    repo: RepositoryRef,
    github_settings: GitHubSettings,
    pipeline_settings: PipelineSettings,
    states: Sequence[str],
    max_issues: Optional[int] = None,
) -> JSONDict:
    client = GitHubGraphQLClient(github_settings)
    issue_numbers = client.list_issue_numbers(
        repo=repo,
        states=list(states),
        max_issues=max_issues,
        min_comments=pipeline_settings.min_comments,
        max_comments=pipeline_settings.max_comments,
    )

    for number in issue_numbers:
        issue = client.fetch_issue(repo, number)
        snapshot_path = issue_snapshot_path(pipeline_settings.output_root, repo, number)
        write_json(snapshot_path, issue)

    manifest = {
        "repository": repo.slug,
        "fetched_issue_count": len(issue_numbers),
        "issue_numbers": issue_numbers,
        "states": list(states),
        "min_comments": pipeline_settings.min_comments,
        "max_comments": pipeline_settings.max_comments,
        "raw_dir": str(raw_repo_dir(pipeline_settings.output_root, repo)),
    }
    write_json(raw_repo_dir(pipeline_settings.output_root, repo) / "manifest.json", manifest)
    return manifest


def curate_repository(
    repo: RepositoryRef,
    pipeline_settings: PipelineSettings,
) -> JSONDict:
    raw_issues_dir = raw_repo_dir(pipeline_settings.output_root, repo) / "issues"
    if not raw_issues_dir.exists():
        raise FileNotFoundError(
            f"No raw issue snapshots found for {repo.slug} in {raw_issues_dir}."
        )

    records: List[JSONDict] = []
    for path in sorted(raw_issues_dir.glob("issue_*.json")):
        raw_issue = read_json(path)
        comment_count = len(raw_issue.get("comments", []))
        if comment_count < pipeline_settings.min_comments:
            continue
        if (
            pipeline_settings.max_comments is not None
            and comment_count > pipeline_settings.max_comments
        ):
            continue
        records.append(normalize_issue(raw_issue, repo))

    dataset_path = curated_dataset_path(pipeline_settings.output_root, repo)
    write_jsonl(dataset_path, records)

    summary = summarize_dataset(records)
    summary["repository"] = repo.slug
    summary["dataset_path"] = str(dataset_path)
    summary["min_comments"] = pipeline_settings.min_comments
    summary["max_comments"] = pipeline_settings.max_comments
    write_json(dataset_summary_path(pipeline_settings.output_root, repo), summary)
    return summary


def build_dataset(
    repo: RepositoryRef,
    github_settings: GitHubSettings,
    pipeline_settings: PipelineSettings,
    states: Sequence[str],
    max_issues: Optional[int] = None,
) -> JSONDict:
    fetch_manifest = fetch_repository(
        repo=repo,
        github_settings=github_settings,
        pipeline_settings=pipeline_settings,
        states=states,
        max_issues=max_issues,
    )
    summary = curate_repository(repo=repo, pipeline_settings=pipeline_settings)
    return {
        "fetch_manifest": fetch_manifest,
        "curation_summary": summary,
    }

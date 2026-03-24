from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from tbyc_dataset.config import github_settings_from_env, pipeline_settings
from tbyc_dataset.dataset.pipeline import build_dataset, curate_repository, fetch_repository
from tbyc_dataset.extraction.discussion_entities_pipeline import extract_discussion_entities
from tbyc_dataset.models import RepositoryRef


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curate GitHub issue discussions for the Think Before You Code benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("fetch-repo", "curate-repo", "build-dataset", "extract-discussion-entities"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--repo", required=True, help="Repository in owner/name format.")
        subparser.add_argument(
            "--output-root",
            default="data",
            help="Directory where raw and processed artifacts are stored.",
        )
        subparser.add_argument(
            "--min-comments",
            type=int,
            default=0,
            help="Minimum comment count required during curation.",
        )
        subparser.add_argument(
            "--max-comments",
            type=int,
            default=None,
            help="Maximum comment count allowed during curation.",
        )

        if command == "extract-discussion-entities":
            subparser.add_argument(
                "--model-id",
                default="gemma3:4b",
                help="Ollama model identifier for LangExtract.",
            )
            subparser.add_argument(
                "--model-url",
                default="http://localhost:11434",
                help="Ollama server URL.",
            )
            subparser.add_argument(
                "--limit-threads",
                type=int,
                default=None,
                help="Optional cap on the number of discussion threads to extract from.",
            )
            subparser.add_argument(
                "--save-annotated",
                action="store_true",
                help="Also save LangExtract annotated documents JSONL for visualization.",
            )

        if command in {"fetch-repo", "build-dataset"}:
            subparser.add_argument(
                "--max-issues",
                type=int,
                default=None,
                help="Optional cap on the number of issues to fetch.",
            )
            subparser.add_argument(
                "--states",
                nargs="+",
                default=["OPEN", "CLOSED"],
                help="GitHub issue states to fetch.",
            )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo = RepositoryRef.parse(args.repo)
    p_settings = pipeline_settings(args.output_root, args.min_comments, args.max_comments)

    if args.command == "curate-repo":
        result = curate_repository(repo=repo, pipeline_settings=p_settings)
    elif args.command == "extract-discussion-entities":
        result = extract_discussion_entities(
            repo=repo,
            output_root=p_settings.output_root,
            model_id=args.model_id,
            model_url=args.model_url,
            limit_threads=args.limit_threads,
            save_annotated=args.save_annotated,
        )
    elif args.command == "fetch-repo":
        g_settings = github_settings_from_env()
        result = fetch_repository(
            repo=repo,
            github_settings=g_settings,
            pipeline_settings=p_settings,
            states=args.states,
            max_issues=args.max_issues,
        )
    else:
        g_settings = github_settings_from_env()
        result = build_dataset(
            repo=repo,
            github_settings=g_settings,
            pipeline_settings=p_settings,
            states=args.states,
            max_issues=args.max_issues,
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

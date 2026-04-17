from __future__ import annotations

import argparse
import json
import logging
from typing import Optional, Sequence

from tbyc_dataset.config import github_settings_from_env, pipeline_settings
from tbyc_dataset.dataset.pipeline import build_dataset, curate_repository, fetch_repository
from tbyc_dataset.extraction.pipeline import ExtractionSettings, extract_discussion_artifacts
from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.viewer import build_processed_viewer


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curate GitHub issue discussions for the Think Before You Code benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in (
        "fetch-repo",
        "curate-repo",
        "build-dataset",
        "extract-discussion-entities",
        "extract-discussion-artifacts",
    ):
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

        if command in {"extract-discussion-entities", "extract-discussion-artifacts"}:
            subparser.add_argument(
                "--model-id",
                default="qwen2.5:14b",
                help="Ollama model identifier for structured artifact extraction.",
            )
            subparser.add_argument(
                "--model-url",
                default="http://localhost:11434",
                help="Ollama server URL.",
            )
            subparser.add_argument(
                "--num-ctx",
                type=int,
                default=8192,
                help="Ollama context window size (num_ctx).",
            )
            subparser.add_argument(
                "--limit-threads",
                type=int,
                default=None,
                help="Optional cap on the number of discussion threads to extract from.",
            )
            subparser.add_argument(
                "--issue-number",
                type=int,
                default=None,
                help="Optional issue number filter to run extraction for a single issue.",
            )
            subparser.add_argument(
                "--parallel-issues",
                type=int,
                default=1,
                help="Number of issues to process concurrently.",
            )
            subparser.add_argument(
                "--no-skip-existing",
                action="store_true",
                help="Reprocess issues even when issue_<n>.json already exists.",
            )
            subparser.add_argument(
                "--save-annotated",
                action="store_true",
                help="Unused compatibility flag kept for CLI stability.",
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

    viewer_parser = subparsers.add_parser("build-viewer")
    viewer_parser.add_argument(
        "--output-root",
        default="data",
        help="Directory containing processed artifacts and where the viewer will be written.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = build_parser()
    args = parser.parse_args(argv)
    LOGGER.info("command=%s output_root=%s", args.command, getattr(args, "output_root", "-"))
    if args.command == "build-viewer":
        LOGGER.info("building viewer")
        result = build_processed_viewer(output_root=pipeline_settings(args.output_root, 0, None).output_root)
        LOGGER.info("viewer build completed")
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    repo = RepositoryRef.parse(args.repo)
    LOGGER.info("repo=%s", repo.slug)
    p_settings = pipeline_settings(args.output_root, args.min_comments, args.max_comments)

    if args.command == "curate-repo":
        LOGGER.info("stage=curate")
        result = curate_repository(repo=repo, pipeline_settings=p_settings)
    elif args.command == "fetch-repo":
        LOGGER.info("stage=fetch")
        g_settings = github_settings_from_env()
        result = fetch_repository(
            repo=repo,
            github_settings=g_settings,
            pipeline_settings=p_settings,
            states=args.states,
            max_issues=args.max_issues,
        )
    elif args.command in {"extract-discussion-artifacts", "extract-discussion-entities"}:
        LOGGER.info("stage=extract model=%s issue_number=%s", args.model_id, args.issue_number)
        e_settings = ExtractionSettings(
            model_id=args.model_id,
            model_url=args.model_url,
            num_ctx=args.num_ctx,
            limit_threads=args.limit_threads,
            issue_number=args.issue_number,
            parallel_issues=args.parallel_issues,
            skip_existing=not args.no_skip_existing,
        )
        result = extract_discussion_artifacts(
            repo=repo,
            pipeline_settings=p_settings,
            extraction_settings=e_settings,
        )
    else:
        LOGGER.info("stage=build-dataset")
        g_settings = github_settings_from_env()
        result = build_dataset(
            repo=repo,
            github_settings=g_settings,
            pipeline_settings=p_settings,
            states=args.states,
            max_issues=args.max_issues,
        )

    LOGGER.info("command completed")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

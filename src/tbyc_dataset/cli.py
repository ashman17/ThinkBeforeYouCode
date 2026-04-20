from __future__ import annotations

import argparse
import json
import logging
from typing import Optional, Sequence

from tbyc_dataset.config import github_settings_from_env, pipeline_settings
from tbyc_dataset.dataset.pipeline import build_dataset, curate_repository, fetch_repository
from tbyc_dataset.evaluation import (
    CodeRetrievalPipeline,
    DerivedExtractionSettings,
    IssueThoughtPipeline,
    IssueThoughtSettings,
    extract_derived_artifacts_from_responses,
)
from tbyc_dataset.extraction.pipeline import ExtractionSettings, extract_discussion_artifacts
from tbyc_dataset.metrics import (
    compute_metadata_matching_metrics,
    compute_summary_matching_metrics,
    compute_tag_matching_metrics,
    compute_type_matching_metrics,
)
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
        "retrieve-code-chunks",
        "generate-issue-thoughts",
        "extract-derived-artifacts",
        "compute-type-metrics",
        "compute-metadata-metrics",
        "compute-tag-metrics",
        "compute-summary-metrics",
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

        if command == "retrieve-code-chunks":
            subparser.add_argument(
                "--embedding-model",
                default="microsoft/codebert-base",
                help="HuggingFace model for embeddings (CodeBERT).",
            )
            subparser.add_argument(
                "--chunk-size",
                type=int,
                default=50,
                help="Lines per chunk for code chunking.",
            )
            subparser.add_argument(
                "--top-n",
                type=int,
                default=20,
                help="Number of top results to return per issue.",
            )

        if command == "generate-issue-thoughts":
            subparser.add_argument(
                "--model-id",
                default="qwen2.5:14b",
                help=(
                    "Model identifier for issue thought generation. "
                    "Use 'ollama/<model>' (or an Ollama tag like qwen2.5:14b) for Ollama, "
                    "or 'api/<model>' for CMU gateway API calls."
                ),
            )
            subparser.add_argument(
                "--model-url",
                default="http://localhost:11434",
                help="Ollama server URL (only used for Ollama models).",
            )
            subparser.add_argument(
                "--num-ctx",
                type=int,
                default=32768,
                help="Ollama context window size (num_ctx, Ollama models only).",
            )
            subparser.add_argument(
                "--include-context",
                dest="include_context",
                action="store_true",
                default=True,
                help="Include retrieved code chunks in the prompt context.",
            )
            subparser.add_argument(
                "--exclude-context",
                dest="include_context",
                action="store_false",
                help="Exclude retrieved code chunks from prompt context.",
            )
            subparser.add_argument(
                "--max-context-chars",
                type=int,
                default=32768,
                help="Maximum total prompt characters (instructions + issue + context).",
            )
            subparser.add_argument(
                "--max-context-chunks",
                type=int,
                default=10,
                help="Maximum number of retrieved chunks to include.",
            )
            subparser.add_argument(
                "--issue-number",
                type=int,
                default=None,
                help="Optional issue number filter.",
            )
            subparser.add_argument(
                "--no-skip-existing",
                action="store_true",
                help="Regenerate response files even when they already exist.",
            )

        if command == "extract-derived-artifacts":
            subparser.add_argument(
                "--model-id",
                default="qwen2.5:14b",
                help="Model identifier used to run derived artifact extraction.",
            )
            subparser.add_argument(
                "--responses-model-id",
                default=None,
                help=(
                    "Model identifier used to locate response files under data/responses. "
                    "Defaults to --model-id for backward compatibility."
                ),
            )
            subparser.add_argument(
                "--model-url",
                default="http://localhost:11434",
                help="Ollama server URL.",
            )
            subparser.add_argument(
                "--num-ctx",
                type=int,
                default=32768,
                help="Ollama context window size (num_ctx).",
            )
            subparser.add_argument(
                "--issue-number",
                type=int,
                default=None,
                help="Optional issue number filter.",
            )
            subparser.add_argument(
                "--no-skip-existing",
                action="store_true",
                help="Regenerate derived files even when they already exist.",
            )

        if command == "compute-type-metrics":
            subparser.add_argument(
                "--model-id",
                default="qwen2.5:7b-instruct",
                help="Model identifier used to locate derived artifacts and metric outputs.",
            )
            subparser.add_argument(
                "--issue-number",
                type=int,
                default=None,
                help="Optional issue number filter.",
            )

        if command == "compute-metadata-metrics":
            subparser.add_argument(
                "--model-id",
                default="qwen2.5:7b-instruct",
                help="Model identifier used to locate derived artifacts and metric outputs.",
            )
            subparser.add_argument(
                "--issue-number",
                type=int,
                default=None,
                help="Optional issue number filter.",
            )
            subparser.add_argument(
                "--similarity-threshold",
                type=float,
                default=0.82,
                help="Minimum phrase similarity used to count a metadata value as a match.",
            )
            subparser.add_argument(
                "--similarity-metric",
                choices=[
                    "all",
                    "max_all",
                    "token_f1",
                    "sequence_ratio",
                    "token_jaccard",
                    "char_3gram_jaccard",
                    "token_containment",
                ],
                default="max_all",
                help="Similarity backend used for metadata phrase matching.",
            )

        if command == "compute-tag-metrics":
            subparser.add_argument(
                "--model-id",
                default="qwen2.5:7b-instruct",
                help="Model identifier used to locate derived artifacts and metric outputs.",
            )
            subparser.add_argument(
                "--issue-number",
                type=int,
                default=None,
                help="Optional issue number filter.",
            )

        if command == "compute-summary-metrics":
            subparser.add_argument(
                "--model-id",
                default="qwen2.5:7b-instruct",
                help="Model identifier used to locate derived artifacts and metric outputs.",
            )
            subparser.add_argument(
                "--issue-number",
                type=int,
                default=None,
                help="Optional issue number filter.",
            )
            subparser.add_argument(
                "--codebert-model",
                default="microsoft/codebert-base",
                help="Encoder model used for the CodeBERT cosine baseline.",
            )
            subparser.add_argument(
                "--bertscore-model",
                default="microsoft/codebert-base",
                help="Encoder model used for BERTScore token-level contextual matching.",
            )
            subparser.add_argument(
                "--bleurt-model",
                default="Elron/bleurt-base-512",
                help="BLEURT model checkpoint for summary quality scoring.",
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
    elif args.command == "retrieve-code-chunks":
        LOGGER.info("stage=retrieve-code-chunks")
        pipeline = CodeRetrievalPipeline(
            cache_dir=str(args.output_root),
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            top_n=args.top_n,
        )
        result = pipeline.run(
            owner=repo.owner,
            repo=repo.name,
            output_dir=str(p_settings.output_root / "evaluation")
        )
    elif args.command == "generate-issue-thoughts":
        LOGGER.info("stage=generate-issue-thoughts model=%s", args.model_id)
        thought_settings = IssueThoughtSettings(
            model_id=args.model_id,
            model_url=args.model_url,
            include_context=args.include_context,
            max_context_chars=args.max_context_chars,
            max_context_chunks=args.max_context_chunks,
            num_ctx=args.num_ctx,
            issue_number=args.issue_number,
            skip_existing=not args.no_skip_existing,
        )
        pipeline = IssueThoughtPipeline(
            output_root=str(p_settings.output_root),
            settings=thought_settings,
        )
        result = pipeline.run(owner=repo.owner, repo=repo.name)
    elif args.command == "extract-derived-artifacts":
        LOGGER.info(
            "stage=extract-derived-artifacts extraction_model=%s responses_model=%s",
            args.model_id,
            args.responses_model_id or args.model_id,
        )
        derived_settings = DerivedExtractionSettings(
            model_id=args.model_id,
            responses_model_id=args.responses_model_id,
            model_url=args.model_url,
            num_ctx=args.num_ctx,
            issue_number=args.issue_number,
            skip_existing=not args.no_skip_existing,
        )
        result = extract_derived_artifacts_from_responses(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            settings=derived_settings,
        )
    elif args.command == "compute-type-metrics":
        LOGGER.info("stage=compute-type-metrics model=%s", args.model_id)
        result = compute_type_matching_metrics(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            model_id=args.model_id,
            issue_number=args.issue_number,
        )
    elif args.command == "compute-metadata-metrics":
        LOGGER.info("stage=compute-metadata-metrics model=%s", args.model_id)
        result = compute_metadata_matching_metrics(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            model_id=args.model_id,
            issue_number=args.issue_number,
            similarity_threshold=args.similarity_threshold,
            similarity_metric=args.similarity_metric,
        )
    elif args.command == "compute-tag-metrics":
        LOGGER.info("stage=compute-tag-metrics model=%s", args.model_id)
        result = compute_tag_matching_metrics(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            model_id=args.model_id,
            issue_number=args.issue_number,
        )
    elif args.command == "compute-summary-metrics":
        LOGGER.info("stage=compute-summary-metrics model=%s", args.model_id)
        result = compute_summary_matching_metrics(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            model_id=args.model_id,
            issue_number=args.issue_number,
            codebert_model=args.codebert_model,
            bertscore_model=args.bertscore_model,
            bleurt_model=args.bleurt_model,
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
    if args.command == "retrieve-code-chunks":
        print(json.dumps({"manifest": result.get("manifest")}, indent=2, sort_keys=True))
    elif args.command == "generate-issue-thoughts":
        print(
            json.dumps(
                {
                    "repository": result.get("repository"),
                    "response_dir": result.get("response_dir"),
                    "issue_count": result.get("issue_count"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "extract-derived-artifacts":
        print(
            json.dumps(
                {
                    "repository": result.get("repository"),
                    "issue_count": result.get("issue_count"),
                    "total_comment_count": result.get("total_comment_count"),
                    "failed_comment_count": result.get("failed_comment_count"),
                    "total_artifact_count": result.get("total_artifact_count"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "compute-type-metrics":
        print(
            json.dumps(
                {
                    "repository": result.get("repository"),
                    "model_id": result.get("model_id"),
                    "issue_count": result.get("issue_count"),
                    "metric": result.get("metric"),
                    "overall": result.get("overall"),
                    "macro_average": result.get("macro_average"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "compute-metadata-metrics":
        print(
            json.dumps(
                (
                    {
                        "repository": result.get("repository"),
                        "model_id": result.get("model_id"),
                        "issue_count": result.get("issue_count"),
                        "metric": result.get("metric"),
                        "similarity": result.get("similarity"),
                        "summary_by_metric": {
                            key: {
                                "overall": value.get("overall", {}),
                                "macro_average": value.get("macro_average", {}),
                            }
                            for key, value in result.get("summary_by_metric", {}).items()
                        },
                    }
                    if result.get("metric") == "metadata_phrase_matching_all"
                    else {
                        "repository": result.get("repository"),
                        "model_id": result.get("model_id"),
                        "issue_count": result.get("issue_count"),
                        "metric": result.get("metric"),
                        "similarity": result.get("similarity"),
                        "overall": result.get("overall", {}).get("pooled"),
                        "macro_average": result.get("macro_average"),
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "compute-tag-metrics":
        print(
            json.dumps(
                {
                    "repository": result.get("repository"),
                    "model_id": result.get("model_id"),
                    "issue_count": result.get("issue_count"),
                    "metric": result.get("metric"),
                    "overall": result.get("overall"),
                    "macro_average": result.get("macro_average"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "compute-summary-metrics":
        print(
            json.dumps(
                {
                    "repository": result.get("repository"),
                    "model_id": result.get("model_id"),
                    "issue_count": result.get("issue_count"),
                    "metric": result.get("metric"),
                    "models": result.get("models"),
                    "overall": result.get("overall"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

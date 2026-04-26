from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
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
from tbyc_dataset.extraction.regex_pipeline import RegexExtractionSettings, extract_discussion_artifacts_regex
from tbyc_dataset.metrics import (
    compute_extraction_comparison_metrics,
    compute_metadata_matching_metrics,
    compute_summary_matching_metrics,
    compute_tag_matching_metrics,
    compute_type_matching_metrics,
    generate_extraction_comparison_visualizations,
    generate_metrics_visualizations,
)
from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import read_json
from tbyc_dataset.viewer import build_processed_viewer


LOGGER = logging.getLogger(__name__)


FEW_SHOT_GENERATION_COMMANDS = {"generate-issue-thoughts-few-shot", "generate-issue-thoughts-regex"}
NO_CONTEXT_GENERATION_COMMANDS = {"generate-issue-thoughts-no-context"}
FEW_SHOT_DERIVED_COMMANDS = {"extract-derived-artifacts-few-shot", "extract-derived-artifacts-regex"}
NO_CONTEXT_DERIVED_COMMANDS = {"extract-derived-artifacts-no-context"}
FEW_SHOT_TYPE_METRIC_COMMANDS = {"compute-type-metrics-few-shot", "compute-type-metrics-regex"}
FEW_SHOT_METADATA_METRIC_COMMANDS = {"compute-metadata-metrics-few-shot", "compute-metadata-metrics-regex"}
FEW_SHOT_TAG_METRIC_COMMANDS = {"compute-tag-metrics-few-shot", "compute-tag-metrics-regex"}
FEW_SHOT_SUMMARY_METRIC_COMMANDS = {"compute-summary-metrics-few-shot", "compute-summary-metrics-regex"}
FEW_SHOT_ALL_METRIC_COMMANDS = {"compute-all-metrics-few-shot", "compute-all-metrics-regex"}
FEW_SHOT_VISUALIZATION_COMMANDS = {"visualize-metrics-few-shot", "visualize-metrics-regex"}


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
        "extract-discussion-artifacts-regex",
        "retrieve-code-chunks",
        "generate-issue-thoughts",
        "generate-issue-thoughts-no-context",
        "generate-issue-thoughts-few-shot",
        "generate-issue-thoughts-regex",
        "extract-derived-artifacts",
        "extract-derived-artifacts-no-context",
        "extract-derived-artifacts-few-shot",
        "extract-derived-artifacts-regex",
        "compute-type-metrics",
        "compute-type-metrics-few-shot",
        "compute-type-metrics-regex",
        "compute-metadata-metrics",
        "compute-metadata-metrics-few-shot",
        "compute-metadata-metrics-regex",
        "compute-tag-metrics",
        "compute-tag-metrics-few-shot",
        "compute-tag-metrics-regex",
        "compute-summary-metrics",
        "compute-summary-metrics-few-shot",
        "compute-summary-metrics-regex",
        "compute-extraction-comparison",
        "compute-all-metrics",
        "compute-all-metrics-few-shot",
        "compute-all-metrics-regex",
        "compute-all-metrics-one-shot",
        "build-leaderboard",
        "visualize-metrics",
        "visualize-metrics-few-shot",
        "visualize-metrics-regex",
        "visualize-extraction-comparison",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--repo",
            required=command
            not in {
                "compute-all-metrics",
                "compute-all-metrics-few-shot",
                "compute-all-metrics-regex",
                "compute-all-metrics-one-shot",
                "build-leaderboard",
                "visualize-metrics",
                "visualize-metrics-few-shot",
                "visualize-metrics-regex",
                "visualize-extraction-comparison",
            },
            default=None,
            help=(
                "Repository in owner/name format. For compute-all-metrics variants, this is optional "
                "and acts as a filter."
            ),
        )
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

        if command == "extract-discussion-artifacts-regex":
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

        if command in {"generate-issue-thoughts"} | FEW_SHOT_GENERATION_COMMANDS | NO_CONTEXT_GENERATION_COMMANDS:
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
                "--limit-issues",
                type=int,
                default=None,
                help="Optional cap to process only the first N issues after filtering.",
            )
            subparser.add_argument(
                "--no-skip-existing",
                action="store_true",
                help="Regenerate response files even when they already exist.",
            )
            if command in FEW_SHOT_GENERATION_COMMANDS:
                subparser.add_argument(
                    "--few-shot-example-count",
                    type=int,
                    default=3,
                    help="Number of in-repo extraction examples to include in the prompt.",
                )
                subparser.add_argument(
                    "--few-shot-artifacts-per-example",
                    type=int,
                    default=5,
                    help="Maximum artifact-derived lines to include from each few-shot example issue.",
                )

        if command in {"extract-derived-artifacts"} | FEW_SHOT_DERIVED_COMMANDS | NO_CONTEXT_DERIVED_COMMANDS:
            subparser.add_argument(
                "--model-id",
                default="qwen2.5:14b",
                help="Model identifier used to run derived artifact extraction.",
            )
            subparser.add_argument(
                "--responses-model-id",
                default=None,
                help=(
                    "Model identifier used to locate response files under the selected responses directory. "
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
                "--limit-issues",
                type=int,
                default=None,
                help="Optional cap to process only the first N issues after filtering.",
            )
            subparser.add_argument(
                "--no-skip-existing",
                action="store_true",
                help="Regenerate derived files even when they already exist.",
            )

        if command in {"compute-type-metrics"} | FEW_SHOT_TYPE_METRIC_COMMANDS:
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

        if command in {"compute-metadata-metrics"} | FEW_SHOT_METADATA_METRIC_COMMANDS:
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

        if command in {"compute-tag-metrics"} | FEW_SHOT_TAG_METRIC_COMMANDS:
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

        if command in {"compute-summary-metrics"} | FEW_SHOT_SUMMARY_METRIC_COMMANDS:
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
            subparser.add_argument(
                "--bleurt-postprocess",
                choices=["sigmoid", "clip", "raw"],
                default="sigmoid",
                help="BLEURT post-processing mode (default: sigmoid).",
            )
            subparser.add_argument(
                "--bleurt-clip-min",
                type=float,
                default=0.0,
                help="Minimum BLEURT score when --bleurt-postprocess=clip (default: 0.0).",
            )
            subparser.add_argument(
                "--bleurt-sigmoid-temperature",
                type=float,
                default=2.0,
                help="Temperature for --bleurt-postprocess=sigmoid (higher = more relaxed).",
            )
            subparser.add_argument(
                "--bleurt-sigmoid-bias",
                type=float,
                default=0.0,
                help="Bias shift for --bleurt-postprocess=sigmoid.",
            )

        if command == "compute-extraction-comparison":
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
                help="Minimum phrase similarity used for metadata comparison.",
            )
            subparser.add_argument(
                "--similarity-metric",
                choices=[
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
            subparser.add_argument(
                "--bleurt-postprocess",
                choices=["sigmoid", "clip", "raw"],
                default="sigmoid",
                help="BLEURT post-processing mode (default: sigmoid).",
            )
            subparser.add_argument(
                "--bleurt-clip-min",
                type=float,
                default=0.0,
                help="Minimum BLEURT score when --bleurt-postprocess=clip (default: 0.0).",
            )
            subparser.add_argument(
                "--bleurt-sigmoid-temperature",
                type=float,
                default=2.0,
                help="Temperature for --bleurt-postprocess=sigmoid (higher = more relaxed).",
            )
            subparser.add_argument(
                "--bleurt-sigmoid-bias",
                type=float,
                default=0.0,
                help="Bias shift for --bleurt-postprocess=sigmoid.",
            )

        if command in {"compute-all-metrics", "compute-all-metrics-one-shot"} | FEW_SHOT_ALL_METRIC_COMMANDS:
            subparser.add_argument(
                "--model-id",
                default=None,
                help=(
                    "Optional model filter. If omitted, run all discovered models "
                    "under the selected derived directory."
                ),
            )
            if command in {"compute-all-metrics"} | FEW_SHOT_ALL_METRIC_COMMANDS:
                subparser.add_argument(
                    "--issue-number",
                    type=int,
                    default=None,
                    help="Optional issue number filter forwarded to all metric computations.",
                )
            subparser.add_argument(
                "--similarity-threshold",
                type=float,
                default=0.82,
                help="Metadata metric similarity threshold.",
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
                help="Metadata metric similarity backend.",
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
            subparser.add_argument(
                "--bleurt-postprocess",
                choices=["sigmoid", "clip", "raw"],
                default="sigmoid",
                help="BLEURT post-processing mode (default: sigmoid).",
            )
            subparser.add_argument(
                "--bleurt-clip-min",
                type=float,
                default=0.0,
                help="Minimum BLEURT score when --bleurt-postprocess=clip (default: 0.0).",
            )
            subparser.add_argument(
                "--bleurt-sigmoid-temperature",
                type=float,
                default=2.0,
                help="Temperature for --bleurt-postprocess=sigmoid (higher = more relaxed).",
            )
            subparser.add_argument(
                "--bleurt-sigmoid-bias",
                type=float,
                default=0.0,
                help="Bias shift for --bleurt-postprocess=sigmoid.",
            )
            if command == "compute-all-metrics-one-shot":
                subparser.add_argument(
                    "--rrf-k",
                    type=int,
                    default=60,
                    help="Reciprocal rank fusion constant k for leaderboard generation (default: 60).",
                )
                subparser.add_argument(
                    "--points-max",
                    type=int,
                    default=100,
                    help="Maximum points assigned to the theoretical max-possible score (default: 100).",
                )
                subparser.add_argument(
                    "--points-step",
                    type=int,
                    default=1,
                    help="Quantization step for leaderboard points (default: 1).",
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

        if command == "build-leaderboard":
            subparser.add_argument(
                "--model-id",
                default=None,
                help="Optional model filter for leaderboard generation.",
            )
            subparser.add_argument(
                "--rrf-k",
                type=int,
                default=60,
                help="Reciprocal rank fusion constant k (default: 60).",
            )
            subparser.add_argument(
                "--points-max",
                type=int,
                default=100,
                help="Maximum points assigned to the theoretical max-possible score (default: 100).",
            )
            subparser.add_argument(
                "--points-step",
                type=int,
                default=1,
                help="Quantization step for leaderboard points (default: 1).",
            )

        if command in {"visualize-metrics"} | FEW_SHOT_VISUALIZATION_COMMANDS:
            subparser.add_argument(
                "--model-id",
                default=None,
                help="Optional model filter for graph generation.",
            )
            subparser.add_argument(
                "--graphs-root",
                default=None,
                help="Optional output directory for graphs (default: <output-root>/graphs).",
            )
            subparser.add_argument(
                "--points-max",
                type=int,
                default=100,
                help="Maximum points for the points graph (default: 100).",
            )
            subparser.add_argument(
                "--points-step",
                type=int,
                default=1,
                help="Quantization step for points graph (default: 1).",
            )

        if command == "visualize-extraction-comparison":
            subparser.add_argument(
                "--graphs-root",
                default=None,
                help="Optional output directory for graphs (default: <output-root>/graphs_regex_comparison).",
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

    if args.command == "compute-all-metrics":
        LOGGER.info("stage=compute-all-metrics model_filter=%s repo_filter=%s", args.model_id, args.repo)
        p_settings = pipeline_settings(args.output_root, args.min_comments, args.max_comments)
        result = _compute_all_metrics(
            output_root=p_settings.output_root,
            model_id_filter=args.model_id,
            repo_filter=args.repo,
            issue_number=args.issue_number,
            similarity_threshold=args.similarity_threshold,
            similarity_metric=args.similarity_metric,
            codebert_model=args.codebert_model,
            bertscore_model=args.bertscore_model,
            bleurt_model=args.bleurt_model,
            bleurt_postprocess=args.bleurt_postprocess,
            bleurt_clip_min=args.bleurt_clip_min,
            bleurt_sigmoid_temperature=args.bleurt_sigmoid_temperature,
            bleurt_sigmoid_bias=args.bleurt_sigmoid_bias,
        )
        print(
            json.dumps(
                {
                    "model_filter": result.get("model_filter"),
                    "repo_filter": result.get("repo_filter"),
                    "target_count": result.get("target_count"),
                    "metric_run_count": result.get("metric_run_count"),
                    "succeeded_metric_count": result.get("succeeded_metric_count"),
                    "failed_metric_count": result.get("failed_metric_count"),
                    "targets": result.get("targets"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command in FEW_SHOT_ALL_METRIC_COMMANDS:
        LOGGER.info("stage=compute-all-metrics-few-shot model_filter=%s repo_filter=%s", args.model_id, args.repo)
        p_settings = pipeline_settings(args.output_root, args.min_comments, args.max_comments)
        result = _compute_all_metrics(
            output_root=p_settings.output_root,
            model_id_filter=args.model_id,
            repo_filter=args.repo,
            issue_number=args.issue_number,
            similarity_threshold=args.similarity_threshold,
            similarity_metric=args.similarity_metric,
            codebert_model=args.codebert_model,
            bertscore_model=args.bertscore_model,
            bleurt_model=args.bleurt_model,
            bleurt_postprocess=args.bleurt_postprocess,
            bleurt_clip_min=args.bleurt_clip_min,
            bleurt_sigmoid_temperature=args.bleurt_sigmoid_temperature,
            bleurt_sigmoid_bias=args.bleurt_sigmoid_bias,
            derived_root_dirname="derived_few-shot",
            metrics_root_dirname="metrics_few-shot",
        )
        print(
            json.dumps(
                {
                    "model_filter": result.get("model_filter"),
                    "repo_filter": result.get("repo_filter"),
                    "target_count": result.get("target_count"),
                    "metric_run_count": result.get("metric_run_count"),
                    "succeeded_metric_count": result.get("succeeded_metric_count"),
                    "failed_metric_count": result.get("failed_metric_count"),
                    "targets": result.get("targets"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "compute-all-metrics-one-shot":
        LOGGER.info("stage=compute-all-metrics-one-shot")
        p_settings = pipeline_settings(args.output_root, args.min_comments, args.max_comments)
        compute_result = _compute_all_metrics(
            output_root=p_settings.output_root,
            model_id_filter=None,
            repo_filter=None,
            issue_number=None,
            similarity_threshold=args.similarity_threshold,
            similarity_metric=args.similarity_metric,
            codebert_model=args.codebert_model,
            bertscore_model=args.bertscore_model,
            bleurt_model=args.bleurt_model,
            bleurt_postprocess=args.bleurt_postprocess,
            bleurt_clip_min=args.bleurt_clip_min,
            bleurt_sigmoid_temperature=args.bleurt_sigmoid_temperature,
            bleurt_sigmoid_bias=args.bleurt_sigmoid_bias,
        )
        leaderboard_result = _build_rank_fusion_leaderboard(
            output_root=p_settings.output_root,
            repo_filter=None,
            model_id_filter=None,
            rrf_k=args.rrf_k,
            points_max=args.points_max,
            points_step=args.points_step,
        )
        metrics_dir = p_settings.output_root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        leaderboard_path = metrics_dir / "leaderboard_rank_fusion.json"
        leaderboard_path.write_text(json.dumps(leaderboard_result, indent=2, sort_keys=True), encoding="utf-8")
        print(
            json.dumps(
                {
                    "metric": "all_metrics_one_shot",
                    "scope": "all_repositories_all_models_all_issues",
                    "metric_runs": {
                        "target_count": compute_result.get("target_count"),
                        "metric_run_count": compute_result.get("metric_run_count"),
                        "succeeded_metric_count": compute_result.get("succeeded_metric_count"),
                        "failed_metric_count": compute_result.get("failed_metric_count"),
                    },
                    "leaderboard": {
                        "repo_count": leaderboard_result.get("repo_count"),
                        "model_count": leaderboard_result.get("model_count"),
                        "saved_to": str(leaderboard_path),
                    },
                    "targets": compute_result.get("targets"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "build-leaderboard":
        p_settings = pipeline_settings(args.output_root, args.min_comments, args.max_comments)
        result = _build_rank_fusion_leaderboard(
            output_root=p_settings.output_root,
            repo_filter=args.repo,
            model_id_filter=args.model_id,
            rrf_k=args.rrf_k,
            points_max=args.points_max,
            points_step=args.points_step,
        )
        metrics_dir = p_settings.output_root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if args.repo:
            repo_slug = args.repo.strip().replace("/", "__")
            leaderboard_path = metrics_dir / f"leaderboard_rank_fusion_{repo_slug}.json"
        else:
            leaderboard_path = metrics_dir / "leaderboard_rank_fusion.json"
        leaderboard_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        result["saved_to"] = str(leaderboard_path)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "visualize-metrics":
        p_settings = pipeline_settings(args.output_root, args.min_comments, args.max_comments)
        result = generate_metrics_visualizations(
            output_root=str(p_settings.output_root),
            graphs_root=args.graphs_root,
            repo=args.repo,
            model_id=args.model_id,
            points_max=args.points_max,
            points_step=args.points_step,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command in FEW_SHOT_VISUALIZATION_COMMANDS:
        p_settings = pipeline_settings(args.output_root, args.min_comments, args.max_comments)
        result = generate_metrics_visualizations(
            output_root=str(p_settings.output_root),
            graphs_root=args.graphs_root,
            repo=args.repo,
            model_id=args.model_id,
            metrics_root_dirname="metrics_few-shot",
            graphs_root_dirname="graphs_few-shot",
            points_max=args.points_max,
            points_step=args.points_step,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "visualize-extraction-comparison":
        p_settings = pipeline_settings(args.output_root, args.min_comments, args.max_comments)
        result = generate_extraction_comparison_visualizations(
            output_root=str(p_settings.output_root),
            graphs_root=args.graphs_root,
            repo=args.repo,
        )
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
    elif args.command == "generate-issue-thoughts" or args.command in FEW_SHOT_GENERATION_COMMANDS or args.command in NO_CONTEXT_GENERATION_COMMANDS:
        LOGGER.info("stage=generate-issue-thoughts model=%s", args.model_id)
        is_few_shot = args.command in FEW_SHOT_GENERATION_COMMANDS
        is_no_context = args.command in NO_CONTEXT_GENERATION_COMMANDS
        thought_settings = IssueThoughtSettings(
            model_id=args.model_id,
            model_url=args.model_url,
            include_context=False if is_no_context else args.include_context,
            max_context_chars=args.max_context_chars,
            max_context_chunks=args.max_context_chunks,
            num_ctx=args.num_ctx,
            response_root_dirname=(
                "responses_few-shot"
                if is_few_shot
                else ("responses_no-context" if is_no_context else "responses")
            ),
            few_shot_from_extractions=is_few_shot,
            few_shot_example_count=getattr(args, "few_shot_example_count", 0),
            few_shot_artifacts_per_example=getattr(args, "few_shot_artifacts_per_example", 0),
            limit_issues=args.limit_issues,
            issue_number=args.issue_number,
            skip_existing=not args.no_skip_existing,
        )
        pipeline = IssueThoughtPipeline(
            output_root=str(p_settings.output_root),
            settings=thought_settings,
        )
        result = pipeline.run(owner=repo.owner, repo=repo.name)
    elif args.command == "extract-derived-artifacts" or args.command in FEW_SHOT_DERIVED_COMMANDS or args.command in NO_CONTEXT_DERIVED_COMMANDS:
        LOGGER.info(
            "stage=extract-derived-artifacts extraction_model=%s responses_model=%s",
            args.model_id,
            args.responses_model_id or args.model_id,
        )
        is_few_shot = args.command in FEW_SHOT_DERIVED_COMMANDS
        is_no_context = args.command in NO_CONTEXT_DERIVED_COMMANDS
        derived_settings = DerivedExtractionSettings(
            model_id=args.model_id,
            responses_model_id=args.responses_model_id,
            responses_root_dirname=(
                "responses_few-shot"
                if is_few_shot
                else ("responses_no-context" if is_no_context else "responses")
            ),
            derived_root_dirname=(
                "derived_few-shot"
                if is_few_shot
                else ("derived_no-context" if is_no_context else "derived")
            ),
            model_url=args.model_url,
            num_ctx=args.num_ctx,
            limit_issues=args.limit_issues,
            issue_number=args.issue_number,
            skip_existing=not args.no_skip_existing,
        )
        result = extract_derived_artifacts_from_responses(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            settings=derived_settings,
        )
    elif args.command == "compute-type-metrics" or args.command in FEW_SHOT_TYPE_METRIC_COMMANDS:
        LOGGER.info("stage=compute-type-metrics model=%s", args.model_id)
        is_few_shot = args.command in FEW_SHOT_TYPE_METRIC_COMMANDS
        result = compute_type_matching_metrics(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            model_id=args.model_id,
            issue_number=args.issue_number,
            derived_root_dirname="derived_few-shot" if is_few_shot else "derived",
            metrics_root_dirname="metrics_few-shot" if is_few_shot else "metrics",
        )
    elif args.command == "compute-metadata-metrics" or args.command in FEW_SHOT_METADATA_METRIC_COMMANDS:
        LOGGER.info("stage=compute-metadata-metrics model=%s", args.model_id)
        is_few_shot = args.command in FEW_SHOT_METADATA_METRIC_COMMANDS
        result = compute_metadata_matching_metrics(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            model_id=args.model_id,
            issue_number=args.issue_number,
            similarity_threshold=args.similarity_threshold,
            similarity_metric=args.similarity_metric,
            derived_root_dirname="derived_few-shot" if is_few_shot else "derived",
            metrics_root_dirname="metrics_few-shot" if is_few_shot else "metrics",
        )
    elif args.command == "compute-tag-metrics" or args.command in FEW_SHOT_TAG_METRIC_COMMANDS:
        LOGGER.info("stage=compute-tag-metrics model=%s", args.model_id)
        is_few_shot = args.command in FEW_SHOT_TAG_METRIC_COMMANDS
        result = compute_tag_matching_metrics(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            model_id=args.model_id,
            issue_number=args.issue_number,
            derived_root_dirname="derived_few-shot" if is_few_shot else "derived",
            metrics_root_dirname="metrics_few-shot" if is_few_shot else "metrics",
        )
    elif args.command == "compute-summary-metrics" or args.command in FEW_SHOT_SUMMARY_METRIC_COMMANDS:
        LOGGER.info("stage=compute-summary-metrics model=%s", args.model_id)
        is_few_shot = args.command in FEW_SHOT_SUMMARY_METRIC_COMMANDS
        result = compute_summary_matching_metrics(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            model_id=args.model_id,
            issue_number=args.issue_number,
            derived_root_dirname="derived_few-shot" if is_few_shot else "derived",
            metrics_root_dirname="metrics_few-shot" if is_few_shot else "metrics",
            codebert_model=args.codebert_model,
            bertscore_model=args.bertscore_model,
            bleurt_model=args.bleurt_model,
            bleurt_postprocess=args.bleurt_postprocess,
            bleurt_clip_min=args.bleurt_clip_min,
            bleurt_sigmoid_temperature=args.bleurt_sigmoid_temperature,
            bleurt_sigmoid_bias=args.bleurt_sigmoid_bias,
        )
    elif args.command == "compute-extraction-comparison":
        LOGGER.info("stage=compute-extraction-comparison repo=%s", repo.slug)
        result = compute_extraction_comparison_metrics(
            owner=repo.owner,
            repo=repo.name,
            output_root=str(p_settings.output_root),
            issue_number=args.issue_number,
            similarity_threshold=args.similarity_threshold,
            similarity_metric=args.similarity_metric,
            codebert_model=args.codebert_model,
            bertscore_model=args.bertscore_model,
            bleurt_model=args.bleurt_model,
            bleurt_postprocess=args.bleurt_postprocess,
            bleurt_clip_min=args.bleurt_clip_min,
            bleurt_sigmoid_temperature=args.bleurt_sigmoid_temperature,
            bleurt_sigmoid_bias=args.bleurt_sigmoid_bias,
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
    elif args.command == "extract-discussion-artifacts-regex":
        LOGGER.info("stage=extract-regex issue_number=%s", args.issue_number)
        regex_settings = RegexExtractionSettings(
            limit_threads=args.limit_threads,
            issue_number=args.issue_number,
            parallel_issues=args.parallel_issues,
            skip_existing=not args.no_skip_existing,
        )
        result = extract_discussion_artifacts_regex(
            repo=repo,
            pipeline_settings=p_settings,
            extraction_settings=regex_settings,
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
    elif args.command == "generate-issue-thoughts" or args.command in FEW_SHOT_GENERATION_COMMANDS or args.command in NO_CONTEXT_GENERATION_COMMANDS:
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
    elif args.command == "extract-derived-artifacts" or args.command in FEW_SHOT_DERIVED_COMMANDS or args.command in NO_CONTEXT_DERIVED_COMMANDS:
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
    elif args.command == "compute-type-metrics" or args.command in FEW_SHOT_TYPE_METRIC_COMMANDS:
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
    elif args.command == "compute-metadata-metrics" or args.command in FEW_SHOT_METADATA_METRIC_COMMANDS:
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
    elif args.command == "compute-tag-metrics" or args.command in FEW_SHOT_TAG_METRIC_COMMANDS:
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
    elif args.command == "compute-summary-metrics" or args.command in FEW_SHOT_SUMMARY_METRIC_COMMANDS:
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
    elif args.command == "compute-extraction-comparison":
        print(
            json.dumps(
                {
                    "repository": result.get("repository"),
                    "issue_count": result.get("issue_count"),
                    "reference_side": result.get("reference_side"),
                    "candidate_side": result.get("candidate_side"),
                    "overall": {
                        "type_f1": result.get("macro_average", {}).get("type", {}).get("f1"),
                        "tag_f1": result.get("macro_average", {}).get("tag", {}).get("f1"),
                        "metadata_f1": result.get("macro_average", {}).get("metadata", {}).get("f1"),
                        "metadata_soft_f1": result.get("macro_average", {}).get("metadata", {}).get("soft_f1"),
                        "summary_bertscore_f1": result.get("macro_average", {}).get("summary", {}).get("bertscore", {}).get("f1"),
                        "summary_codebert": result.get("macro_average", {}).get("summary", {}).get("codebert", {}).get("cosine"),
                        "summary_bleurt": result.get("macro_average", {}).get("summary", {}).get("bleurt", {}).get("score"),
                    },
                    "counts": result.get("overall", {}).get("counts"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


def _compute_all_metrics(
    *,
    output_root: Path,
    model_id_filter: Optional[str],
    repo_filter: Optional[str],
    issue_number: Optional[int],
    similarity_threshold: float,
    similarity_metric: str,
    codebert_model: str,
    bertscore_model: str,
    bleurt_model: str,
    bleurt_postprocess: str,
    bleurt_clip_min: float,
    bleurt_sigmoid_temperature: float,
    bleurt_sigmoid_bias: float,
    derived_root_dirname: str = "derived",
    metrics_root_dirname: str = "metrics",
) -> dict:
    targets = _discover_metric_targets(
        output_root=output_root,
        model_id_filter=model_id_filter,
        repo_filter=repo_filter,
        derived_root_dirname=derived_root_dirname,
    )
    if not targets:
        raise FileNotFoundError(
            "No derived metric targets found for the provided filters under "
            f"{output_root / derived_root_dirname}."
        )

    target_reports = []
    succeeded_metric_count = 0
    failed_metric_count = 0
    metric_run_count = len(targets) * 4

    for model_id, repo_ref in targets:
        target_report = {
            "model_id": model_id,
            "repository": repo_ref.slug,
            "metrics": {},
            "errors": [],
        }

        metric_runners = (
            (
                "type",
                lambda: compute_type_matching_metrics(
                    owner=repo_ref.owner,
                    repo=repo_ref.name,
                    output_root=str(output_root),
                    model_id=model_id,
                    issue_number=issue_number,
                    derived_root_dirname=derived_root_dirname,
                    metrics_root_dirname=metrics_root_dirname,
                ),
            ),
            (
                "metadata",
                lambda: compute_metadata_matching_metrics(
                    owner=repo_ref.owner,
                    repo=repo_ref.name,
                    output_root=str(output_root),
                    model_id=model_id,
                    issue_number=issue_number,
                    similarity_threshold=similarity_threshold,
                    similarity_metric=similarity_metric,
                    derived_root_dirname=derived_root_dirname,
                    metrics_root_dirname=metrics_root_dirname,
                ),
            ),
            (
                "tag",
                lambda: compute_tag_matching_metrics(
                    owner=repo_ref.owner,
                    repo=repo_ref.name,
                    output_root=str(output_root),
                    model_id=model_id,
                    issue_number=issue_number,
                    derived_root_dirname=derived_root_dirname,
                    metrics_root_dirname=metrics_root_dirname,
                ),
            ),
            (
                "summary",
                lambda: compute_summary_matching_metrics(
                    owner=repo_ref.owner,
                    repo=repo_ref.name,
                    output_root=str(output_root),
                    model_id=model_id,
                    issue_number=issue_number,
                    derived_root_dirname=derived_root_dirname,
                    metrics_root_dirname=metrics_root_dirname,
                    codebert_model=codebert_model,
                    bertscore_model=bertscore_model,
                    bleurt_model=bleurt_model,
                    bleurt_postprocess=bleurt_postprocess,
                    bleurt_clip_min=bleurt_clip_min,
                    bleurt_sigmoid_temperature=bleurt_sigmoid_temperature,
                    bleurt_sigmoid_bias=bleurt_sigmoid_bias,
                ),
            ),
        )

        for metric_name, runner in metric_runners:
            try:
                result = runner()
                target_report["metrics"][metric_name] = {
                    "metric": result.get("metric"),
                    "issue_count": result.get("issue_count"),
                }
                succeeded_metric_count += 1
            except Exception as exc:  # pragma: no cover - defensive aggregation path.
                target_report["errors"].append({"metric": metric_name, "error": str(exc)})
                failed_metric_count += 1

        target_reports.append(target_report)

    return {
        "model_filter": model_id_filter,
        "repo_filter": repo_filter,
        "target_count": len(targets),
        "metric_run_count": metric_run_count,
        "succeeded_metric_count": succeeded_metric_count,
        "failed_metric_count": failed_metric_count,
        "targets": target_reports,
    }


def _discover_metric_targets(
    *,
    output_root: Path,
    model_id_filter: Optional[str],
    repo_filter: Optional[str],
    derived_root_dirname: str = "derived",
) -> list[tuple[str, RepositoryRef]]:
    derived_root = output_root / derived_root_dirname
    if not derived_root.exists():
        return []

    repo_filter_ref = RepositoryRef.parse(repo_filter) if repo_filter else None
    model_dir_filter = _model_dir_name(model_id_filter) if model_id_filter else None

    targets: list[tuple[str, RepositoryRef]] = []
    for model_dir in sorted(path for path in derived_root.iterdir() if path.is_dir()):
        if model_dir_filter and model_dir.name != model_dir_filter:
            continue
        model_id = _model_id_from_dir(model_dir.name)

        repo_dirs = sorted(path for path in model_dir.iterdir() if path.is_dir())
        for repo_dir in repo_dirs:
            repo_ref = _repo_ref_from_fs_slug(repo_dir.name)
            if repo_ref is None:
                continue
            if repo_filter_ref and repo_ref != repo_filter_ref:
                continue
            if not any(repo_dir.glob("issue_*.json")):
                continue
            targets.append((model_id, repo_ref))

    return targets


def _model_dir_name(model_id: Optional[str]) -> str:
    if model_id is None:
        return ""
    return model_id.strip().replace("/", "__")


def _model_id_from_dir(model_dir_name: str) -> str:
    return model_dir_name.replace("__", "/")


def _repo_ref_from_fs_slug(fs_slug: str) -> Optional[RepositoryRef]:
    if "__" not in fs_slug:
        return None
    owner, name = fs_slug.split("__", 1)
    owner = owner.strip()
    name = name.strip()
    if not owner or not name:
        return None
    return RepositoryRef(owner=owner, name=name)


def _build_rank_fusion_leaderboard(
    *,
    output_root: Path,
    repo_filter: Optional[str],
    model_id_filter: Optional[str],
    rrf_k: int,
    points_max: int,
    points_step: int,
) -> dict:
    metrics_root = output_root / "metrics"
    if not metrics_root.exists():
        raise FileNotFoundError(f"No metrics directory found at {metrics_root}")

    repo_filter_ref = RepositoryRef.parse(repo_filter) if repo_filter else None
    model_dir_filter = _model_dir_name(model_id_filter) if model_id_filter else None

    # repo_slug -> model_id -> component -> value
    repo_model_components: dict[str, dict[str, dict[str, float]]] = {}
    all_models: set[str] = set()

    for model_dir in sorted(path for path in metrics_root.iterdir() if path.is_dir()):
        if model_dir_filter and model_dir.name != model_dir_filter:
            continue
        model_id = _model_id_from_dir(model_dir.name)

        for repo_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            repo_ref = _repo_ref_from_fs_slug(repo_dir.name)
            if repo_ref is None:
                continue
            if repo_filter_ref and repo_ref != repo_filter_ref:
                continue

            components = _load_leaderboard_components(repo_dir)
            if not components:
                continue

            repo_bucket = repo_model_components.setdefault(repo_ref.slug, {})
            repo_bucket[model_id] = components
            all_models.add(model_id)

    if not repo_model_components:
        raise FileNotFoundError("No leaderboard-compatible metrics found for the provided filters.")

    per_repo_reports = []
    for repo_slug in sorted(repo_model_components.keys()):
        model_components = repo_model_components[repo_slug]
        repo_scores, repo_max_possible = _compute_rrf_scores(model_components, rrf_k=rrf_k)
        per_repo_reports.append(
            {
                "repository": repo_slug,
                "model_count": len(model_components),
                "max_possible_score": repo_max_possible,
                "leaderboard": _serialize_leaderboard_rows(
                    repo_scores,
                    max_possible_score=repo_max_possible,
                    points_max=points_max,
                    points_step=points_step,
                ),
            }
        )

    combined_scores, combined_components, combined_max_possible = _compute_combined_rrf_scores(
        repo_model_components,
        rrf_k=rrf_k,
    )

    return {
        "metric": "rank_fusion_leaderboard",
        "rrf_k": int(max(1, rrf_k)),
        "points": {
            "max": int(max(1, points_max)),
            "step": int(max(1, points_step)),
            "formula": "points = round_to_step((score / max_possible_score) * points_max)",
        },
        "repo_filter": repo_filter,
        "model_filter": model_id_filter,
        "components": sorted(_leaderboard_component_names()),
        "repo_count": len(repo_model_components),
        "model_count": len(all_models),
        "per_repo": per_repo_reports,
        "all_repos_combined": {
            "max_possible_score": combined_max_possible,
            "leaderboard": _serialize_leaderboard_rows(
                combined_scores,
                component_values=combined_components,
                max_possible_score=combined_max_possible,
                points_max=points_max,
                points_step=points_step,
            ),
        },
    }


def _load_leaderboard_components(repo_metrics_dir: Path) -> dict[str, float]:
    components: dict[str, float] = {}

    type_path = repo_metrics_dir / "type_matching.json"
    if type_path.exists():
        payload = read_json(type_path)
        _put_float(components, "type_f1", _get_path(payload, ("macro_average", "f1")))

    metadata_path = repo_metrics_dir / "metadata_matching.json"
    if metadata_path.exists():
        payload = read_json(metadata_path)
        _put_float(components, "metadata_f1", _get_path(payload, ("macro_average", "f1")))
        _put_float(components, "metadata_soft_f1", _get_path(payload, ("macro_average", "soft_f1")))

    tag_path = repo_metrics_dir / "tag_matching.json"
    if tag_path.exists():
        payload = read_json(tag_path)
        _put_float(components, "tag_f1", _get_path(payload, ("overall", "f1")))

    summary_path = repo_metrics_dir / "summary_matching.json"
    if summary_path.exists():
        payload = read_json(summary_path)
        base = (
            "overall",
            "all_issues_macro_with_unmatched_penalty",
        )
        _put_float(components, "summary_codebert", _get_path(payload, base + ("codebert", "cosine")))
        _put_float(components, "summary_bertscore_f1", _get_path(payload, base + ("bertscore", "f1")))
        _put_float(components, "summary_bleurt", _get_path(payload, base + ("bleurt", "score")))

    return components


def _leaderboard_component_names() -> tuple[str, ...]:
    return (
        "type_f1",
        "metadata_f1",
        "metadata_soft_f1",
        "tag_f1",
        "summary_codebert",
        "summary_bertscore_f1",
        "summary_bleurt",
    )


def _compute_rrf_scores(
    model_components: dict[str, dict[str, float]],
    *,
    rrf_k: int,
) -> tuple[list[dict], float]:
    k = int(max(1, rrf_k))
    scores: dict[str, float] = defaultdict(float)
    rank_maps: dict[str, dict[str, int]] = {}
    contributing_rank_lists = 0

    for component in _leaderboard_component_names():
        ranked = sorted(
            ((model_id, values[component]) for model_id, values in model_components.items() if component in values),
            key=lambda item: (-float(item[1]), item[0]),
        )
        if not ranked:
            continue
        contributing_rank_lists += 1
        rank_map: dict[str, int] = {}
        for rank, (model_id, _value) in enumerate(ranked, start=1):
            scores[model_id] += 1.0 / float(k + rank)
            rank_map[model_id] = rank
        rank_maps[component] = rank_map

    rows = []
    for model_id in sorted(model_components.keys()):
        rows.append(
            {
                "model_id": model_id,
                "score": float(scores.get(model_id, 0.0)),
                "component_values": model_components.get(model_id, {}),
                "component_ranks": {
                    component: rank_maps.get(component, {}).get(model_id)
                    for component in _leaderboard_component_names()
                    if component in model_components.get(model_id, {})
                },
            }
        )

    rows.sort(key=lambda item: (-float(item["score"]), item["model_id"]))
    max_possible_score = float(contributing_rank_lists) * (1.0 / float(k + 1))
    return rows, max_possible_score


def _compute_combined_rrf_scores(
    repo_model_components: dict[str, dict[str, dict[str, float]]],
    *,
    rrf_k: int,
) -> tuple[list[dict], dict[str, dict[str, float]], float]:
    k = int(max(1, rrf_k))
    scores: dict[str, float] = defaultdict(float)
    component_values_by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    repo_presence: dict[str, set[str]] = defaultdict(set)
    contributing_rank_lists = 0

    for repo_slug, model_components in repo_model_components.items():
        for model_id, comps in model_components.items():
            repo_presence[model_id].add(repo_slug)
            for component, value in comps.items():
                component_values_by_model[model_id][component].append(float(value))

        for component in _leaderboard_component_names():
            ranked = sorted(
                ((model_id, values[component]) for model_id, values in model_components.items() if component in values),
                key=lambda item: (-float(item[1]), item[0]),
            )
            if ranked:
                contributing_rank_lists += 1
            for rank, (model_id, _value) in enumerate(ranked, start=1):
                scores[model_id] += 1.0 / float(k + rank)

    averaged_components: dict[str, dict[str, float]] = {}
    for model_id, component_values in component_values_by_model.items():
        averaged_components[model_id] = {
            component: (sum(values) / float(len(values)))
            for component, values in component_values.items()
            if values
        }

    rows = []
    for model_id in sorted(averaged_components.keys()):
        rows.append(
            {
                "model_id": model_id,
                "score": float(scores.get(model_id, 0.0)),
                "repo_count": len(repo_presence.get(model_id, set())),
            }
        )
    rows.sort(key=lambda item: (-float(item["score"]), item["model_id"]))
    max_possible_score = float(contributing_rank_lists) * (1.0 / float(k + 1))
    return rows, averaged_components, max_possible_score


def _serialize_leaderboard_rows(
    rows: list[dict],
    *,
    component_values: Optional[dict[str, dict[str, float]]] = None,
    max_possible_score: Optional[float] = None,
    points_max: int = 1000,
    points_step: int = 10,
) -> list[dict]:
    if not rows:
        return []

    top_score = float(rows[0].get("score", 0.0))
    max_score = float(max_possible_score) if max_possible_score is not None else top_score
    output_rows = []
    max_points = int(max(1, points_max))
    step = int(max(1, points_step))
    leader_points = 0
    for index, row in enumerate(rows, start=1):
        model_id = str(row.get("model_id", ""))
        score = float(row.get("score", 0.0))
        normalized = 0.0 if max_score <= 0.0 else (score / max_score) * 100.0
        raw_points = 0.0 if max_score <= 0.0 else (score / max_score) * float(max_points)
        quantized_points = int(round(raw_points / float(step)) * step)
        quantized_points = max(0, min(max_points, quantized_points))
        if index == 1:
            leader_points = quantized_points
        payload = {
            "rank": index,
            "model_id": model_id,
            "score": score,
            "normalized_score": normalized,
            "points": quantized_points,
            "points_to_leader": leader_points - quantized_points,
        }
        if "repo_count" in row:
            payload["repo_count"] = int(row.get("repo_count", 0))
        if "component_values" in row:
            payload["component_values"] = row.get("component_values", {})
        elif component_values is not None:
            payload["component_values"] = component_values.get(model_id, {})
        if "component_ranks" in row:
            payload["component_ranks"] = row.get("component_ranks", {})
        output_rows.append(payload)
    return output_rows


def _get_path(payload: dict, path: tuple[str, ...]) -> Optional[float]:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _put_float(target: dict[str, float], key: str, value: Optional[float]) -> None:
    if value is None:
        return
    target[key] = float(value)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tbyc_dataset.models import RepositoryRef


COMPONENT_KEYS: Sequence[str] = (
    "metadata_f1",
    "metadata_soft_f1",
    "tag_f1",
    "type_f1",
    "summary_bleurt",
    "summary_bertscore_f1",
    "summary_codebert",
)


@dataclass(frozen=True)
class _MetricsBundle:
    repository: str
    model_id: str
    components: Dict[str, float]
    per_issue: Dict[str, List[float]]
    per_type: Dict[str, Dict[str, float]]
    human_tag_frequency: Dict[str, int]
    llm_tag_frequency: Dict[str, int]


def generate_metrics_visualizations(
    *,
    output_root: str,
    graphs_root: Optional[str] = None,
    repo: Optional[str] = None,
    model_id: Optional[str] = None,
    metrics_root_dirname: str = "metrics",
    graphs_root_dirname: str = "graphs",
    points_max: int = 100,
    points_step: int = 1,
    random_seed: int = 7,
) -> Dict[str, Any]:
    plt, np = _load_plotting_dependencies()
    random.seed(random_seed)

    data_root = Path(output_root)
    metrics_root = data_root / metrics_root_dirname
    if not metrics_root.exists():
        raise FileNotFoundError(f"No metrics directory found at {metrics_root}")

    target_root = Path(graphs_root) if graphs_root else (data_root / graphs_root_dirname)
    target_root.mkdir(parents=True, exist_ok=True)

    repo_filter = RepositoryRef.parse(repo) if repo else None
    model_dir_filter = _model_dir_name(model_id) if model_id else None

    bundles = _load_metric_bundles(metrics_root, repo_filter=repo_filter, model_dir_filter=model_dir_filter)
    if not bundles:
        raise FileNotFoundError("No metric bundles found for the provided filters.")

    model_colors = _build_model_color_map([bundle.model_id for bundle in bundles], plt)

    files: List[str] = []
    bundles_by_repo: Dict[str, List[_MetricsBundle]] = {}
    for bundle in bundles:
        bundles_by_repo.setdefault(bundle.repository, []).append(bundle)

    for repo_slug, repo_bundles in sorted(bundles_by_repo.items()):
        files.extend(_plot_repo_overall_bars(target_root, repo_slug, repo_bundles, plt, np, model_colors=model_colors))
        files.extend(_plot_repo_issue_boxplots(target_root, repo_slug, repo_bundles, plt, np, model_colors=model_colors))
        files.extend(_plot_repo_issue_violin(target_root, repo_slug, repo_bundles, plt, np, model_colors=model_colors))
        files.extend(_plot_repo_issue_scatter(target_root, repo_slug, repo_bundles, plt, np, model_colors=model_colors))

    files.extend(_plot_global_component_heatmap(target_root, bundles, plt, np, model_colors=model_colors))
    files.extend(_plot_global_model_lines(target_root, bundles, plt, np, model_colors=model_colors))
    files.extend(
        _plot_global_points_bar(
            target_root,
            bundles,
            plt,
            np,
            model_colors=model_colors,
            points_max=points_max,
            points_step=points_step,
        )
    )
    files.extend(_plot_global_per_type_metric_bars(target_root, bundles, plt, np, model_colors=model_colors))
    files.extend(_plot_global_per_type_metric_spiders(target_root, bundles, plt, np, model_colors=model_colors))
    dataset_stats = _compute_dataset_stats(bundles)
    files.extend(_plot_dataset_tag_frequencies(target_root, bundles, plt, np, top_k=25))

    manifest = {
        "output_root": str(data_root),
        "graphs_root": str(target_root),
        "repo_filter": repo,
        "model_filter": model_id,
        "points": {
            "max": int(max(1, points_max)),
            "step": int(max(1, points_step)),
            "formula": "points = round_to_step(score * points_max)",
            "score_definition": "average across component metrics",
        },
        "repository_count": len(bundles_by_repo),
        "bundle_count": len(bundles),
        "dataset_stats": dataset_stats,
        "graph_count": len(files),
        "files": sorted(files),
    }
    (target_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_plotting_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "Visualization dependencies unavailable. Install matplotlib and numpy to use visualize-metrics."
        ) from exc
    return plt, np


def _write_graph_sidecar(path: Path, payload: Mapping[str, Any]) -> str:
    txt_path = path.with_suffix(".txt")
    txt_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(txt_path)


def _finalize_graph(fig, path: Path, plt, payload: Mapping[str, Any]) -> List[str]:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    _write_graph_sidecar(path, payload)
    return [str(path)]


def _load_metric_bundles(
    metrics_root: Path,
    *,
    repo_filter: Optional[RepositoryRef],
    model_dir_filter: Optional[str],
) -> List[_MetricsBundle]:
    data_root = metrics_root.parent
    bundles: List[_MetricsBundle] = []
    for model_dir in sorted(path for path in metrics_root.iterdir() if path.is_dir()):
        if model_dir_filter and model_dir.name != model_dir_filter:
            continue
        model_name = _model_id_from_dir(model_dir.name)
        for repo_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            repo_ref = _repo_ref_from_fs_slug(repo_dir.name)
            if repo_ref is None:
                continue
            if repo_filter and repo_ref != repo_filter:
                continue

            type_payload = _read_json_if_exists(repo_dir / "type_matching.json")
            metadata_payload = _read_json_if_exists(repo_dir / "metadata_matching.json")
            tag_payload = _read_json_if_exists(repo_dir / "tag_matching.json")
            summary_payload = _read_json_if_exists(repo_dir / "summary_matching.json")
            if not (type_payload and metadata_payload and tag_payload and summary_payload):
                continue

            components = {
                "type_f1": _float_path(type_payload, ("macro_average", "f1")),
                "metadata_f1": _float_path(metadata_payload, ("macro_average", "f1")),
                "metadata_soft_f1": _float_path(metadata_payload, ("macro_average", "soft_f1")),
                "tag_f1": _float_path(tag_payload, ("overall", "f1")),
                "summary_bertscore_f1": _float_path(
                    summary_payload,
                    ("overall", "all_issues_macro_with_unmatched_penalty", "bertscore", "f1"),
                ),
                "summary_bleurt": _float_path(
                    summary_payload,
                    ("overall", "all_issues_macro_with_unmatched_penalty", "bleurt", "score"),
                ),
                "summary_codebert": _float_path(
                    summary_payload,
                    ("overall", "all_issues_macro_with_unmatched_penalty", "codebert", "cosine"),
                ),
            }

            per_issue = {
                "type_f1": _issue_values(type_payload, ("f1",)),
                "metadata_f1": _issue_values(metadata_payload, ("pooled", "f1")),
                "metadata_soft_f1": _issue_values(metadata_payload, ("pooled", "soft_f1")),
                "tag_f1": _issue_values(tag_payload, ("pooled", "f1")),
                "summary_bertscore_f1": _issue_values(
                    summary_payload,
                    ("scores_with_unmatched_penalty", "bertscore", "f1"),
                ),
                "summary_bleurt": _issue_values(
                    summary_payload,
                    ("scores_with_unmatched_penalty", "bleurt", "score"),
                ),
                "summary_codebert": _issue_values(
                    summary_payload,
                    ("scores_with_unmatched_penalty", "codebert", "cosine"),
                ),
            }

            per_type = _extract_per_type_metrics(
                type_payload=type_payload,
                metadata_payload=metadata_payload,
                tag_payload=tag_payload,
                summary_payload=summary_payload,
            )
            human_tag_frequency = _load_tag_frequency_from_issue_payloads(data_root / "extractions" / repo_ref.fs_slug)
            llm_tag_frequency = _load_tag_frequency_from_issue_payloads(
                data_root / "derived" / model_dir.name / repo_ref.fs_slug
            )

            bundles.append(
                _MetricsBundle(
                    repository=repo_ref.slug,
                    model_id=model_name,
                    components={k: v for k, v in components.items() if not math.isnan(v)},
                    per_issue=per_issue,
                    per_type=per_type,
                    human_tag_frequency=human_tag_frequency,
                    llm_tag_frequency=llm_tag_frequency,
                )
            )
    return bundles


def _plot_repo_overall_bars(
    target_root: Path,
    repo_slug: str,
    bundles: List[_MetricsBundle],
    plt,
    np,
    *,
    model_colors: Mapping[str, Any],
) -> List[str]:
    if not bundles:
        return []
    ordered_bundles = sorted(bundles, key=lambda bundle: _model_sort_key(bundle.model_id))
    models = [bundle.model_id for bundle in ordered_bundles]
    model_labels = [_display_model_name(bundle.model_id) for bundle in ordered_bundles]
    x = np.arange(len(COMPONENT_KEYS))
    keys = [key for key in COMPONENT_KEYS if any(key in bundle.components for bundle in ordered_bundles)]
    if not keys:
        return []

    fig, ax = plt.subplots(figsize=(max(12, len(models) * 1.4), 7))
    family_offsets, width = _family_relative_offsets(models, cluster_width=0.82, gap_units=0.16)
    for idx, bundle in enumerate(ordered_bundles):
        values = [bundle.components.get(key, 0.0) for key in keys]
        offset = x + family_offsets[idx]
        ax.bar(
            offset,
            values,
            width=width,
            label=model_labels[idx],
            color=model_colors.get(bundle.model_id),
        )
    ax.set_title(f"Overall Metrics by Model ({repo_slug})")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    out = target_root / f"repo_{repo_slug.replace('/', '__')}_overall_grouped_bar.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "repo_overall_grouped_bar",
            "repository": repo_slug,
            "models": models,
            "model_labels": model_labels,
            "metric_keys": keys,
            "values_by_model": {
                bundle.model_id: {key: bundle.components.get(key, 0.0) for key in keys} for bundle in ordered_bundles
            },
        },
    )


def _plot_repo_issue_boxplots(
    target_root: Path,
    repo_slug: str,
    bundles: List[_MetricsBundle],
    plt,
    np,
    *,
    model_colors: Mapping[str, Any],
) -> List[str]:
    keys = ["type_f1", "metadata_f1", "tag_f1", "summary_bertscore_f1"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for i, key in enumerate(keys):
        ax = axes_flat[i]
        data = [bundle.per_issue.get(key, []) for bundle in bundles]
        labels = [_display_model_name(bundle.model_id) for bundle in bundles]
        if not any(data):
            ax.set_visible(False)
            continue
        box = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        for j, patch in enumerate(box.get("boxes", [])):
            color = model_colors.get(bundles[j].model_id)
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for j, whisker in enumerate(box.get("whiskers", [])):
            color = model_colors.get(bundles[j // 2].model_id)
            whisker.set_color(color)
        for j, cap in enumerate(box.get("caps", [])):
            color = model_colors.get(bundles[j // 2].model_id)
            cap.set_color(color)
        for j, median in enumerate(box.get("medians", [])):
            color = model_colors.get(bundles[j].model_id)
            median.set_color(color)
            median.set_linewidth(2.0)
        for idx, values in enumerate(data, start=1):
            if not values:
                continue
            jitter = np.random.normal(loc=0.0, scale=0.04, size=len(values))
            xs = np.full(len(values), idx) + jitter
            ax.scatter(xs, values, s=12, alpha=0.35, color=model_colors.get(bundles[idx - 1].model_id))
        ax.set_title(f"Per-Issue {key}")
        if key == "metadata_f1":
            _set_data_driven_ylim(ax, data)
        else:
            ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"Per-Issue Boxplots with Data Points ({repo_slug})")
    out = target_root / f"repo_{repo_slug.replace('/', '__')}_issue_boxplots.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "repo_issue_boxplots",
            "repository": repo_slug,
            "values_by_model": {
                bundle.model_id: {key: bundle.per_issue.get(key, []) for key in keys} for bundle in bundles
            },
        },
    )


def _plot_repo_issue_violin(
    target_root: Path,
    repo_slug: str,
    bundles: List[_MetricsBundle],
    plt,
    np,
    *,
    model_colors: Mapping[str, Any],
) -> List[str]:
    keys = ["metadata_soft_f1", "summary_bleurt", "summary_codebert", "summary_bertscore_f1"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    for i, key in enumerate(keys):
        ax = axes_flat[i]
        data = [bundle.per_issue.get(key, []) for bundle in bundles]
        labels = [_display_model_name(bundle.model_id) for bundle in bundles]
        plotted_indices = [j for j, series in enumerate(data) if series]
        plotted_data = [data[j] for j in plotted_indices]
        if not plotted_data:
            ax.set_visible(False)
            continue
        vp = ax.violinplot(plotted_data, showmeans=True, showextrema=False)
        for body, source_index in zip(vp.get("bodies", []), plotted_indices):
            color = model_colors.get(bundles[source_index].model_id)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.3)
        if "cmeans" in vp:
            vp["cmeans"].set_color("black")
            vp["cmeans"].set_linewidth(1.2)
        for idx, values in enumerate(data, start=1):
            if not values:
                continue
            jitter = np.random.normal(loc=0.0, scale=0.03, size=len(values))
            xs = np.full(len(values), idx) + jitter
            ax.scatter(xs, values, s=8, alpha=0.25, color=model_colors.get(bundles[idx - 1].model_id))
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(f"Per-Issue {key}")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"Per-Issue Violin Plots ({repo_slug})")
    out = target_root / f"repo_{repo_slug.replace('/', '__')}_issue_violin.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "repo_issue_violin",
            "repository": repo_slug,
            "values_by_model": {
                bundle.model_id: {key: bundle.per_issue.get(key, []) for key in keys} for bundle in bundles
            },
        },
    )


def _plot_repo_issue_scatter(
    target_root: Path,
    repo_slug: str,
    bundles: List[_MetricsBundle],
    plt,
    np,
    *,
    model_colors: Mapping[str, Any],
) -> List[str]:
    fig, ax = plt.subplots(figsize=(12, 7))
    any_points = False
    aggregate_label_added = False
    for bundle in bundles:
        xvals = bundle.per_issue.get("summary_bertscore_f1", [])
        yvals = bundle.per_issue.get("metadata_soft_f1", [])
        count = min(len(xvals), len(yvals))
        if count == 0:
            continue
        any_points = True
        x = np.array(xvals[:count])
        y = np.array(yvals[:count])
        color = model_colors.get(bundle.model_id)
        ax.scatter(x, y, s=20, alpha=0.45, color=color, label=_display_model_name(bundle.model_id))
        cross_label = "Model aggregate (X)" if not aggregate_label_added else None
        ax.scatter(
            [bundle.components.get("summary_bertscore_f1", 0.0)],
            [bundle.components.get("metadata_soft_f1", 0.0)],
            s=120,
            marker="X",
            color=color,
            edgecolors="black",
            linewidths=0.7,
            label=cross_label,
        )
        aggregate_label_added = True

    if not any_points:
        plt.close(fig)
        return []

    ax.set_title(f"Per-Issue Scatter: Summary vs Metadata Soft F1 ({repo_slug})")
    ax.set_xlabel("summary_bertscore_f1")
    ax.set_ylabel("metadata_soft_f1")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 0.4)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    out = target_root / f"repo_{repo_slug.replace('/', '__')}_issue_scatter.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "repo_issue_scatter",
            "repository": repo_slug,
            "x_metric": "summary_bertscore_f1",
            "y_metric": "metadata_soft_f1",
            "points_by_model": {
                bundle.model_id: {
                    "x": bundle.per_issue.get("summary_bertscore_f1", []),
                    "y": bundle.per_issue.get("metadata_soft_f1", []),
                    "aggregate_x": bundle.components.get("summary_bertscore_f1", 0.0),
                    "aggregate_y": bundle.components.get("metadata_soft_f1", 0.0),
                }
                for bundle in bundles
            },
        },
    )


def _plot_global_component_heatmap(
    target_root: Path,
    bundles: List[_MetricsBundle],
    plt,
    np,
    *,
    model_colors: Mapping[str, Any],
) -> List[str]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for bundle in bundles:
        row = grouped.setdefault(bundle.model_id, {key: [] for key in COMPONENT_KEYS})
        for key in COMPONENT_KEYS:
            if key in bundle.components:
                row[key].append(bundle.components[key])

    models = sorted(grouped.keys())
    if not models:
        return []

    matrix = []
    for model in models:
        matrix.append([
            (sum(grouped[model][key]) / float(len(grouped[model][key])) if grouped[model][key] else 0.0)
            for key in COMPONENT_KEYS
        ])
    arr = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, max(5, len(models) * 0.6)))
    im = ax.imshow(arr, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Model x Component Heatmap (Averaged Across Repos)")
    ax.set_xticks(np.arange(len(COMPONENT_KEYS)))
    ax.set_xticklabels(COMPONENT_KEYS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([_display_model_name(model) for model in models])
    for tick_label, model in zip(ax.get_yticklabels(), models):
        tick_label.set_color(model_colors.get(model))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    out = target_root / "global_component_heatmap.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "global_component_heatmap",
            "models": models,
            "metric_keys": list(COMPONENT_KEYS),
            "matrix": arr.tolist(),
        },
    )


def _plot_global_model_lines(
    target_root: Path,
    bundles: List[_MetricsBundle],
    plt,
    np,
    *,
    model_colors: Mapping[str, Any],
) -> List[str]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for bundle in bundles:
        row = grouped.setdefault(bundle.model_id, {key: [] for key in COMPONENT_KEYS})
        for key in COMPONENT_KEYS:
            if key in bundle.components:
                row[key].append(bundle.components[key])
    if not grouped:
        return []

    models = sorted(grouped.keys())
    models = sorted(models, key=_model_sort_key)
    display_models = [_display_model_name(model) for model in models]
    component_scores: List[List[float]] = []
    for model in models:
        vals_row: List[float] = []
        for key in COMPONENT_KEYS:
            values = grouped[model].get(key, [])
            vals_row.append(sum(values) / float(len(values)) if values else 0.0)
        component_scores.append(vals_row)

    score_array = np.array(component_scores)
    x = np.arange(len(COMPONENT_KEYS))
    family_offsets, width = _family_relative_offsets(models, cluster_width=0.82, gap_units=0.16)

    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, model_name in enumerate(display_models):
        offsets = x + family_offsets[idx]
        ax.bar(offsets, score_array[idx], width=width, label=model_name, color=model_colors.get(models[idx]))

    data_min = float(np.min(score_array)) if score_array.size else 0.0
    data_max = float(np.max(score_array)) if score_array.size else 1.0
    if data_max - data_min < 0.04:
        pad = 0.02
    else:
        pad = (data_max - data_min) * 0.15
    lower = max(0.0, data_min - pad)
    upper = min(1.0, data_max + pad)
    if upper <= lower:
        upper = min(1.0, lower + 0.05)

    ax.set_title("Model Profiles Across Metric Components (Grouped Bar)")
    ax.set_xticks(x)
    ax.set_xticklabels(COMPONENT_KEYS, rotation=20, ha="right")
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    out = target_root / "global_model_component_lines.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "global_model_component_lines",
            "models": models,
            "metric_keys": list(COMPONENT_KEYS),
            "scores": {model: component_scores[idx] for idx, model in enumerate(models)},
        },
    )


def _plot_global_points_bar(
    target_root: Path,
    bundles: List[_MetricsBundle],
    plt,
    np,
    *,
    model_colors: Mapping[str, Any],
    points_max: int,
    points_step: int,
) -> List[str]:
    grouped: Dict[str, List[float]] = {}
    for bundle in bundles:
        vals = [bundle.components.get(key, 0.0) for key in COMPONENT_KEYS]
        score = sum(vals) / float(len(COMPONENT_KEYS))
        grouped.setdefault(bundle.model_id, []).append(score)

    models = sorted(grouped.keys())
    if not models:
        return []

    scores = {model: sum(grouped[model]) / float(len(grouped[model])) for model in models}
    ranked = sorted(scores.items(), key=lambda item: (_model_sort_key(item[0]), -float(item[1])))
    max_points = int(max(1, points_max))
    step = int(max(1, points_step))
    points = []
    for _, score in ranked:
        raw_points = float(score) * float(max_points)
        quantized = int(round(raw_points / float(step)) * step)
        points.append(max(0, min(max_points, quantized)))

    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [_display_model_name(name) for name, _ in ranked]
    colors = [model_colors.get(name) for name, _ in ranked]
    ranked_models = [name for name, _ in ranked]
    xpos = _family_spaced_positions(ranked_models, gap_units=0.16)
    bars = ax.bar(xpos, points, width=1.0, color=colors)
    ax.set_title("Discrete Points (Average Component Baseline, Max-Possible Normalized)")
    ax.set_ylabel("Points")
    y_top = min(max_points, 50)
    ax.set_ylim(0, y_top)
    ax.axhline(max_points, color="red", linestyle=":", linewidth=1.8, label="max score")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, points):
        x = bar.get_x() + (bar.get_width() / 2.0)
        y = val + 0.5
        if y >= y_top:
            y = max(0.5, y_top - 0.8)
        ax.text(x, y, str(val), ha="center", va="bottom", fontsize=9)
    out = target_root / "global_points_bar.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "global_points_bar",
            "points_max": max_points,
            "points_step": step,
            "ranked_models": [name for name, _ in ranked],
            "scores": {name: score for name, score in ranked},
            "points": {name: points[idx] for idx, (name, _) in enumerate(ranked)},
        },
    )


def _plot_global_per_type_metric_bars(
    target_root: Path,
    bundles: List[_MetricsBundle],
    plt,
    np,
    *,
    model_colors: Mapping[str, Any],
) -> List[str]:
    aggregated = _aggregate_per_type_scores(bundles)
    models = sorted(aggregated.keys(), key=_model_sort_key)
    if not models:
        return []

    files: List[str] = []
    family_offsets, width = _family_relative_offsets(models, cluster_width=0.84, gap_units=0.16)
    for metric in COMPONENT_KEYS:
        type_names = sorted(
            {
                type_name
                for model in models
                for type_name, values in aggregated[model].items()
                if metric in values
            }
        )
        if not type_names:
            continue

        x = np.arange(len(type_names))
        fig, ax = plt.subplots(figsize=(max(16, len(type_names) * 0.42), 8))
        for idx, model in enumerate(models):
            vals = [aggregated[model].get(type_name, {}).get(metric, 0.0) for type_name in type_names]
            ax.bar(
                x + family_offsets[idx],
                vals,
                width=width,
                color=model_colors.get(model),
                label=_display_model_name(model),
            )

        ax.set_title(f"Per-Type {metric} by Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels([_display_type_name(type_name) for type_name in type_names], rotation=65, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        out = target_root / f"global_per_type_{metric}_grouped_bar.png"
        files.extend(
            _finalize_graph(
                fig,
                out,
                plt,
                {
                    "graph": "global_per_type_grouped_bar",
                    "metric": metric,
                    "type_names": type_names,
                    "values_by_model": {
                        model: [aggregated[model].get(type_name, {}).get(metric, 0.0) for type_name in type_names]
                        for model in models
                    },
                },
            )
        )

    return files


def _plot_global_per_type_metric_spiders(
    target_root: Path,
    bundles: List[_MetricsBundle],
    plt,
    np,
    *,
    model_colors: Mapping[str, Any],
) -> List[str]:
    aggregated = _aggregate_per_type_scores(bundles)
    models = sorted(aggregated.keys(), key=_model_sort_key)
    if not models:
        return []

    files: List[str] = []
    for metric in COMPONENT_KEYS:
        type_names = sorted(
            {
                type_name
                for model in models
                for type_name, values in aggregated[model].items()
                if metric in values
            }
        )
        if len(type_names) < 3:
            continue

        score_by_type: Dict[str, float] = {}
        for type_name in type_names:
            vals = [aggregated[model].get(type_name, {}).get(metric) for model in models]
            clean = [float(v) for v in vals if isinstance(v, (int, float))]
            if clean:
                score_by_type[type_name] = sum(clean) / float(len(clean))
        ordered_types = _spread_types_for_spider(type_names, score_by_type)

        angles = np.linspace(0, 2 * np.pi, len(ordered_types), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={"projection": "polar"})
        for model in models:
            vals = [aggregated[model].get(type_name, {}).get(metric, 0.0) for type_name in ordered_types]
            vals += vals[:1]
            color = model_colors.get(model)
            ax.plot(angles, vals, linewidth=1.6, color=color, label=_display_model_name(model))
            ax.fill(angles, vals, color=color, alpha=0.04)

        ax.set_title(f"Per-Type Spider: {metric}")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([_display_type_name(type_name) for type_name in ordered_types], fontsize=7)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.05), fontsize=8)
        out = target_root / f"global_per_type_{metric}_spider.png"
        files.extend(
            _finalize_graph(
                fig,
                out,
                plt,
                {
                    "graph": "global_per_type_spider",
                    "metric": metric,
                    "ordered_types": ordered_types,
                    "values_by_model": {
                        model: [aggregated[model].get(type_name, {}).get(metric, 0.0) for type_name in ordered_types]
                        for model in models
                    },
                },
            )
        )

    return files


def _plot_dataset_tag_frequencies(
    target_root: Path,
    bundles: Sequence[_MetricsBundle],
    plt,
    np,
    *,
    top_k: int,
) -> List[str]:
    human_counts: Dict[str, int] = {}
    llm_counts: Dict[str, int] = {}
    for bundle in bundles:
        for tag, count in bundle.human_tag_frequency.items():
            human_counts[tag] = human_counts.get(tag, 0) + int(count)
        for tag, count in bundle.llm_tag_frequency.items():
            llm_counts[tag] = llm_counts.get(tag, 0) + int(count)

    if not human_counts and not llm_counts:
        return []

    tags_union = set(human_counts.keys()) | set(llm_counts.keys())
    ranked_tags = sorted(
        tags_union,
        key=lambda tag: (-(human_counts.get(tag, 0) + llm_counts.get(tag, 0)), tag),
    )
    selected_tags = ranked_tags[: int(max(1, top_k))]
    labels = [_display_type_name(tag) for tag in selected_tags]
    human_values = [human_counts.get(tag, 0) for tag in selected_tags]
    llm_values = [llm_counts.get(tag, 0) for tag in selected_tags]

    y = np.arange(len(selected_tags))
    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, len(selected_tags) * 0.42)), sharey=True)

    axes[0].barh(y, human_values, color="#4C78A8")
    axes[0].set_title("Human Tag Frequency (Dataset-Level)")
    axes[0].set_xlabel("Count")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)
    _annotate_barh_counts(axes[0], human_values)

    axes[1].barh(y, llm_values, color="#F58518")
    axes[1].set_title("Model-Extracted Tag Frequency (Dataset-Level)")
    axes[1].set_xlabel("Count")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].tick_params(axis="y", labelleft=True)
    _annotate_barh_counts(axes[1], llm_values)

    fig.suptitle("Tag Frequency Across Repositories and Issues")
    out = target_root / "dataset_tag_frequency_comparison.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "dataset_tag_frequency_comparison",
            "tags": selected_tags,
            "human_values": human_values,
            "llm_values": llm_values,
        },
    )


def _annotate_barh_counts(ax, values: Sequence[int]) -> None:
    max_value = max(values) if values else 0
    x_pad = max(1.0, max_value * 0.01)
    for idx, value in enumerate(values):
        ax.text(
            float(value) + x_pad,
            idx,
            str(int(value)),
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )


def _compute_dataset_stats(bundles: Sequence[_MetricsBundle]) -> Dict[str, Any]:
    repositories = sorted({bundle.repository for bundle in bundles})
    models = sorted({bundle.model_id for bundle in bundles}, key=_model_sort_key)
    issue_points = sum(len(bundle.per_issue.get("type_f1", [])) for bundle in bundles)

    human_counts: Dict[str, int] = {}
    llm_counts: Dict[str, int] = {}
    for bundle in bundles:
        for tag, count in bundle.human_tag_frequency.items():
            human_counts[tag] = human_counts.get(tag, 0) + int(count)
        for tag, count in bundle.llm_tag_frequency.items():
            llm_counts[tag] = llm_counts.get(tag, 0) + int(count)

    return {
        "repository_count": len(repositories),
        "repositories": repositories,
        "model_count": len(models),
        "models": models,
        "issue_points": int(issue_points),
        "human_unique_tag_count": len(human_counts),
        "llm_unique_tag_count": len(llm_counts),
        "human_total_tag_mentions": int(sum(human_counts.values())),
        "llm_total_tag_mentions": int(sum(llm_counts.values())),
    }


def _aggregate_per_type_scores(bundles: Sequence[_MetricsBundle]) -> Dict[str, Dict[str, Dict[str, float]]]:
    aggregate_lists: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for bundle in bundles:
        model_slot = aggregate_lists.setdefault(bundle.model_id, {})
        for type_name, metric_values in bundle.per_type.items():
            type_slot = model_slot.setdefault(type_name, {})
            for metric, value in metric_values.items():
                if metric not in COMPONENT_KEYS:
                    continue
                if not isinstance(value, (int, float)) or math.isnan(float(value)):
                    continue
                type_slot.setdefault(metric, []).append(float(value))

    aggregate_means: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_id, types_map in aggregate_lists.items():
        out_types: Dict[str, Dict[str, float]] = {}
        for type_name, metric_lists in types_map.items():
            out_metrics: Dict[str, float] = {}
            for metric, values in metric_lists.items():
                if values:
                    out_metrics[metric] = sum(values) / float(len(values))
            if out_metrics:
                out_types[type_name] = out_metrics
        aggregate_means[model_id] = out_types
    return aggregate_means


def _set_data_driven_ylim(ax, data: Sequence[Sequence[float]]) -> None:
    flat = [float(value) for series in data for value in series]
    if not flat:
        ax.set_ylim(0.0, 1.0)
        return
    max_value = max(flat)
    upper = min(1.0, max(0.05, max_value * 1.25))
    ax.set_ylim(0.0, upper)


def _build_model_color_map(model_ids: Sequence[str], plt) -> Dict[str, Any]:
    ordered = sorted({model_id for model_id in model_ids})
    if not ordered:
        return {}

    family_base: Dict[str, tuple[float, float, float]] = {
        "gpt": (0.11, 0.47, 0.78),
        "claude": (0.84, 0.41, 0.12),
        "gemini": (0.12, 0.63, 0.41),
        "qwen": (0.54, 0.35, 0.74),
        "other": (0.42, 0.42, 0.42),
    }

    family_groups: Dict[str, List[str]] = {}
    for model_id in ordered:
        family, _ = _model_family_and_recency(model_id)
        family_groups.setdefault(family, []).append(model_id)

    mapping: Dict[str, Any] = {}
    for family, models in family_groups.items():
        base = family_base.get(family, family_base["other"])
        ranked = sorted(models, key=lambda model: (_model_family_and_recency(model)[1], model))
        count = len(ranked)
        for idx, model in enumerate(ranked):
            if count == 1:
                blend = 0.9
            else:
                blend = 0.35 + (0.65 * (idx / float(count - 1)))
            mapping[model] = _blend_rgb((1.0, 1.0, 1.0), base, blend)
    return mapping


def _blend_rgb(
    rgb_a: tuple[float, float, float],
    rgb_b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    clamped = max(0.0, min(1.0, t))
    return (
        (rgb_a[0] * (1.0 - clamped)) + (rgb_b[0] * clamped),
        (rgb_a[1] * (1.0 - clamped)) + (rgb_b[1] * clamped),
        (rgb_a[2] * (1.0 - clamped)) + (rgb_b[2] * clamped),
    )


def _model_family_and_recency(model_id: str) -> tuple[str, float]:
    normalized = model_id.strip().lower()

    if "gpt" in normalized:
        return "gpt", _extract_version_recency(normalized, "gpt")

    if "claude" in normalized or "anthropic" in normalized:
        tier = 0.0
        if "opus" in normalized:
            tier = 3.0
        elif "sonnet" in normalized:
            tier = 2.0
        elif "haiku" in normalized:
            tier = 1.0
        version = _extract_first_number(normalized)
        return "claude", (tier * 100.0) + version

    if "gemini" in normalized:
        return "gemini", _extract_version_recency(normalized, "gemini")

    if "qwen" in normalized:
        return "qwen", _extract_version_recency(normalized, "qwen")

    return "other", _extract_first_number(normalized)


def _model_sort_key(model_id: str) -> tuple[int, float, str]:
    family, recency = _model_family_and_recency(model_id)
    family_rank = {
        "gpt": 0,
        "claude": 1,
        "gemini": 2,
        "qwen": 3,
        "other": 4,
    }.get(family, 5)
    return (family_rank, recency, model_id)


def _family_relative_offsets(
    model_ids: Sequence[str],
    *,
    cluster_width: float,
    gap_units: float,
) -> tuple[List[float], float]:
    if not model_ids:
        return [], 0.0

    families = [_model_family_and_recency(model_id)[0] for model_id in model_ids]
    transitions = 0
    for i in range(1, len(families)):
        if families[i] != families[i - 1]:
            transitions += 1

    total_units = float(len(model_ids)) + (float(transitions) * float(gap_units))
    width = float(cluster_width) / max(1.0, total_units)
    start = -(cluster_width / 2.0) + (width / 2.0)

    offsets: List[float] = []
    cursor = 0.0
    prev_family: Optional[str] = None
    for model_id in model_ids:
        family = _model_family_and_recency(model_id)[0]
        if prev_family is not None and family != prev_family:
            cursor += float(gap_units)
        offsets.append(start + (cursor * width))
        cursor += 1.0
        prev_family = family

    return offsets, width


def _family_spaced_positions(model_ids: Sequence[str], gap_units: float) -> List[float]:
    if not model_ids:
        return []
    positions: List[float] = []
    cursor = 0.0
    prev_family: Optional[str] = None
    for model_id in model_ids:
        family = _model_family_and_recency(model_id)[0]
        if prev_family is not None and family != prev_family:
            cursor += float(gap_units)
        positions.append(cursor)
        cursor += 1.0
        prev_family = family
    return positions


def _extract_version_recency(text: str, token: str) -> float:
    pivot = text.find(token)
    if pivot < 0:
        return _extract_first_number(text)
    tail = text[pivot + len(token) :]
    return _extract_first_number(tail)


def _extract_first_number(text: str) -> float:
    digits = ""
    saw_digit = False
    saw_dot = False
    for ch in text:
        if ch.isdigit():
            digits += ch
            saw_digit = True
            continue
        if ch == "." and saw_digit and not saw_dot:
            digits += ch
            saw_dot = True
            continue
        if saw_digit:
            break
    try:
        return float(digits) if digits else 0.0
    except ValueError:
        return 0.0


def _display_model_name(model_id: str) -> str:
    normalized = model_id.strip()
    mapping = {
        "gemini-2.5-flash": "Gemini 2.5 Flash",
        "gemini/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite Preview",
        "us.anthropic.claude-sonnet-4-6": "Claude Sonnet 4.6",
        "us.anthropic.claude-opus-4-6-v1": "Claude Opus 4.6",
        "openai/gpt-5": "OpenAI GPT-5",
        "gpt-5.4": "OpenAI GPT-5.4",
        "gpt-4.1-mini": "OpenAI GPT-4.1 Mini",
        "qwen2.5:7b-instruct": "Qwen 2.5 7B Instruct",
    }
    if normalized in mapping:
        return mapping[normalized]

    cleaned = normalized.replace("/", " ").replace("__", " ").replace(":", " ").replace("-", " ")
    words = [word for word in cleaned.split() if word]
    if not words:
        return normalized
    return " ".join(word.upper() if word.lower() in {"gpt", "llm", "api"} else word.capitalize() for word in words)


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _float_path(payload: Mapping[str, Any], path: Sequence[str]) -> float:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return math.nan
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return math.nan


def _issue_values(payload: Mapping[str, Any], path: Sequence[str]) -> List[float]:
    values: List[float] = []
    rows = payload.get("per_issue", []) if isinstance(payload, Mapping) else []
    if not isinstance(rows, list):
        return values
    for row in rows:
        current: Any = row
        ok = True
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                ok = False
                break
            current = current[key]
        if ok and isinstance(current, (int, float)):
            values.append(float(current))
    return values


def _extract_per_type_metrics(
    *,
    type_payload: Mapping[str, Any],
    metadata_payload: Mapping[str, Any],
    tag_payload: Mapping[str, Any],
    summary_payload: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}

    _merge_metric_by_type(out, _extract_type_f1_by_type(type_payload), "type_f1")
    _merge_metric_by_type(out, _extract_metadata_metric_by_type(metadata_payload, "f1"), "metadata_f1")
    _merge_metric_by_type(out, _extract_metadata_metric_by_type(metadata_payload, "soft_f1"), "metadata_soft_f1")
    _merge_metric_by_type(out, _extract_tag_f1_by_type(tag_payload), "tag_f1")
    _merge_metric_by_type(out, _extract_summary_metric_by_type(summary_payload, "bleurt", "score"), "summary_bleurt")
    _merge_metric_by_type(
        out,
        _extract_summary_metric_by_type(summary_payload, "bertscore", "f1"),
        "summary_bertscore_f1",
    )
    _merge_metric_by_type(
        out,
        _extract_summary_metric_by_type(summary_payload, "codebert", "cosine"),
        "summary_codebert",
    )

    return out


def _merge_metric_by_type(
    target: Dict[str, Dict[str, float]],
    values: Mapping[str, float],
    metric_name: str,
) -> None:
    for type_name, value in values.items():
        if not isinstance(value, (int, float)):
            continue
        target.setdefault(type_name, {})[metric_name] = float(value)


def _extract_type_f1_by_type(type_payload: Mapping[str, Any]) -> Dict[str, float]:
    tp: Dict[str, int] = {}
    fp: Dict[str, int] = {}
    fn: Dict[str, int] = {}

    rows = type_payload.get("per_issue", []) if isinstance(type_payload, Mapping) else []
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        human_types = set(row.get("human_types", []) or [])
        llm_types = set(row.get("llm_types", []) or [])
        for type_name in human_types & llm_types:
            tp[type_name] = tp.get(type_name, 0) + 1
        for type_name in llm_types - human_types:
            fp[type_name] = fp.get(type_name, 0) + 1
        for type_name in human_types - llm_types:
            fn[type_name] = fn.get(type_name, 0) + 1

    out: Dict[str, float] = {}
    all_types = set(tp.keys()) | set(fp.keys()) | set(fn.keys())
    for type_name in all_types:
        t = tp.get(type_name, 0)
        p = fp.get(type_name, 0)
        n = fn.get(type_name, 0)
        denom = (2 * t) + p + n
        out[type_name] = 0.0 if denom <= 0 else (2.0 * float(t) / float(denom))
    return out


def _extract_metadata_metric_by_type(metadata_payload: Mapping[str, Any], metric_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    rows = _safe_path(metadata_payload, ("overall", "per_type"))
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        type_name = row.get("type")
        pooled = row.get("pooled")
        if not isinstance(type_name, str) or not isinstance(pooled, Mapping):
            continue
        value = pooled.get(metric_name)
        if isinstance(value, (int, float)):
            out[type_name] = float(value)
    return out


def _extract_tag_f1_by_type(tag_payload: Mapping[str, Any]) -> Dict[str, float]:
    sums: Dict[str, Dict[str, float]] = {}
    rows = tag_payload.get("per_issue", []) if isinstance(tag_payload, Mapping) else []
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        items = row.get("per_matching_type", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            type_name = item.get("type")
            if not isinstance(type_name, str):
                continue
            slot = sums.setdefault(type_name, {"human": 0.0, "llm": 0.0, "tp": 0.0})
            if isinstance(item.get("human_tag_count"), (int, float)):
                slot["human"] += float(item["human_tag_count"])
            if isinstance(item.get("llm_tag_count"), (int, float)):
                slot["llm"] += float(item["llm_tag_count"])
            if isinstance(item.get("intersection_count"), (int, float)):
                slot["tp"] += float(item["intersection_count"])

    out: Dict[str, float] = {}
    for type_name, slot in sums.items():
        human = slot["human"]
        llm = slot["llm"]
        tp_val = slot["tp"]
        denom = human + llm
        out[type_name] = 0.0 if denom <= 0.0 else (2.0 * tp_val / denom)
    return out


def _extract_tag_frequencies(tag_payload: Mapping[str, Any]) -> tuple[Dict[str, int], Dict[str, int]]:
    human_counts: Dict[str, int] = {}
    llm_counts: Dict[str, int] = {}
    rows = tag_payload.get("per_issue", []) if isinstance(tag_payload, Mapping) else []
    if not isinstance(rows, list):
        return human_counts, llm_counts

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        items = row.get("per_matching_type", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            human_tags = item.get("human_tags", [])
            llm_tags = item.get("llm_tags", [])
            if isinstance(human_tags, list):
                for tag in human_tags:
                    if isinstance(tag, str) and tag:
                        human_counts[tag] = human_counts.get(tag, 0) + 1
            if isinstance(llm_tags, list):
                for tag in llm_tags:
                    if isinstance(tag, str) and tag:
                        llm_counts[tag] = llm_counts.get(tag, 0) + 1

    return human_counts, llm_counts


def _load_tag_frequency_from_issue_payloads(base_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not base_dir.exists():
        return counts

    for path in sorted(base_dir.glob("issue_*.json")):
        payload = _read_json_if_exists(path)
        if not isinstance(payload, Mapping):
            continue
        issue = payload.get("issue", {})
        if not isinstance(issue, Mapping):
            continue
        comments = issue.get("comments", [])
        if not isinstance(comments, list):
            continue
        for comment in comments:
            if not isinstance(comment, Mapping):
                continue
            artifacts = comment.get("artifacts", [])
            if not isinstance(artifacts, list):
                continue
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    continue
                tags = artifact.get("tags", [])
                if not isinstance(tags, list):
                    continue
                for tag in tags:
                    if isinstance(tag, str) and tag:
                        counts[tag] = counts.get(tag, 0) + 1
    return counts


def _extract_summary_metric_by_type(
    summary_payload: Mapping[str, Any],
    score_key: str,
    value_key: str,
) -> Dict[str, float]:
    lists: Dict[str, List[float]] = {}
    rows = summary_payload.get("per_issue", []) if isinstance(summary_payload, Mapping) else []
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        items = row.get("per_matching_type", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            type_name = item.get("type")
            scores = item.get("scores")
            if not isinstance(type_name, str) or not isinstance(scores, Mapping):
                continue
            node = scores.get(score_key)
            if not isinstance(node, Mapping):
                continue
            value = node.get(value_key)
            if isinstance(value, (int, float)):
                lists.setdefault(type_name, []).append(float(value))

    out: Dict[str, float] = {}
    for type_name, values in lists.items():
        if values:
            out[type_name] = sum(values) / float(len(values))
    return out


def _spread_types_for_spider(type_names: Sequence[str], score_by_type: Mapping[str, float]) -> List[str]:
    ordered = sorted(type_names, key=lambda name: (score_by_type.get(name, 0.0), name))
    if len(ordered) <= 6:
        return ordered

    bucket_count = 5 if len(ordered) >= 20 else 4
    chunk_size = int(math.ceil(len(ordered) / float(bucket_count)))
    buckets: List[List[str]] = []
    for i in range(0, len(ordered), chunk_size):
        buckets.append(ordered[i : i + chunk_size])

    woven: List[str] = []
    max_len = max(len(bucket) for bucket in buckets)
    for i in range(max_len):
        row = [bucket[i] for bucket in buckets if i < len(bucket)]
        if i % 2 == 1:
            row.reverse()
        woven.extend(row)

    seen = set()
    out: List[str] = []
    for type_name in woven:
        if type_name in seen:
            continue
        seen.add(type_name)
        out.append(type_name)
    return out


def _display_type_name(type_name: str) -> str:
    return type_name.replace("_", " ")


def _safe_path(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _model_dir_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__") or "unknown-model"


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

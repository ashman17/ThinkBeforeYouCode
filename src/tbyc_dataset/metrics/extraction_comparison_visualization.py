from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from tbyc_dataset.models import RepositoryRef


COMPONENTS = (
    ("type_f1", "Type F1"),
    ("tag_f1", "Tag F1"),
    ("metadata_f1", "Metadata F1"),
    ("metadata_soft_f1", "Metadata Soft F1"),
    ("summary_bertscore_f1", "Summary BERTScore F1"),
    ("summary_codebert", "Summary CodeBERT"),
    ("summary_bleurt", "Summary BLEURT"),
)


def generate_extraction_comparison_visualizations(
    *,
    output_root: str,
    graphs_root: Optional[str] = None,
    repo: Optional[str] = None,
) -> Dict[str, Any]:
    plt, np = _load_plotting_dependencies()
    data_root = Path(output_root)
    metrics_root = data_root / "metrics_regex_comparison"
    if not metrics_root.exists():
        raise FileNotFoundError(f"No regex comparison metrics directory found at {metrics_root}")

    target_root = Path(graphs_root) if graphs_root else (data_root / "graphs_regex_comparison")
    target_root.mkdir(parents=True, exist_ok=True)
    repo_filter = RepositoryRef.parse(repo) if repo else None
    bundles = _load_bundles(metrics_root, repo_filter=repo_filter)
    if not bundles:
        raise FileNotFoundError("No extraction comparison bundles found for the provided filters.")

    files: List[str] = []
    for bundle in bundles:
        files.extend(_plot_repo_component_bar(target_root, bundle, plt, np))
        files.extend(_plot_repo_issue_boxplot(target_root, bundle, plt, np))
        files.extend(_plot_repo_per_type_bar(target_root, bundle, plt, np))
        files.extend(_plot_repo_tag_frequency(target_root, bundle, plt, np))

    aggregate_bundle = _aggregate_bundles(bundles)
    files.extend(_plot_global_overall_bar(target_root, aggregate_bundle, plt, np))
    files.extend(_plot_global_issue_boxplot(target_root, aggregate_bundle, plt, np))
    files.extend(_plot_global_tag_frequency(target_root, aggregate_bundle, plt, np))

    if len(bundles) > 1:
        files.extend(_plot_global_heatmap(target_root, bundles, plt, np))

    manifest = {
        "output_root": str(data_root),
        "graphs_root": str(target_root),
        "repo_filter": repo,
        "repository_count": len(bundles),
        "aggregate_issue_count": int(aggregate_bundle.get("issue_count", 0)),
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
            "Visualization dependencies unavailable. Install matplotlib and numpy to use visualize-extraction-comparison."
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


def _load_bundles(metrics_root: Path, *, repo_filter: Optional[RepositoryRef]) -> List[Dict[str, Any]]:
    bundles: List[Dict[str, Any]] = []
    for repo_dir in sorted(path for path in metrics_root.iterdir() if path.is_dir()):
        repo_ref = _repo_ref_from_fs_slug(repo_dir.name)
        if repo_ref is None:
            continue
        if repo_filter and repo_ref != repo_filter:
            continue
        path = repo_dir / "comparison.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        bundles.append(payload)
    return bundles


def _plot_repo_component_bar(target_root: Path, bundle: Mapping[str, Any], plt, np) -> List[str]:
    repo_slug = str(bundle.get("repository", "unknown/repo"))
    macro = bundle.get("macro_average", {})
    values = [
        float(macro.get("type", {}).get("f1", 0.0)),
        float(macro.get("tag", {}).get("f1", 0.0)),
        float(macro.get("metadata", {}).get("f1", 0.0)),
        float(macro.get("metadata", {}).get("soft_f1", 0.0)),
        float(macro.get("summary", {}).get("bertscore", {}).get("f1", 0.0)),
        float(macro.get("summary", {}).get("codebert", {}).get("cosine", 0.0)),
        float(macro.get("summary", {}).get("bleurt", {}).get("score", 0.0)),
    ]
    labels = [label for _, label in COMPONENTS]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(np.arange(len(labels)), values, color=["#29524A", "#94A187", "#E9BC73", "#C17C74", "#5B8E7D", "#3E5C76", "#D66853"])
    ax.set_title(f"LLM vs Regex Extraction Metrics ({repo_slug})")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    out = target_root / f"repo_{repo_slug.replace('/', '__')}_comparison_overall_bar.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {"graph": "repo_comparison_overall_bar", "repository": repo_slug, "labels": labels, "values": values},
    )


def _plot_repo_issue_boxplot(target_root: Path, bundle: Mapping[str, Any], plt, np) -> List[str]:
    repo_slug = str(bundle.get("repository", "unknown/repo"))
    per_issue = bundle.get("per_issue", [])
    if not per_issue:
        return []
    series = [
        [float(item.get("type", {}).get("f1", 0.0)) for item in per_issue],
        [float(item.get("tag", {}).get("f1", 0.0)) for item in per_issue],
        [float(item.get("metadata", {}).get("f1", 0.0)) for item in per_issue],
        [float(item.get("summary", {}).get("bertscore", {}).get("f1", 0.0)) for item in per_issue],
    ]
    labels = ["Type F1", "Tag F1", "Metadata F1", "Summary F1"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(series, labels=labels, patch_artist=True)
    ax.set_title(f"Per-Issue Metric Distribution ({repo_slug})")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    out = target_root / f"repo_{repo_slug.replace('/', '__')}_comparison_issue_boxplot.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {"graph": "repo_comparison_issue_boxplot", "repository": repo_slug, "labels": labels, "series": series},
    )


def _plot_repo_per_type_bar(target_root: Path, bundle: Mapping[str, Any], plt, np) -> List[str]:
    repo_slug = str(bundle.get("repository", "unknown/repo"))
    per_type = bundle.get("per_type", [])
    if not per_type:
        return []
    top = sorted(per_type, key=lambda item: float(item.get("type_metrics", {}).get("f1", 0.0)), reverse=True)[:12]
    labels = [str(item.get("type", "unknown")) for item in top]
    type_vals = [float(item.get("type_metrics", {}).get("f1", 0.0)) for item in top]
    tag_vals = [float(item.get("tag_metrics", {}).get("f1", 0.0)) for item in top]
    meta_vals = [float(item.get("metadata_metrics", {}).get("f1", 0.0)) for item in top]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 5.8))
    ax.bar(x - width, type_vals, width=width, label="Type F1", color="#29524A")
    ax.bar(x, tag_vals, width=width, label="Tag F1", color="#E9BC73")
    ax.bar(x + width, meta_vals, width=width, label="Metadata F1", color="#C17C74")
    ax.set_title(f"Per-Type Agreement ({repo_slug})")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    out = target_root / f"repo_{repo_slug.replace('/', '__')}_comparison_per_type_bar.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "repo_comparison_per_type_bar",
            "repository": repo_slug,
            "type_labels": labels,
            "type_f1": type_vals,
            "tag_f1": tag_vals,
            "metadata_f1": meta_vals,
        },
    )


def _plot_repo_tag_frequency(target_root: Path, bundle: Mapping[str, Any], plt, np) -> List[str]:
    repo_slug = str(bundle.get("repository", "unknown/repo"))
    llm = bundle.get("tag_frequency", {}).get("llm", {})
    regex = bundle.get("tag_frequency", {}).get("regex", {})
    tags = sorted(set(llm.keys()) | set(regex.keys()), key=lambda tag: -(int(llm.get(tag, 0)) + int(regex.get(tag, 0))))[:15]
    if not tags:
        return []
    llm_vals = [int(llm.get(tag, 0)) for tag in tags]
    regex_vals = [int(regex.get(tag, 0)) for tag in tags]
    x = np.arange(len(tags))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(tags) * 0.8), 5.8))
    ax.bar(x - width / 2.0, regex_vals, width=width, label="Regex", color="#94A187")
    ax.bar(x + width / 2.0, llm_vals, width=width, label="LLM", color="#3E5C76")
    ax.set_title(f"Tag Frequency Comparison ({repo_slug})")
    ax.set_ylabel("Mentions")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    out = target_root / f"repo_{repo_slug.replace('/', '__')}_comparison_tag_frequency.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "repo_comparison_tag_frequency",
            "repository": repo_slug,
            "tags": tags,
            "regex_values": regex_vals,
            "llm_values": llm_vals,
        },
    )


def _plot_global_heatmap(target_root: Path, bundles: List[Mapping[str, Any]], plt, np) -> List[str]:
    rows = []
    labels = []
    for bundle in bundles:
        macro = bundle.get("macro_average", {})
        rows.append(
            [
                float(macro.get("type", {}).get("f1", 0.0)),
                float(macro.get("tag", {}).get("f1", 0.0)),
                float(macro.get("metadata", {}).get("f1", 0.0)),
                float(macro.get("summary", {}).get("bertscore", {}).get("f1", 0.0)),
            ]
        )
        labels.append(str(bundle.get("repository", "unknown/repo")))
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.8)))
    image = ax.imshow(np.array(rows), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("LLM vs Regex Comparison Heatmap")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Type F1", "Tag F1", "Metadata F1", "Summary F1"], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    out = target_root / "global_regex_comparison_heatmap.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {"graph": "global_regex_comparison_heatmap", "repositories": labels, "matrix": rows},
    )


def _aggregate_bundles(bundles: List[Mapping[str, Any]]) -> Dict[str, Any]:
    per_issue: List[Mapping[str, Any]] = []
    llm_tags_sum: Dict[str, int] = {}
    regex_tags_sum: Dict[str, int] = {}
    llm_artifact_count = 0
    regex_artifact_count = 0
    llm_comment_count = 0
    regex_comment_count = 0
    repo_count = max(1, len(bundles))

    for bundle in bundles:
        per_issue.extend(bundle.get("per_issue", []))
        freq = bundle.get("tag_frequency", {})
        _merge_counts(llm_tags_sum, freq.get("llm", {}))
        _merge_counts(regex_tags_sum, freq.get("regex", {}))
        counts = bundle.get("overall", {}).get("counts", {})
        llm_artifact_count += int(counts.get("llm_artifact_count", 0))
        regex_artifact_count += int(counts.get("regex_artifact_count", 0))
        llm_comment_count += int(counts.get("llm_comment_count", 0))
        regex_comment_count += int(counts.get("regex_comment_count", 0))

    issue_count = len(per_issue)
    macro = {
        "type": {"f1": _mean(float(item.get("type", {}).get("f1", 0.0)) for item in per_issue)},
        "tag": {"f1": _mean(float(item.get("tag", {}).get("f1", 0.0)) for item in per_issue)},
        "metadata": {
            "f1": _mean(float(item.get("metadata", {}).get("f1", 0.0)) for item in per_issue),
            "soft_f1": _mean(float(item.get("metadata", {}).get("soft_f1", 0.0)) for item in per_issue),
        },
        "summary": {
            "bertscore": {"f1": _mean(float(item.get("summary", {}).get("bertscore", {}).get("f1", 0.0)) for item in per_issue)},
            "codebert": {"cosine": _mean(float(item.get("summary", {}).get("codebert", {}).get("cosine", 0.0)) for item in per_issue)},
            "bleurt": {"score": _mean(float(item.get("summary", {}).get("bleurt", {}).get("score", 0.0)) for item in per_issue)},
        },
    }
    llm_tags_avg = {key: float(value) / float(repo_count) for key, value in llm_tags_sum.items()}
    regex_tags_avg = {key: float(value) / float(repo_count) for key, value in regex_tags_sum.items()}
    return {
        "repository": "ALL_REPOS",
        "repository_count": repo_count,
        "issue_count": issue_count,
        "macro_average": macro,
        "per_issue": per_issue,
        "overall": {
            "counts": {
                "llm_artifact_count": llm_artifact_count,
                "regex_artifact_count": regex_artifact_count,
                "llm_comment_count": llm_comment_count,
                "regex_comment_count": regex_comment_count,
                "llm_artifact_count_avg_per_repo": float(llm_artifact_count) / float(repo_count),
                "regex_artifact_count_avg_per_repo": float(regex_artifact_count) / float(repo_count),
                "llm_comment_count_avg_per_repo": float(llm_comment_count) / float(repo_count),
                "regex_comment_count_avg_per_repo": float(regex_comment_count) / float(repo_count),
            }
        },
        "tag_frequency": {
            "llm": llm_tags_sum,
            "regex": regex_tags_sum,
        },
        "tag_frequency_avg_per_repo": {
            "llm": llm_tags_avg,
            "regex": regex_tags_avg,
        },
    }


def _plot_global_overall_bar(target_root: Path, bundle: Mapping[str, Any], plt, np) -> List[str]:
    macro = bundle.get("macro_average", {})
    values = [
        float(macro.get("type", {}).get("f1", 0.0)),
        float(macro.get("tag", {}).get("f1", 0.0)),
        float(macro.get("metadata", {}).get("f1", 0.0)),
        float(macro.get("metadata", {}).get("soft_f1", 0.0)),
        float(macro.get("summary", {}).get("bertscore", {}).get("f1", 0.0)),
        float(macro.get("summary", {}).get("codebert", {}).get("cosine", 0.0)),
        float(macro.get("summary", {}).get("bleurt", {}).get("score", 0.0)),
    ]
    labels = [label for _, label in COMPONENTS]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(np.arange(len(labels)), values, color=["#1B4332", "#40916C", "#D4A373", "#BC6C25", "#577590", "#277DA1", "#E76F51"])
    ax.set_title("Overall LLM vs Regex Extraction Metrics Across All Repos")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    out = target_root / "global_regex_comparison_overall_bar.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {"graph": "global_regex_comparison_overall_bar", "labels": labels, "values": values},
    )


def _plot_global_issue_boxplot(target_root: Path, bundle: Mapping[str, Any], plt, np) -> List[str]:
    per_issue = bundle.get("per_issue", [])
    if not per_issue:
        return []
    series = [
        [float(item.get("type", {}).get("f1", 0.0)) for item in per_issue],
        [float(item.get("tag", {}).get("f1", 0.0)) for item in per_issue],
        [float(item.get("metadata", {}).get("f1", 0.0)) for item in per_issue],
        [float(item.get("summary", {}).get("bertscore", {}).get("f1", 0.0)) for item in per_issue],
    ]
    labels = ["Type F1", "Tag F1", "Metadata F1", "Summary F1"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(series, labels=labels, patch_artist=True)
    ax.set_title("Per-Issue Metric Distribution Across All Repos")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    out = target_root / "global_regex_comparison_issue_boxplot.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {"graph": "global_regex_comparison_issue_boxplot", "labels": labels, "series": series},
    )


def _plot_global_tag_frequency(target_root: Path, bundle: Mapping[str, Any], plt, np) -> List[str]:
    llm = bundle.get("tag_frequency_avg_per_repo", {}).get("llm", {})
    regex = bundle.get("tag_frequency_avg_per_repo", {}).get("regex", {})
    tags = sorted(
        set(llm.keys()) | set(regex.keys()),
        key=lambda tag: -(float(llm.get(tag, 0.0)) + float(regex.get(tag, 0.0))),
    )[:20]
    if not tags:
        return []
    llm_vals = [float(llm.get(tag, 0.0)) for tag in tags]
    regex_vals = [float(regex.get(tag, 0.0)) for tag in tags]
    x = np.arange(len(tags))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(13, len(tags) * 0.75), 6))
    ax.bar(x - width / 2.0, regex_vals, width=width, label="Regex", color="#95D5B2")
    ax.bar(x + width / 2.0, llm_vals, width=width, label="LLM", color="#1D3557")
    ax.set_title("Global Average Tag Frequency Per Repo")
    ax.set_ylabel("Average mentions per repo")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    out = target_root / "global_regex_comparison_tag_frequency.png"
    return _finalize_graph(
        fig,
        out,
        plt,
        {
            "graph": "global_regex_comparison_tag_frequency",
            "tags": tags,
            "regex_values": regex_vals,
            "llm_values": llm_vals,
        },
    )


def _merge_counts(dst: Dict[str, int], src: Mapping[str, Any]) -> None:
    for key, value in src.items():
        dst[str(key)] = dst.get(str(key), 0) + int(value)


def _mean(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def _repo_ref_from_fs_slug(value: str) -> Optional[RepositoryRef]:
    if "__" not in value:
        return None
    owner, name = value.split("__", 1)
    if not owner or not name:
        return None
    return RepositoryRef(owner=owner, name=name)

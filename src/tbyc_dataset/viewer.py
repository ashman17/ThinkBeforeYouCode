from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import ensure_directory, read_json, viewer_index_path


COMPONENT_LABELS: Dict[str, str] = {
    "global_points": "Global Points",
    "overall_composite": "Overall Composite",
    "type_f1": "Type F1",
    "metadata_f1": "Metadata F1",
    "metadata_soft_f1": "Metadata Soft F1",
    "tag_f1": "Tag F1",
    "summary_bleurt": "Summary BLEURT",
    "summary_bertscore_f1": "Summary BERTScore F1",
    "summary_codebert": "Summary CodeBERT",
    "cost_per_issue": "Cost / Issue",
}

LEADERBOARD_COMPONENT_KEYS: Sequence[str] = (
    "type_f1",
    "metadata_f1",
    "metadata_soft_f1",
    "tag_f1",
    "summary_bleurt",
    "summary_bertscore_f1",
    "summary_codebert",
)

PAID_MODEL_COSTS: Dict[str, Dict[str, Any]] = {
    "us.anthropic.claude-opus-4-6-v1": {"total_cost": 2.41, "request_count": 44},
    "us.anthropic.claude-sonnet-4-6": {"total_cost": 1.47, "request_count": 44},
    "gpt-5.4": {"total_cost": 1.31, "request_count": 44},
    "gpt-5": {"total_cost": 1.54, "request_count": 20},
    "gemini-2.5-flash": {"total_cost": 0.41, "request_count": 44},
    "gpt-4.1-mini": {"total_cost": 0.13, "request_count": 44},
    "gemini-3.1-flash-lite-preview": {"total_cost": 0.10, "request_count": 44},
}


def build_processed_viewer(output_root: Path | str) -> Dict[str, Any]:
    root = Path(output_root)
    output_path = viewer_index_path(root)
    ensure_directory(output_path.parent)

    payload = {
        "issues": _collect_issue_payloads(root),
        "leaderboard": _build_leaderboard_payload(root),
        "metric_options": [{"key": key, "label": label} for key, label in COMPONENT_LABELS.items()],
    }
    output_path.write_text(_render_viewer_html(payload), encoding="utf-8")
    return {
        "viewer_path": str(output_path),
        "issue_count": len(payload["issues"]),
        "leaderboard_model_count": len(payload["leaderboard"].get("rows", [])),
    }


def _collect_issue_payloads(root: Path) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    raw_root = root / "raw"
    extraction_root = root / "extractions"
    derived_root = root / "derived"
    if not raw_root.exists():
        return issues

    for repo_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        repo_ref = _repo_ref_from_fs_slug(repo_dir.name)
        if repo_ref is None:
            continue
        issue_dir = repo_dir / "issues"
        extraction_repo_dir = extraction_root / repo_dir.name
        if not issue_dir.exists() or not extraction_repo_dir.exists():
            continue

        for raw_issue_path in sorted(issue_dir.glob("issue_*.json")):
            raw_payload = read_json(raw_issue_path)
            issue_number = _extract_issue_number(raw_payload, raw_issue_path)
            if issue_number is None:
                continue
            human_extraction_path = extraction_repo_dir / raw_issue_path.name
            if not human_extraction_path.exists():
                continue

            derived_payloads = _collect_derived_payloads(derived_root, repo_dir.name, raw_issue_path.name)
            issues.append(
                {
                    "id": f"{repo_ref.fs_slug}#{issue_number}",
                    "repository": repo_ref.slug,
                    "issue_number": issue_number,
                    "title": str(raw_payload.get("title", f"Issue {issue_number}")),
                    "url": str(raw_payload.get("url", "")),
                    "raw_issue": _simplify_raw_issue(raw_payload),
                    "human_extraction": read_json(human_extraction_path),
                    "derived_extractions": derived_payloads,
                    "available_models": sorted(derived_payloads.keys()),
                }
            )

    issues.sort(key=lambda item: (item["repository"], int(item["issue_number"])))
    return issues


def _collect_derived_payloads(derived_root: Path, repo_fs_slug: str, issue_filename: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not derived_root.exists():
        return out
    for model_dir in sorted(path for path in derived_root.iterdir() if path.is_dir()):
        payload_path = model_dir / repo_fs_slug / issue_filename
        if payload_path.exists():
            out[_display_model_id(model_dir.name)] = read_json(payload_path)
    return out


def _simplify_raw_issue(raw_payload: Mapping[str, Any]) -> Dict[str, Any]:
    comments_out: List[Dict[str, str]] = []
    for comment in raw_payload.get("comments", []) if isinstance(raw_payload.get("comments"), list) else []:
        if not isinstance(comment, Mapping):
            continue
        comments_out.append(
            {
                "author": _comment_author_name(comment.get("author")),
                "body": str(comment.get("body", "") or ""),
                "url": str(comment.get("url", "") or ""),
            }
        )

    return {
        "title": str(raw_payload.get("title", "") or ""),
        "url": str(raw_payload.get("url", "") or ""),
        "body": str(raw_payload.get("body", "") or ""),
        "state": str(raw_payload.get("state", "") or ""),
        "comments": comments_out,
    }


def _build_leaderboard_payload(root: Path) -> Dict[str, Any]:
    metrics_root = root / "metrics"
    if not metrics_root.exists():
        return {"rows": [], "cost_note": ""}

    rows: List[Dict[str, Any]] = []
    for model_dir in sorted(path for path in metrics_root.iterdir() if path.is_dir()):
        model_id = _display_model_id(model_dir.name)
        repo_scores: List[Dict[str, float]] = []
        for repo_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            components = _load_leaderboard_components(repo_dir)
            if components:
                repo_scores.append(components)
        if not repo_scores:
            continue

        metric_values = _mean_components(repo_scores)
        overall_composite = _safe_mean([metric_values.get(key, 0.0) for key in LEADERBOARD_COMPONENT_KEYS])
        cost_info = _cost_info_for_model(model_id)
        rows.append(
            {
                "model_id": model_id,
                "display_name": model_id,
                "provider": _provider_name(model_id),
                "logo_path": _provider_logo_path(_provider_key(model_id)),
                "repositories_scored": len(repo_scores),
                "metrics": {
                    **metric_values,
                    "overall_composite": overall_composite,
                    "global_points": _global_points(overall_composite),
                    "cost_per_issue": float(cost_info["cost_per_issue"]),
                },
                "cost": cost_info,
            }
        )

    rows.sort(key=lambda item: (-float(item["metrics"].get("global_points", 0.0)), item["display_name"]))
    return {
        "rows": rows,
        "cost_note": "Models without provided API costs are shown as $0.0000/issue* and marked as local models.",
    }


def _load_leaderboard_components(repo_metrics_dir: Path) -> Dict[str, float]:
    components: Dict[str, float] = {}
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
        base = ("overall", "all_issues_macro_with_unmatched_penalty")
        _put_float(components, "summary_codebert", _get_path(payload, base + ("codebert", "cosine")))
        _put_float(components, "summary_bertscore_f1", _get_path(payload, base + ("bertscore", "f1")))
        _put_float(components, "summary_bleurt", _get_path(payload, base + ("bleurt", "score")))
    return components


def _mean_components(repo_scores: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in LEADERBOARD_COMPONENT_KEYS:
        out[key] = _safe_mean([float(item[key]) for item in repo_scores if key in item])
    return out


def _cost_info_for_model(model_id: str) -> Dict[str, Any]:
    info = PAID_MODEL_COSTS.get(model_id)
    if info is None:
        return {
            "cost_per_issue": 0.0,
            "display_per_issue": "$0.0000/issue*",
            "is_local": True,
        }
    total_cost = float(info["total_cost"])
    request_count = int(info["request_count"])
    per_issue = total_cost / float(request_count) if request_count > 0 else 0.0
    return {
        "cost_per_issue": per_issue,
        "display_per_issue": f"${per_issue:.4f}/issue",
        "is_local": False,
    }


def _global_points(overall_composite: float, points_max: int = 100, points_step: int = 1) -> int:
    raw_points = float(overall_composite) * float(max(1, points_max))
    quantized = int(round(raw_points / float(max(1, points_step))) * max(1, points_step))
    return max(0, min(max(1, points_max), quantized))


def _display_model_id(model_dir_name: str) -> str:
    if model_dir_name == "openai__gpt-5":
        return "gpt-5"
    if model_dir_name == "gemini__gemini-3.1-flash-lite-preview":
        return "gemini-3.1-flash-lite-preview"
    return model_dir_name


def _provider_key(model_id: str) -> str:
    lowered = model_id.lower()
    if "anthropic" in lowered or "claude" in lowered:
        return "anthropic"
    if "gemini" in lowered:
        return "google"
    if "gpt" in lowered or "openai" in lowered:
        return "openai"
    if "qwen" in lowered:
        return "alibaba"
    if "deepseek" in lowered:
        return "deepseek"
    return "local"


def _provider_name(model_id: str) -> str:
    return {
        "anthropic": "Anthropic",
        "google": "Google",
        "openai": "OpenAI",
        "alibaba": "Alibaba",
        "deepseek": "DeepSeek",
        "local": "Local",
    }.get(_provider_key(model_id), "Local")


def _provider_logo_path(provider_key: str) -> str:
    return {
        "openai": "openai.svg",
        "anthropic": "anthropic.svg",
        "google": "gemini.webp",
        "alibaba": "qwen.png",
        "deepseek": "deepseek.webp",
        "local": "qwen.png",
    }.get(provider_key, "qwen.png")


def _render_viewer_html(payload: Mapping[str, Any]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    payload_json = payload_json.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TBYC Viewer</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255,255,255,0.86);
      --panel-strong: rgba(255,255,255,0.95);
      --border: rgba(46,44,39,0.12);
      --text: #1c1917;
      --muted: #5b5249;
      --forest: #1f6a5f;
      --shadow: 0 18px 46px rgba(37,31,21,0.14);
      --shadow-soft: 0 10px 24px rgba(37,31,21,0.09);
      --radius-xl: 24px;
      --radius-lg: 18px;
      --mono: "SFMono-Regular","Menlo","Monaco",monospace;
      --sans: "Avenir Next","Segoe UI","Helvetica Neue",sans-serif;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      min-height: 100%;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(245,158,11,0.18), transparent 26%),
        radial-gradient(circle at top right, rgba(31,106,95,0.14), transparent 24%),
        linear-gradient(180deg, #faf7f1 0%, #f4efe6 48%, #ece2ce 100%);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(28,25,23,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(28,25,23,0.03) 1px, transparent 1px);
      background-size: 36px 36px;
      pointer-events: none;
      mask-image: linear-gradient(180deg, rgba(0,0,0,0.45), rgba(0,0,0,0.12));
    }
    .shell {
      width: min(1440px, calc(100vw - 32px));
      margin: 20px auto 44px;
      position: relative;
      z-index: 1;
    }
    .hero {
      padding: 18px 22px;
      border-radius: 30px;
      background: linear-gradient(135deg, rgba(17,24,39,0.97), rgba(35,39,31,0.90) 55%, rgba(180,83,9,0.90));
      color: #fff9f1;
      box-shadow: var(--shadow);
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(22px, 2.3vw, 32px);
      line-height: 1.15;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .view-switch {
      display: inline-flex;
      gap: 10px;
      margin-top: 16px;
      padding: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
    }
    .view-switch button {
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      background: transparent;
      color: rgba(255,249,241,0.84);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    .view-switch button.active {
      background: rgba(255,255,255,0.96);
      color: #111827;
    }
    .view-panel { display: none; }
    .view-panel.active { display: block; }
    .layout {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      gap: 20px;
      margin-top: 20px;
    }
    .sidebar, .main-panel, .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow-soft);
      backdrop-filter: blur(16px);
    }
    .sidebar {
      padding: 20px;
      position: sticky;
      top: 18px;
      align-self: start;
    }
    .main-panel { padding: 20px; }
    .main-grid { display: grid; gap: 18px; }
    .card {
      padding: 20px;
      overflow: hidden;
      background: var(--panel-strong);
    }
    .section-title {
      margin: 0 0 12px;
      font-size: 13px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .field { margin-bottom: 14px; }
    .field label {
      display: block;
      margin-bottom: 8px;
      font-size: 13px;
      font-weight: 700;
      color: var(--muted);
    }
    select {
      width: 100%;
      border: 1px solid rgba(28,25,23,0.14);
      background: rgba(255,255,255,0.94);
      color: var(--text);
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      outline: none;
    }
    .quick-stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 18px;
    }
    .stat-card {
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.74));
      border: 1px solid rgba(28,25,23,0.08);
      padding: 14px;
    }
    .stat-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .stat-value {
      font-size: 24px;
      font-weight: 800;
      line-height: 1;
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 14px;
    }
    .card-header h2 {
      margin: 0;
      font-size: 24px;
    }
    .subtle {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    a.inline {
      color: var(--forest);
      text-decoration: none;
      font-weight: 700;
    }
    .scroll-panel {
      max-height: 420px;
      overflow: auto;
      scrollbar-width: thin;
    }
    .issue-body, .timeline {
      border-radius: var(--radius-lg);
      border: 1px solid rgba(28,25,23,0.08);
      background: rgba(244,239,230,0.62);
    }
    .issue-body {
      padding: 18px;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.6;
    }
    .timeline {
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(250,247,241,0.92));
    }
    .comment-block {
      padding: 16px 18px;
      white-space: pre-wrap;
      line-height: 1.6;
      word-break: break-word;
    }
    .comment-block + .comment-block {
      border-top: 1px dashed rgba(28,25,23,0.14);
    }
    .comment-author {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
      font-weight: 800;
    }
    .comment-author a {
      color: var(--forest);
      text-decoration: none;
      font-size: 13px;
    }
    .segmented-inline {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 12px;
    }
    .segmented-inline button {
      border: 1px solid rgba(28,25,23,0.10);
      background: rgba(255,255,255,0.92);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }
    .segmented-inline button.active {
      background: linear-gradient(180deg, #1f6a5f, #184f47);
      color: #fff;
      border-color: rgba(31,106,95,0.60);
    }
    .json-frame {
      border-radius: var(--radius-lg);
      background: #111827;
      color: #ecf1f8;
      border: 1px solid rgba(17,24,39,0.35);
      overflow: hidden;
    }
    .json-toolbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 12px 16px;
      background: rgba(255,255,255,0.06);
      border-bottom: 1px solid rgba(255,255,255,0.08);
      font-size: 13px;
      color: rgba(236,241,248,0.75);
    }
    pre {
      margin: 0;
      padding: 18px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 12.5px;
      line-height: 1.65;
      max-height: 420px;
      overflow: auto;
    }
    .leaderboard-shell { margin-top: 20px; }
    .leaderboard-wrap {
      background:
        radial-gradient(circle at top right, rgba(245,158,11,0.16), transparent 28%),
        linear-gradient(180deg, rgba(17,24,39,0.98), rgba(20,28,45,0.96));
      color: #f7f4ec;
    }
    .leaderboard-controls { margin-bottom: 18px; }
    .leaderboard-controls select { max-width: 280px; }
    .leaderboard-grid { display: grid; gap: 8px; }
    .lb-row {
      display: grid;
      grid-template-columns: 64px minmax(0, 1.8fr) minmax(0, 0.9fr) minmax(150px, 0.7fr);
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.08);
      background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
    }
    .lb-rank {
      display: grid;
      place-items: center;
      width: 48px;
      height: 48px;
      border-radius: 14px;
      font-weight: 900;
      font-size: 18px;
      color: #1c1917;
      background: linear-gradient(180deg, #fcd34d, #f59e0b);
    }
    .lb-model {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .lb-logo {
      width: 40px;
      height: 40px;
      border-radius: 12px;
      overflow: hidden;
      background: rgba(255,255,255,0.95);
      display: grid;
      place-items: center;
      flex: 0 0 auto;
    }
    .lb-logo img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .lb-model-name {
      font-size: 15px;
      font-weight: 800;
      margin: 0 0 3px;
      word-break: break-word;
    }
    .lb-provider {
      color: rgba(247,244,236,0.68);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
    }
    .lb-score-value {
      font-size: 27px;
      line-height: 1;
      font-weight: 900;
      margin-bottom: 5px;
    }
    .lb-score-label {
      color: rgba(247,244,236,0.68);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      margin-bottom: 8px;
    }
    .meter {
      height: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,0.10);
      overflow: hidden;
    }
    .meter > span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #fbbf24, #fb7185 45%, #22c55e 100%);
    }
    .lb-cost { text-align: right; }
    .lb-cost-total {
      font-size: 16px;
      line-height: 1.1;
      font-weight: 900;
    }
    .lb-cost-meta {
      font-size: 12px;
      color: rgba(247,244,236,0.72);
      margin-top: 4px;
    }
    .note {
      margin-top: 14px;
      color: rgba(247,244,236,0.70);
      font-size: 13px;
    }
    .empty {
      padding: 18px;
      border-radius: 18px;
      background: rgba(28,25,23,0.04);
      border: 1px dashed rgba(28,25,23,0.12);
      color: var(--muted);
    }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .lb-row { grid-template-columns: 64px 1fr; }
      .lb-score, .lb-cost { grid-column: 2; }
      .lb-cost { text-align: left; }
    }
    @media (max-width: 720px) {
      .shell { width: min(100vw - 16px, 100%); margin: 8px auto 28px; }
      .hero, .sidebar, .main-panel, .card { border-radius: 22px; }
      .hero { padding: 16px; }
      .hero h1 { white-space: normal; }
      .main-panel { padding: 14px; }
      .card { padding: 16px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Think Before You Code: Benchmarking Thinking and Reasoning in Code LLMs</h1>
      <div class="view-switch">
        <button id="explorerViewButton" class="active" type="button">Explorer</button>
        <button id="leaderboardViewButton" type="button">Leaderboard</button>
      </div>
    </section>

    <div id="explorerView" class="view-panel active">
      <div class="layout">
        <aside class="sidebar">
          <h2 class="section-title">Explorer</h2>
          <div class="field">
            <label for="repoSelect">Repository</label>
            <select id="repoSelect"></select>
          </div>
          <div class="field">
            <label for="issueSelect">Issue</label>
            <select id="issueSelect"></select>
          </div>
          <div class="quick-stats">
            <div class="stat-card">
              <div class="stat-label">Comments</div>
              <div class="stat-value" id="commentCount">0</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Artifacts</div>
              <div class="stat-value" id="artifactCount">0</div>
            </div>
          </div>
        </aside>

        <main class="main-panel">
          <div class="main-grid">
            <section class="card">
              <div class="card-header">
                <div>
                  <h2 id="issueTitle">Issue</h2>
                  <div class="subtle" id="issueMeta"></div>
                </div>
                <div><a class="inline" id="issueLink" href="#" target="_blank" rel="noopener noreferrer">Open on GitHub</a></div>
              </div>
              <div class="section-title">Raw Issue Body</div>
              <div class="issue-body scroll-panel" id="issueBody"></div>
              <div style="height:14px"></div>
              <div class="section-title">Discussion</div>
              <div class="timeline scroll-panel" id="issueComments"></div>
            </section>

            <section class="card">
              <div class="card-header">
                <div>
                  <h2>Extractions on Raw Issue</h2>
                  <div class="subtle">Human-side extraction payload rendered as clean, formatted JSON.</div>
                </div>
              </div>
              <div class="json-frame">
                <div class="json-toolbar">
                  <span>Source: <strong>data/extractions</strong></span>
                  <span id="humanExtractionMeta"></span>
                </div>
                <pre id="humanExtraction"></pre>
              </div>
            </section>

            <section class="card">
              <div class="card-header">
                <div>
                  <h2>Extractions on LLM Response</h2>
                  <div class="subtle">Choose a model below to inspect the derived extraction JSON for this issue.</div>
                </div>
              </div>
              <div class="segmented-inline" id="derivedModelButtons"></div>
              <div class="json-frame">
                <div class="json-toolbar">
                  <span>Source: <strong>data/derived/&lt;model&gt;</strong></span>
                  <span id="derivedExtractionMeta"></span>
                </div>
                <pre id="derivedExtraction"></pre>
              </div>
            </section>
          </div>
        </main>
      </div>
    </div>

    <div id="leaderboardView" class="view-panel leaderboard-shell">
      <section class="card leaderboard-wrap">
        <div class="card-header">
          <div>
            <h2>Leaderboard</h2>
            <div class="subtle">Compare models on benchmark metrics and cost per issue.</div>
          </div>
        </div>
        <div class="leaderboard-controls">
          <select id="metricSelect"></select>
        </div>
        <div class="leaderboard-grid" id="leaderboard"></div>
        <div class="note" id="leaderboardNote"></div>
      </section>
    </div>
  </div>

  <script id="viewer-data" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("viewer-data").textContent);
    const state = { repository: null, issueId: null, modelId: null, leaderboardMetric: "global_points", view: "explorer" };

    const repoSelect = document.getElementById("repoSelect");
    const issueSelect = document.getElementById("issueSelect");
    const metricSelect = document.getElementById("metricSelect");
    const derivedModelButtons = document.getElementById("derivedModelButtons");
    const explorerView = document.getElementById("explorerView");
    const leaderboardView = document.getElementById("leaderboardView");
    const explorerViewButton = document.getElementById("explorerViewButton");
    const leaderboardViewButton = document.getElementById("leaderboardViewButton");
    const issueTitle = document.getElementById("issueTitle");
    const issueMeta = document.getElementById("issueMeta");
    const issueLink = document.getElementById("issueLink");
    const issueBody = document.getElementById("issueBody");
    const issueComments = document.getElementById("issueComments");
    const humanExtraction = document.getElementById("humanExtraction");
    const humanExtractionMeta = document.getElementById("humanExtractionMeta");
    const derivedExtraction = document.getElementById("derivedExtraction");
    const derivedExtractionMeta = document.getElementById("derivedExtractionMeta");
    const leaderboard = document.getElementById("leaderboard");
    const leaderboardNote = document.getElementById("leaderboardNote");
    const commentCount = document.getElementById("commentCount");
    const artifactCount = document.getElementById("artifactCount");

    const issues = Array.isArray(DATA.issues) ? DATA.issues : [];
    const repoNames = [...new Set(issues.map((item) => item.repository))].sort();

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function formatMetricValue(key, value) {
      const num = Number(value ?? 0);
      if (key === "global_points") return String(Math.round(num));
      if (key === "cost_per_issue") return `$${num.toFixed(4)}/issue`;
      return num.toFixed(3);
    }

    function formatMeterPercent(key, value) {
      const num = Number(value ?? 0);
      if (key === "global_points") return Math.max(0, Math.min(100, num));
      if (key === "cost_per_issue") return Math.max(0, Math.min(100, 100 - (num * 1000)));
      return Math.max(0, Math.min(100, num * 100));
    }

    function setView(viewName) {
      state.view = viewName;
      explorerView.classList.toggle("active", viewName === "explorer");
      leaderboardView.classList.toggle("active", viewName === "leaderboard");
      explorerViewButton.classList.toggle("active", viewName === "explorer");
      leaderboardViewButton.classList.toggle("active", viewName === "leaderboard");
    }

    function issuesForRepo(repo) {
      return issues.filter((item) => item.repository === repo);
    }

    function currentIssue() {
      return issues.find((item) => item.id === state.issueId) || null;
    }

    function populateRepositories() {
      repoSelect.innerHTML = repoNames.map((repo) => `<option value="${escapeHtml(repo)}">${escapeHtml(repo)}</option>`).join("");
      state.repository = repoNames[0] || null;
    }

    function populateIssues() {
      const repoIssues = issuesForRepo(state.repository);
      issueSelect.innerHTML = repoIssues.map((item) => {
        const label = `#${item.issue_number}  ${item.title}`;
        return `<option value="${escapeHtml(item.id)}">${escapeHtml(label)}</option>`;
      }).join("");
      state.issueId = repoIssues[0] ? repoIssues[0].id : null;
    }

    function renderDerivedModelButtons() {
      const issue = currentIssue();
      const models = issue ? issue.available_models : [];
      if (!models.length) {
        state.modelId = null;
        derivedModelButtons.innerHTML = `<div class="empty">No derived model extraction available for this issue.</div>`;
        return;
      }
      if (!state.modelId || !models.includes(state.modelId)) state.modelId = models[0];
      derivedModelButtons.innerHTML = models.map((model) => `
        <button type="button" data-model="${escapeHtml(model)}" class="${model === state.modelId ? "active" : ""}">
          ${escapeHtml(model)}
        </button>
      `).join("");
      derivedModelButtons.querySelectorAll("button[data-model]").forEach((button) => {
        button.addEventListener("click", () => {
          state.modelId = button.getAttribute("data-model");
          renderDerivedModelButtons();
          renderIssue();
        });
      });
    }

    function renderIssue() {
      const issue = currentIssue();
      if (!issue) {
        issueTitle.textContent = "No issue selected";
        issueMeta.textContent = "";
        issueBody.textContent = "";
        issueComments.innerHTML = `<div class="empty">No issue data available.</div>`;
        humanExtraction.textContent = "";
        derivedExtraction.textContent = "";
        derivedModelButtons.innerHTML = "";
        return;
      }
      const raw = issue.raw_issue || {};
      issueTitle.textContent = `#${issue.issue_number} ${issue.title}`;
      issueMeta.textContent = `${issue.repository} • ${raw.state || "unknown state"} • ${Array.isArray(raw.comments) ? raw.comments.length : 0} comments`;
      issueLink.href = raw.url || issue.url || "#";
      issueBody.textContent = raw.body || "(empty issue body)";
      commentCount.textContent = String(Array.isArray(raw.comments) ? raw.comments.length : 0);
      artifactCount.textContent = String((((issue.human_extraction || {}).issue || {}).artifact_count) || 0);

      const comments = Array.isArray(raw.comments) ? raw.comments : [];
      issueComments.innerHTML = comments.length ? comments.map((comment) => `
        <div class="comment-block">
          <div class="comment-author">
            <span>${escapeHtml(comment.author || "unknown")}</span>
            ${comment.url ? `<a href="${escapeHtml(comment.url)}" target="_blank" rel="noopener noreferrer">permalink</a>` : ""}
          </div>
          <div>${escapeHtml(comment.body || "")}</div>
        </div>
      `).join("") : `<div class="empty">No comments recorded for this issue.</div>`;

      humanExtraction.textContent = JSON.stringify(issue.human_extraction || {}, null, 2);
      humanExtractionMeta.textContent = `${issue.repository} • #${issue.issue_number}`;

      const derived = (issue.derived_extractions || {})[state.modelId];
      derivedExtraction.textContent = JSON.stringify(derived || {}, null, 2);
      derivedExtractionMeta.textContent = state.modelId ? `${state.modelId} • #${issue.issue_number}` : "No derived extraction";
    }

    function populateMetricOptions() {
      const options = Array.isArray(DATA.metric_options) ? DATA.metric_options : [];
      metricSelect.innerHTML = options.map((item) => `<option value="${escapeHtml(item.key)}">${escapeHtml(item.label)}</option>`).join("");
      metricSelect.value = state.leaderboardMetric;
    }

    function renderLeaderboard() {
      const metricKey = state.leaderboardMetric;
      const metricLabel = (DATA.metric_options || []).find((item) => item.key === metricKey)?.label || metricKey;
      const rows = [...((DATA.leaderboard || {}).rows || [])];
      rows.sort((a, b) => {
        const left = Number((a.metrics || {})[metricKey] ?? 0);
        const right = Number((b.metrics || {})[metricKey] ?? 0);
        if (right !== left) return right - left;
        return String(a.display_name).localeCompare(String(b.display_name));
      });

      leaderboard.innerHTML = rows.map((row, index) => {
        const metricValue = Number((row.metrics || {})[metricKey] ?? 0);
        const width = formatMeterPercent(metricKey, metricValue);
        const cost = row.cost || {};
        return `
          <div class="lb-row">
            <div class="lb-rank">#${index + 1}</div>
            <div class="lb-model">
              <div class="lb-logo"><img src="${escapeHtml(row.logo_path || "qwen.png")}" alt="${escapeHtml(row.provider || "Model")} logo"></div>
              <div>
                <div class="lb-model-name">${escapeHtml(row.display_name || row.model_id || "model")}</div>
                <div class="lb-provider">${escapeHtml(row.provider || "Provider")} • ${escapeHtml(String(row.repositories_scored || 0))} repos scored</div>
              </div>
            </div>
            <div class="lb-score">
              <div class="lb-score-value">${escapeHtml(formatMetricValue(metricKey, metricValue))}</div>
              <div class="lb-score-label">${escapeHtml(metricLabel)}</div>
              <div class="meter"><span style="width:${width}%"></span></div>
            </div>
            <div class="lb-cost">
              <div class="lb-cost-total">${escapeHtml(cost.display_per_issue || "$0.0000/issue")}</div>
              <div class="lb-cost-meta">Cost per issue</div>
            </div>
          </div>
        `;
      }).join("");
      leaderboardNote.textContent = (DATA.leaderboard || {}).cost_note || "";
    }

    repoSelect.addEventListener("change", () => {
      state.repository = repoSelect.value;
      populateIssues();
      renderDerivedModelButtons();
      renderIssue();
    });
    issueSelect.addEventListener("change", () => {
      state.issueId = issueSelect.value;
      renderDerivedModelButtons();
      renderIssue();
    });
    metricSelect.addEventListener("change", () => {
      state.leaderboardMetric = metricSelect.value;
      renderLeaderboard();
    });
    explorerViewButton.addEventListener("click", () => setView("explorer"));
    leaderboardViewButton.addEventListener("click", () => setView("leaderboard"));

    populateRepositories();
    populateIssues();
    renderDerivedModelButtons();
    populateMetricOptions();
    renderIssue();
    renderLeaderboard();
    setView("explorer");
  </script>
</body>
</html>
"""
    return html.replace("__PAYLOAD_JSON__", payload_json)


def _extract_issue_number(payload: Mapping[str, Any], path: Path) -> Optional[int]:
    candidates = [
        payload.get("number"),
        payload.get("issue_number"),
        payload.get("issue", {}).get("number") if isinstance(payload.get("issue"), Mapping) else None,
        payload.get("issue", {}).get("issue_number") if isinstance(payload.get("issue"), Mapping) else None,
    ]
    for candidate in candidates:
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    stem = path.stem
    if stem.startswith("issue_"):
        try:
            return int(stem.split("_", 1)[1])
        except (TypeError, ValueError):
            return None
    return None


def _repo_ref_from_fs_slug(fs_slug: str) -> Optional[RepositoryRef]:
    if "__" not in fs_slug:
        return None
    owner, name = fs_slug.split("__", 1)
    if not owner or not name:
        return None
    return RepositoryRef(owner=owner, name=name)


def _comment_author_name(raw_author: Any) -> str:
    if isinstance(raw_author, Mapping):
        login = raw_author.get("login")
        if isinstance(login, str) and login.strip():
            return login.strip()
    if isinstance(raw_author, str) and raw_author.strip():
        return raw_author.strip()
    return "unknown"


def _put_float(target: Dict[str, float], key: str, value: Any) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if math.isnan(number) or math.isinf(number):
        return
    target[key] = number


def _get_path(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _safe_mean(values: Sequence[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / float(len(items))

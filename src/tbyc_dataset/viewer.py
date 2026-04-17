from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from tbyc_dataset.models import JSONDict
from tbyc_dataset.roles import display_role_from_association
from tbyc_dataset.storage import read_jsonl, viewer_index_path


def build_processed_viewer(output_root: Path) -> JSONDict:
    payload = build_viewer_payload(output_root)
    html = render_viewer_html(payload)
    output_path = viewer_index_path(output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return {
        "repository_count": len(payload["repositories"]),
        "issue_count": sum(repo["issue_count"] for repo in payload["repositories"]),
        "viewer_path": str(output_path),
    }


def build_viewer_payload(output_root: Path) -> JSONDict:
    processed_root = output_root / "processed"
    repositories: List[JSONDict] = []

    if processed_root.exists():
        for repo_dir in sorted(path for path in processed_root.iterdir() if path.is_dir()):
            curated_path = repo_dir / "curated.jsonl"
            if not curated_path.exists():
                continue
            records = read_jsonl(curated_path)
            if not records:
                continue
            repositories.append(build_repository_payload(repo_dir.name, records))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repositories": repositories,
    }


def build_repository_payload(fs_slug: str, records: Iterable[JSONDict]) -> JSONDict:
    rows = sorted(records, key=lambda record: record["issue_number"])
    repository = rows[0]["repository"]
    issues = [
        {
            **{
                "discussion_entries": build_discussion_entries(row.get("deliberation_thread", [])),
            },
            **{
                "issue_number": row["issue_number"],
                "issue_url": row["issue_url"],
                "title": row["input_vector"]["title"],
                "body": row["input_vector"].get("body") or "",
                "files": row.get("files", []),
                "comment_count": len(row.get("deliberation_thread", [])),
                "labels": row.get("taxonomic_metadata", {}).get("labels", []),
                "issue_state": row.get("taxonomic_metadata", {}).get("issue_state"),
                "created_at": row.get("taxonomic_metadata", {}).get("created_at"),
                "closed_at": row.get("taxonomic_metadata", {}).get("closed_at"),
            },
            "formatted_discussion": format_discussion_entries(
                build_discussion_entries(row.get("deliberation_thread", []))
            ),
        }
        for row in rows
    ]
    return {
        "repository": repository,
        "fs_slug": fs_slug,
        "issue_count": len(issues),
        "issues": issues,
    }


def build_discussion_entries(deliberation_thread: Iterable[JSONDict]) -> List[JSONDict]:
    entries: List[JSONDict] = []
    for comment in deliberation_thread:
        body = comment.get("body") or ""
        if not body.strip():
            continue
        author_id = comment.get("author_login") or "unknown"
        author_role = display_role_from_association(comment.get("author_association"))
        entries.append(
            {
                "author_id": author_id,
                "author_role": author_role,
                "content": body,
            }
        )
    return entries


def format_discussion_entries(entries: Iterable[JSONDict]) -> str:
    return "\n".join(
        f"{entry['author_id']} ({entry['author_role']}):{squash_whitespace(entry['content'])}"
        for entry in entries
    )


def squash_whitespace(text: str) -> str:
    return " ".join(text.split())


def render_viewer_html(payload: JSONDict) -> str:
    data_json = json.dumps(payload, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TBYC Processed Dataset Viewer</title>
  <style>
    :root {{
      --page: #f4efe7;
      --ink: #1d2430;
      --muted: #5f6876;
      --panel: rgba(255, 252, 247, 0.9);
      --line: rgba(29, 36, 48, 0.14);
      --accent: #a34c1f;
      --accent-soft: rgba(163, 76, 31, 0.12);
      --mono: "SFMono-Regular", "SF Mono", Menlo, Consolas, monospace;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(163, 76, 31, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(37, 86, 120, 0.10), transparent 25%),
        linear-gradient(180deg, #fbf7f0 0%, var(--page) 100%);
    }}

    .shell {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}

    .hero {{
      margin-bottom: 24px;
    }}

    h1 {{
      margin: 0;
      font-family: var(--serif);
      font-size: clamp(2rem, 3vw, 3rem);
      font-weight: 700;
      letter-spacing: -0.03em;
    }}

    .subhead {{
      margin-top: 10px;
      color: var(--muted);
      max-width: 70ch;
      line-height: 1.5;
    }}

    .controls,
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      backdrop-filter: blur(8px);
      box-shadow: 0 18px 40px rgba(38, 32, 24, 0.06);
    }}

    .controls {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      padding: 18px;
      margin-bottom: 20px;
    }}

    label {{
      display: block;
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    select {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
      font: inherit;
      color: var(--ink);
    }}

    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 20px;
    }}

    .chip {{
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.75);
      color: var(--muted);
      font-size: 0.9rem;
    }}

    .content {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      gap: 20px;
    }}

    .discussion-panel {{
      grid-column: 1 / -1;
    }}

    .panel {{
      padding: 20px;
    }}

    .eyebrow {{
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    h2 {{
      margin: 10px 0 12px;
      font-family: var(--serif);
      font-size: 1.6rem;
      line-height: 1.2;
    }}

    .meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 16px 0 20px;
    }}

    .meta-card {{
      padding: 12px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid var(--line);
    }}

    .meta-card strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }}

    .section-title {{
      margin: 18px 0 8px;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 700;
    }}

    .link {{
      color: var(--accent);
      text-decoration: none;
      word-break: break-all;
    }}

    .list {{
      margin: 0;
      padding-left: 18px;
    }}

    .list li {{
      margin: 0 0 6px;
      font-family: var(--mono);
      font-size: 0.92rem;
    }}

    .text-block {{
      margin: 0;
      padding: 14px;
      min-height: 160px;
      max-height: 70vh;
      overflow: auto;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(249,246,240,0.95));
      white-space: pre-wrap;
      font-family: var(--mono);
      font-size: 0.92rem;
      line-height: 1.5;
    }}

    .body-block {{
      min-height: 120px;
      font-family: var(--sans);
      background: rgba(255, 255, 255, 0.7);
    }}

    .discussion-block {{
      padding: 18px;
    }}

    .discussion-entry + .discussion-entry {{
      margin-top: 22px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
    }}

    .discussion-speaker {{
      margin: 0 0 8px;
      font-family: var(--sans);
      font-size: 1rem;
      font-weight: 800;
      color: var(--ink);
    }}

    .discussion-content {{
      margin: 0;
      white-space: pre-wrap;
      font-family: var(--mono);
      font-size: 0.92rem;
      line-height: 1.6;
      color: var(--ink);
    }}

    .empty {{
      padding: 18px;
      border-radius: 14px;
      background: var(--accent-soft);
      color: var(--muted);
    }}

    @media (max-width: 920px) {{
      .controls,
      .content,
      .meta {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <h1>Processed Dataset Viewer</h1>
      <p class="subhead">Browse normalized GitHub issue records by repository and issue. The viewer is generated from <code>data/processed/*/curated.jsonl</code> and includes the LLM-ready formatted discussion.</p>
    </div>

    <div class="controls">
      <div>
        <label for="repo-select">Repository</label>
        <select id="repo-select"></select>
      </div>
      <div>
        <label for="issue-select">Issue</label>
        <select id="issue-select"></select>
      </div>
    </div>

    <div class="stats" id="stats"></div>

    <div class="content" id="content" hidden>
      <section class="panel">
        <div class="eyebrow" id="repo-name"></div>
        <h2 id="issue-title"></h2>
        <a class="link" id="issue-url" target="_blank" rel="noreferrer"></a>
        <div class="meta">
          <div class="meta-card"><strong>Issue Number</strong><span id="issue-number"></span></div>
          <div class="meta-card"><strong>State</strong><span id="issue-state"></span></div>
          <div class="meta-card"><strong>Created</strong><span id="created-at"></span></div>
          <div class="meta-card"><strong>Closed</strong><span id="closed-at"></span></div>
        </div>

        <div class="section-title">Issue Body</div>
        <pre class="text-block body-block" id="issue-body"></pre>
      </section>

      <aside class="panel">
        <div class="section-title">Files</div>
        <ul class="list" id="files-list"></ul>

        <div class="section-title">Labels</div>
        <ul class="list" id="labels-list"></ul>

      </aside>

      <section class="panel discussion-panel">
        <div class="section-title">Discussion Thread</div>
        <div class="text-block discussion-block" id="discussion"></div>
      </section>
    </div>

    <div class="empty" id="empty-state" hidden>No processed repositories were found.</div>
  </div>

  <script id="viewer-data" type="application/json">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("viewer-data").textContent);
    const repoSelect = document.getElementById("repo-select");
    const issueSelect = document.getElementById("issue-select");
    const stats = document.getElementById("stats");
    const content = document.getElementById("content");
    const emptyState = document.getElementById("empty-state");

    const repositories = payload.repositories || [];

    function setList(node, items, emptyText) {{
      node.innerHTML = "";
      if (!items || items.length === 0) {{
        const li = document.createElement("li");
        li.textContent = emptyText;
        node.appendChild(li);
        return;
      }}
      items.forEach((item) => {{
        const li = document.createElement("li");
        li.textContent = item;
        node.appendChild(li);
      }});
    }}

    function renderStats(repo, issue) {{
      const chips = [
        `${{repositories.length}} repos`,
        `${{repo.issue_count}} issues in repo`,
        `${{issue.comment_count}} discussion entries`,
        `viewer generated ${{new Date(payload.generated_at).toLocaleString()}}`,
      ];
      stats.innerHTML = "";
      chips.forEach((text) => {{
        const span = document.createElement("span");
        span.className = "chip";
        span.textContent = text;
        stats.appendChild(span);
      }});
    }}

    function renderIssue(repoIndex, issueIndex) {{
      const repo = repositories[repoIndex];
      const issue = repo.issues[issueIndex];

      document.getElementById("repo-name").textContent = repo.repository;
      document.getElementById("issue-title").textContent = issue.title || `Issue #${{issue.issue_number}}`;
      document.getElementById("issue-number").textContent = `#${{issue.issue_number}}`;
      document.getElementById("issue-state").textContent = issue.issue_state || "unknown";
      document.getElementById("created-at").textContent = issue.created_at || "-";
      document.getElementById("closed-at").textContent = issue.closed_at || "-";
      document.getElementById("issue-body").textContent = issue.body || "";

      const issueUrl = document.getElementById("issue-url");
      issueUrl.href = issue.issue_url;
      issueUrl.textContent = issue.issue_url;

      setList(document.getElementById("files-list"), issue.files, "No explicit files");
      setList(document.getElementById("labels-list"), issue.labels, "No labels");
      renderDiscussion(issue.discussion_entries || []);
      renderStats(repo, issue);
      content.hidden = false;
    }}

    function renderDiscussion(entries) {{
      const discussion = document.getElementById("discussion");
      discussion.innerHTML = "";

      if (!entries.length) {{
        discussion.textContent = "No discussion entries";
        return;
      }}

      entries.forEach((entry) => {{
        const wrapper = document.createElement("article");
        wrapper.className = "discussion-entry";

        const speaker = document.createElement("p");
        speaker.className = "discussion-speaker";
        speaker.textContent = `${{entry.author_id}} (${{entry.author_role}})`;

        const content = document.createElement("pre");
        content.className = "discussion-content";
        content.textContent = entry.content;

        wrapper.appendChild(speaker);
        wrapper.appendChild(content);
        discussion.appendChild(wrapper);
      }});
    }}

    function populateIssueSelect(repoIndex) {{
      const repo = repositories[repoIndex];
      issueSelect.innerHTML = "";
      repo.issues.forEach((issue, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `#${{issue.issue_number}} - ${{issue.title}}`;
        issueSelect.appendChild(option);
      }});
      renderIssue(repoIndex, 0);
    }}

    function initialize() {{
      if (repositories.length === 0) {{
        emptyState.hidden = false;
        repoSelect.disabled = true;
        issueSelect.disabled = true;
        return;
      }}

      repositories.forEach((repo, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${{repo.repository}} (${{repo.issue_count}})`;
        repoSelect.appendChild(option);
      }});

      repoSelect.addEventListener("change", () => populateIssueSelect(Number(repoSelect.value)));
      issueSelect.addEventListener("change", () => renderIssue(Number(repoSelect.value), Number(issueSelect.value)));
      populateIssueSelect(0);
    }}

    initialize();
  </script>
</body>
</html>
"""

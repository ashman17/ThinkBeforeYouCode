from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ollama import Client

from tbyc_dataset.evaluation.prompt import build_issue_thought_prompt
from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import curated_dataset_path, ensure_directory, read_json, read_jsonl, write_json


LOGGER = logging.getLogger(__name__)
PROGRESS_LOG_EVERY = 25


@dataclass(frozen=True)
class IssueThoughtSettings:
    model_id: str = "qwen2.5:14b"
    model_url: str = "http://localhost:11434"
    include_context: bool = True
    max_context_chars: int = 32768
    max_context_chunks: int = 10
    num_ctx: int = 32768
    issue_number: Optional[int] = None
    skip_existing: bool = True


class IssueThoughtPipeline:
    def __init__(self, output_root: str, settings: IssueThoughtSettings) -> None:
        self.output_root = Path(output_root)
        self.settings = settings

    def run(self, owner: str, repo: str) -> Dict[str, Any]:
        repo_ref = RepositoryRef(owner=owner, name=repo)
        issue_dir = self.output_root / "raw" / repo_ref.fs_slug / "issues"
        retrieval_results_dir = self.output_root / "evaluation" / repo_ref.fs_slug / "results"
        responses_dir = self.output_root / "responses" / repo_ref.fs_slug
        ensure_directory(responses_dir)
        processed_issues = self._load_processed_issue_lookup(repo_ref)

        issue_paths = sorted(issue_dir.glob("issue_*.json"))
        if self.settings.issue_number is not None:
            issue_paths = [path for path in issue_paths if path.name == f"issue_{self.settings.issue_number}.json"]

        if not issue_paths:
            raise FileNotFoundError(f"No issue snapshots found under {issue_dir}")

        LOGGER.info("loaded %s issues from %s", len(issue_paths), issue_dir)
        LOGGER.info("responses will be written to %s", responses_dir)

        client = Client(host=self.settings.model_url)
        outputs: List[Dict[str, Any]] = []
        issue_iterator = self._progress_iter(
            issue_paths,
            desc="Generating issue thoughts",
            unit="issue",
            total=len(issue_paths),
        )
        for issue_path in issue_iterator:
            raw_issue = read_json(issue_path)
            issue_number = _issue_number_from_payload(raw_issue, issue_path)
            prompt_issue = _build_prompt_issue(raw_issue, processed_issues.get(issue_number))
            response_path = responses_dir / f"issue_{issue_number}.json"
            if self.settings.skip_existing and response_path.exists():
                outputs.append(read_json(response_path))
                continue

            retrieval_chunks = self._load_retrieval_chunks(retrieval_results_dir, issue_number)
            selected_chunks = select_shortest_context_chunks(
                retrieval_chunks,
                max_chunks=self.settings.max_context_chunks,
                max_chars=self.settings.max_context_chars,
            )
            prompt, selected_chunks = build_prompt_with_budget(
                issue=prompt_issue,
                include_context=self.settings.include_context,
                candidate_context_blocks=selected_chunks,
                max_prompt_chars=self.settings.max_context_chars,
                max_context_chunks=self.settings.max_context_chunks,
            )
            response = client.chat(
                model=self.settings.model_id,
                messages=[{"role": "user", "content": prompt}],
                options={"num_ctx": max(512, int(self.settings.num_ctx))},
            )
            content = str(response.get("message", {}).get("content", "")).strip()

            payload = {
                "repository": repo_ref.slug,
                "issue_number": issue_number,
                "model_id": self.settings.model_id,
                "model_url": self.settings.model_url,
                "prompt_flags": {
                    "include_context": self.settings.include_context,
                },
                "context": {
                    "max_prompt_chars": self.settings.max_context_chars,
                    "max_context_chunks": self.settings.max_context_chunks,
                    "retrieved_chunk_count": len(retrieval_chunks),
                    "selected_chunk_count": len(selected_chunks),
                    "selected_chunk_ids": [str(chunk.get("chunk_id") or "") for chunk in selected_chunks],
                    "selected_text_chars": sum(len(str(chunk.get("text") or "")) for chunk in selected_chunks),
                    "prompt_chars": len(prompt),
                },
                "issue": {
                    "number": issue_number,
                    "title": str(prompt_issue.get("title") or ""),
                    "body": str(prompt_issue.get("body") or ""),
                    "url": raw_issue.get("url"),
                    "createdAt": raw_issue.get("createdAt"),
                },
                "prompt": prompt,
                "response": content,
            }
            write_json(response_path, payload)
            outputs.append(payload)

        return {
            "repository": repo_ref.slug,
            "response_dir": str(responses_dir),
            "issue_count": len(outputs),
            "responses": outputs,
        }

    def _load_processed_issue_lookup(self, repo_ref: RepositoryRef) -> Dict[int, Mapping[str, Any]]:
        path = curated_dataset_path(self.output_root, repo_ref)
        if not path.exists():
            return {}
        rows = read_jsonl(path)
        lookup: Dict[int, Mapping[str, Any]] = {}
        for row in rows:
            try:
                raw_issue_number = row.get("issue_number")
                if raw_issue_number is None:
                    continue
                issue_number = int(raw_issue_number)
            except (TypeError, ValueError):
                continue
            lookup[issue_number] = row
        return lookup

    def _load_retrieval_chunks(self, retrieval_results_dir: Path, issue_number: int) -> List[Mapping[str, Any]]:
        result_path = retrieval_results_dir / f"issue_{issue_number}.json"
        if not result_path.exists():
            LOGGER.warning("no retrieval result found for issue %s at %s", issue_number, result_path)
            return []
        payload = read_json(result_path)
        results = payload.get("results", [])
        if not isinstance(results, list):
            return []
        return [chunk for chunk in results if isinstance(chunk, dict)]

    def _progress_iter(
        self,
        iterable: Iterable[Any],
        *,
        desc: str,
        unit: str,
        total: Optional[int] = None,
    ) -> Iterable[Any]:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return _log_progress_iter(iterable, desc=desc, unit=unit, total=total)
        return tqdm(iterable, desc=desc, unit=unit, total=total)


def select_shortest_context_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    max_chunks: int,
    max_chars: int,
) -> List[Mapping[str, Any]]:
    ranked = sorted(chunks, key=lambda item: len(str(item.get("text") or "")))
    selected: List[Mapping[str, Any]] = []
    total_chars = 0

    for chunk in ranked:
        if len(selected) >= max(0, max_chunks):
            break
        text = str(chunk.get("text") or "")
        chunk_chars = len(text)
        if chunk_chars <= 0:
            continue
        if total_chars + chunk_chars > max(0, max_chars):
            break
        selected.append(chunk)
        total_chars += chunk_chars

    return selected


def build_prompt_with_budget(
    *,
    issue: Mapping[str, Any],
    include_context: bool,
    candidate_context_blocks: Sequence[Mapping[str, Any]],
    max_prompt_chars: int,
    max_context_chunks: int,
) -> tuple[str, List[Mapping[str, Any]]]:
    budget = max(1, int(max_prompt_chars))
    base_prompt = build_issue_thought_prompt(
        issue,
        include_context=False,
        context_blocks=[],
    )
    # If issue-only prompt already exceeds the configured budget, keep it as-is and
    # skip context rather than failing the run for this issue.
    if len(base_prompt) > budget:
        return base_prompt, []

    if not include_context:
        return base_prompt, []

    ranked = sorted(candidate_context_blocks, key=lambda item: len(str(item.get("text") or "")))
    selected: List[Mapping[str, Any]] = []
    for chunk in ranked:
        if len(selected) >= max(0, max_context_chunks):
            break
        text = str(chunk.get("text") or "")
        if not text:
            continue
        trial = selected + [chunk]
        trial_prompt = build_issue_thought_prompt(
            issue,
            include_context=True,
            context_blocks=trial,
        )
        if len(trial_prompt) > budget:
            break
        selected = trial

    final_prompt = build_issue_thought_prompt(
        issue,
        include_context=True,
        context_blocks=selected,
    )
    if len(final_prompt) > budget:
        return base_prompt, []
    return final_prompt, selected


def _issue_number_from_payload(issue: Mapping[str, Any], issue_path: Path) -> int:
    raw_number = issue.get("number")
    if raw_number is None:
        stem = issue_path.stem
        if stem.startswith("issue_"):
            return int(stem.split("_", 1)[1])
        raise ValueError(f"Issue payload missing number field: {issue_path}")
    return int(raw_number)


def _build_prompt_issue(raw_issue: Mapping[str, Any], processed_issue: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    title = str(raw_issue.get("title") or "").strip()
    body = str(raw_issue.get("body") or "").strip()

    if processed_issue:
        input_vector = processed_issue.get("input_vector")
        if isinstance(input_vector, dict):
            title = str(input_vector.get("title") or title).strip()
            body = str(input_vector.get("body") or body).strip()

        thread = processed_issue.get("deliberation_thread")
        if isinstance(thread, list):
            for comment in thread:
                if not isinstance(comment, dict):
                    continue
                if comment.get("is_issue_body"):
                    opening = str(comment.get("body") or "").strip()
                    if opening:
                        body = opening
                    break

    return {
        "number": raw_issue.get("number"),
        "title": title,
        "body": body,
    }


def _log_progress_iter(
    iterable: Iterable[Any],
    *,
    desc: str,
    unit: str,
    total: Optional[int] = None,
) -> Iterable[Any]:
    for index, item in enumerate(iterable, start=1):
        if index == 1 or index % PROGRESS_LOG_EVERY == 0 or (total is not None and index == total):
            if total is None:
                LOGGER.info("%s: %s %s processed", desc, index, unit)
            else:
                LOGGER.info("%s: %s/%s %s", desc, index, total, unit)
        yield item
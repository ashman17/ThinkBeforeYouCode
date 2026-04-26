from __future__ import annotations

import logging
import os
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv
from ollama import Client

from tbyc_dataset.evaluation.prompt import build_issue_thought_prompt
from tbyc_dataset.extraction.prompt import ARTIFACT_TYPES
from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import curated_dataset_path, ensure_directory, read_json, read_jsonl, write_json


LOGGER = logging.getLogger(__name__)
PROGRESS_LOG_EVERY = 25
DEFAULT_LLM_API_BASE_URL = "https://ai-gateway.andrew.cmu.edu"


@dataclass(frozen=True)
class IssueThoughtSettings:
    model_id: str = "qwen2.5:14b"
    model_url: str = "http://localhost:11434"
    llm_api_base_url: str = DEFAULT_LLM_API_BASE_URL
    llm_api_key_env_var: str = "LLM_KEY"
    include_context: bool = True
    max_context_chars: int = 32768
    max_context_chunks: int = 10
    num_ctx: int = 32768
    response_root_dirname: str = "responses"
    few_shot_from_extractions: bool = False
    few_shot_example_count: int = 3
    few_shot_artifacts_per_example: int = 5
    limit_issues: Optional[int] = None
    issue_number: Optional[int] = None
    skip_existing: bool = True


class IssueThoughtPipeline:
    def __init__(self, output_root: str, settings: IssueThoughtSettings) -> None:
        self.output_root = Path(output_root)
        self.settings = settings

    def run(self, owner: str, repo: str) -> Dict[str, Any]:
        repo_ref = RepositoryRef(owner=owner, name=repo)
        model_dir = _model_dir_name(self.settings.model_id)
        issue_dir = self.output_root / "raw" / repo_ref.fs_slug / "issues"
        retrieval_results_dir = self.output_root / "evaluation" / repo_ref.fs_slug / "results"
        responses_dir = self.output_root / self.settings.response_root_dirname / model_dir / repo_ref.fs_slug
        ensure_directory(responses_dir)
        processed_issues = self._load_processed_issue_lookup(repo_ref)
        raw_issues = self._load_raw_issue_lookup(issue_dir)
        few_shot_examples = self._load_few_shot_examples(
            repo_ref=repo_ref,
            raw_issues=raw_issues,
            processed_issues=processed_issues,
        )

        issue_paths = sorted(issue_dir.glob("issue_*.json"))
        if self.settings.issue_number is not None:
            issue_paths = [path for path in issue_paths if path.name == f"issue_{self.settings.issue_number}.json"]
        if self.settings.limit_issues is not None:
            issue_paths = issue_paths[: max(0, int(self.settings.limit_issues))]

        if not issue_paths:
            raise FileNotFoundError(f"No issue snapshots found under {issue_dir}")

        LOGGER.info("loaded %s issues from %s", len(issue_paths), issue_dir)
        LOGGER.info("responses will be written to %s", responses_dir)

        provider, resolved_model_id, run_completion = _build_issue_thought_completion_runner(self.settings)
        LOGGER.info("issue thought provider=%s model=%s", provider, resolved_model_id)
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
                few_shot_examples=_few_shot_examples_for_issue(
                    few_shot_examples,
                    issue_number=issue_number,
                    max_examples=self.settings.few_shot_example_count,
                ),
            )
            content = run_completion(prompt)

            payload = {
                "repository": repo_ref.slug,
                "issue_number": issue_number,
                "model_id": self.settings.model_id,
                "resolved_model_id": resolved_model_id,
                "provider": provider,
                "model_url": self.settings.model_url,
                "llm_api_base_url": self.settings.llm_api_base_url if provider == "api" else None,
                "prompt_flags": {
                    "include_context": self.settings.include_context,
                    "few_shot_from_extractions": self.settings.few_shot_from_extractions,
                },
                "context": {
                    "max_prompt_chars": self.settings.max_context_chars,
                    "max_context_chunks": self.settings.max_context_chunks,
                    "retrieved_chunk_count": len(retrieval_chunks),
                    "selected_chunk_count": len(selected_chunks),
                    "selected_chunk_ids": [str(chunk.get("chunk_id") or "") for chunk in selected_chunks],
                    "selected_text_chars": sum(len(str(chunk.get("text") or "")) for chunk in selected_chunks),
                    "prompt_chars": len(prompt),
                    "few_shot_example_count": len(
                        _few_shot_examples_for_issue(
                            few_shot_examples,
                            issue_number=issue_number,
                            max_examples=self.settings.few_shot_example_count,
                        )
                    ),
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
            "model_id": self.settings.model_id,
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

    def _load_raw_issue_lookup(self, issue_dir: Path) -> Dict[int, Mapping[str, Any]]:
        lookup: Dict[int, Mapping[str, Any]] = {}
        for issue_path in sorted(issue_dir.glob("issue_*.json")):
            payload = read_json(issue_path)
            try:
                number = _issue_number_from_payload(payload, issue_path)
            except Exception:
                continue
            lookup[number] = payload
        return lookup

    def _load_few_shot_examples(
        self,
        *,
        repo_ref: RepositoryRef,
        raw_issues: Mapping[int, Mapping[str, Any]],
        processed_issues: Mapping[int, Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.settings.few_shot_from_extractions:
            return []

        extraction_dir = self.output_root / "extractions" / repo_ref.fs_slug
        if not extraction_dir.exists():
            LOGGER.warning("few-shot extraction directory missing at %s", extraction_dir)
            return []

        examples: List[Dict[str, Any]] = []
        for path in sorted(extraction_dir.glob("issue_*.json")):
            payload = read_json(path)
            issue = payload.get("issue", {})
            if not isinstance(issue, dict):
                continue
            try:
                issue_number = int(issue.get("issue_number"))
            except Exception:
                continue

            response = _format_few_shot_response_from_issue(
                issue,
                max_artifacts=self.settings.few_shot_artifacts_per_example,
            )
            if not response:
                continue

            raw_issue = raw_issues.get(issue_number, {})
            processed_issue = processed_issues.get(issue_number)
            prompt_issue = _build_prompt_issue(raw_issue, processed_issue)
            examples.append(
                {
                    "issue_number": issue_number,
                    "title": str(prompt_issue.get("title") or ""),
                    "body": _truncate_text(str(prompt_issue.get("body") or ""), 900),
                    "response": response,
                }
            )
        return examples

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


def _model_dir_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__") or "unknown-model"


def _build_issue_thought_completion_runner(
    settings: IssueThoughtSettings,
) -> Tuple[str, str, Callable[[str], str]]:
    provider, resolved_model_id = _resolve_issue_thought_provider(settings.model_id)
    if provider == "api":
        load_dotenv()
        api_key = os.environ.get(settings.llm_api_key_env_var, "").strip()
        if not api_key:
            raise ValueError(
                f"{settings.llm_api_key_env_var} is required for API models. "
                "Set it in your shell or place it in a .env file."
            )

        try:
            openai_module = importlib.import_module("openai")
            OpenAI = getattr(openai_module, "OpenAI")
        except Exception as exc:
            raise ImportError("openai is required for API model calls. Install project dependencies.") from exc

        client = OpenAI(api_key=api_key, base_url=settings.llm_api_base_url)

        def _run_api_completion(prompt: str) -> str:
            response = client.chat.completions.create(
                model=resolved_model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            message = response.choices[0].message if response.choices else None
            return str(getattr(message, "content", "") or "").strip()

        return provider, resolved_model_id, _run_api_completion

    client = Client(host=settings.model_url)

    def _run_ollama_completion(prompt: str) -> str:
        response = client.chat(
            model=resolved_model_id,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": max(512, int(settings.num_ctx))},
        )
        return str(response.get("message", {}).get("content", "")).strip()

    return provider, resolved_model_id, _run_ollama_completion


def _resolve_issue_thought_provider(model_id: str) -> Tuple[str, str]:
    normalized = model_id.strip()
    lowered = normalized.lower()

    # Explicit prefixes are the safest way to route requests.
    if lowered.startswith("api/"):
        return "api", normalized[4:]
    if lowered.startswith("openai/"):
        return "api", normalized[7:]
    if lowered.startswith("ollama/"):
        return "ollama", normalized[7:]

    # For unprefixed IDs, use a small heuristic:
    # - Ollama tags commonly use size labels (qwen2.5:14b, llama3.1:8b, :latest).
    # - API providers (for example Bedrock model IDs) often use numeric revision
    #   suffixes like claude-opus-4-20250514-v1:0.
    if ":" in normalized:
        _, suffix = normalized.rsplit(":", 1)
        suffix_lower = suffix.lower()
        if suffix.isdigit():
            return "api", normalized
        if re.match(r"^\d+(?:\.\d+)?[bm](?:[-_].*)?$", suffix_lower):
            return "ollama", normalized
        if suffix_lower.endswith(("b", "m")) or suffix_lower == "latest":
            return "ollama", normalized
        return "api", normalized

    # Default to API for untagged model IDs.
    return "api", normalized


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
    few_shot_examples: Optional[Sequence[Mapping[str, Any]]] = None,
) -> tuple[str, List[Mapping[str, Any]]]:
    budget = max(1, int(max_prompt_chars))
    base_prompt = build_issue_thought_prompt(
        issue,
        include_context=False,
        context_blocks=[],
        few_shot_examples=few_shot_examples,
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
            few_shot_examples=few_shot_examples,
        )
        if len(trial_prompt) > budget:
            break
        selected = trial

    final_prompt = build_issue_thought_prompt(
        issue,
        include_context=True,
        context_blocks=selected,
        few_shot_examples=few_shot_examples,
    )
    if len(final_prompt) > budget:
        return base_prompt, []
    return final_prompt, selected


_TYPE_TO_TOPIC = {
    "problem_statement": "Problem Statement",
    "proposed_solution": "Proposed Solution",
    "alternative_solution": "Alternative Solution",
    "design_decision": "Design Decision",
    "trade_off_argument": "Trade-off Argument",
    "rationale": "Rationale",
    "constraint": "Constraint",
    "assumption": "Assumption",
    "implementation_detail": "Implementation Detail",
    "code_snippet": "Code Snippet",
    "algorithm_approach": "Algorithm / Approach",
    "api_design": "API Design",
    "data_structure_choice": "Data Structure Choice",
    "configuration_choice": "Configuration Choice",
    "benchmark_result": "Benchmark Result",
    "performance_claim": "Performance Claim",
    "test_case": "Test Case",
    "bug_reproduction_steps": "Bug Reproduction Steps",
    "edge_case": "Edge Case",
    "empirical_evidence": "Empirical Evidence",
    "question": "Question",
    "answer_clarification": "Answer / Clarification",
    "agreement": "Agreement",
    "disagreement": "Disagreement",
    "suggestion": "Suggestion",
    "critique": "Critique",
    "task_assignment": "Task Assignment",
    "status_update": "Status Update",
    "priority_discussion": "Priority Discussion",
    "blocking_issue": "Blocking Issue",
    "dependency": "Dependency",
}


def _few_shot_examples_for_issue(
    examples: Sequence[Mapping[str, Any]],
    *,
    issue_number: int,
    max_examples: int,
) -> List[Mapping[str, Any]]:
    selected = [example for example in examples if int(example.get("issue_number", -1)) != issue_number]
    return selected[: max(0, max_examples)]


def _format_few_shot_response_from_issue(issue: Mapping[str, Any], *, max_artifacts: int) -> str:
    lines: List[str] = []
    for comment in issue.get("comments", []) if isinstance(issue.get("comments"), list) else []:
        if not isinstance(comment, Mapping):
            continue
        for artifact in comment.get("artifacts", []) if isinstance(comment.get("artifacts"), list) else []:
            if not isinstance(artifact, Mapping):
                continue
            artifact_type = str(artifact.get("type") or "").strip()
            summary = str(artifact.get("summary") or "").strip()
            if artifact_type not in ARTIFACT_TYPES or not summary:
                continue
            topic = _TYPE_TO_TOPIC.get(artifact_type)
            if not topic:
                continue
            lines.append(f"[{topic}] {_truncate_text(summary, 220)}")
            if len(lines) >= max(1, max_artifacts):
                return "\n".join(lines)
    return "\n".join(lines)


def _truncate_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


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

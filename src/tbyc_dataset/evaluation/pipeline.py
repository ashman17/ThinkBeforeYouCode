from __future__ import annotations

import ast
import json
import logging
import os
import pickle
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast
from urllib import error, parse, request

import numpy as np

from tbyc_dataset.evaluation.config import RetrievalSettings
from tbyc_dataset.models import RepositoryRef
from tbyc_dataset.storage import ensure_directory, read_json, write_json


LOGGER = logging.getLogger(__name__)

HTTP_TIMEOUT_SECONDS = 60
PROGRESS_LOG_EVERY = 250


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    file_path: str
    symbol_name: Optional[str]
    language: str
    start_line: int
    end_line: int
    text: str

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IssueRecord:
    number: int
    title: str
    body: str
    created_at: str
    url: Optional[str]

    @property
    def query(self) -> str:
        return "\n\n".join(part for part in (self.title.strip(), self.body.strip()) if part)

    def to_json(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "created_at": self.created_at,
            "url": self.url,
        }


class CodeRetrievalPipeline:
    def __init__(
        self,
        cache_dir: str,
        embedding_model: str = "microsoft/codebert-base",
        chunk_size: int = 50,
        top_n: int = 20,
        bm25_top_k: int = 50,
        dense_top_k: int = 50,
        chunk_overlap: int = 10,
        rrf_k: int = 60,
        max_tokens: int = 512,
    ) -> None:
        self.settings = RetrievalSettings(
            cache_dir=Path(cache_dir),
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            top_n=top_n,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
            chunk_overlap=chunk_overlap,
            rrf_k=rrf_k,
            max_tokens=max_tokens,
        )

    def run(self, owner: str, repo: str, output_dir: str) -> Dict[str, Any]:
        repo_ref = RepositoryRef(owner=owner, name=repo)
        issue_dir = self.settings.cache_dir / "raw" / repo_ref.fs_slug / "issues"
        issues = self._load_issues(issue_dir)
        if not issues:
            raise FileNotFoundError(f"No issue snapshots found under {issue_dir}")

        output_root = Path(output_dir) / repo_ref.fs_slug
        ensure_directory(output_root)
        index_dir = self._find_resumable_index_dir(output_root)

        LOGGER.info("loaded %s issues from %s", len(issues), issue_dir)
        if index_dir is not None:
            snapshot_commit = index_dir.name
            LOGGER.info(
                "reusing cached indexes from %s; skipping snapshot download and index rebuild",
                index_dir,
            )
            chunks = self._read_chunks(index_dir / "chunks.jsonl")
            LOGGER.info("loaded %s code chunks", len(chunks))
            bundle = self._load_or_build_indexes(chunks, index_dir)
            LOGGER.info("indexes ready")
        else:
            LOGGER.info("resolving snapshot commit before earliest issue")
            snapshot_commit = self._resolve_snapshot_commit(owner, repo, issues)
            LOGGER.info("using snapshot commit %s", snapshot_commit)
            index_dir = output_root / "indexes" / snapshot_commit
            snapshot_dir = self._ensure_repo_snapshot(owner, repo, snapshot_commit)
            LOGGER.info("snapshot ready at %s", snapshot_dir)
            chunks = self._load_or_build_chunks(snapshot_dir, index_dir)
            LOGGER.info("loaded %s code chunks", len(chunks))
            bundle = self._load_or_build_indexes(chunks, index_dir)
            LOGGER.info("indexes ready")
            self._cleanup_repo_snapshot(owner, repo, snapshot_commit)

        issue_results: List[Dict[str, Any]] = []
        for issue in self._progress_iter(
            issues,
            desc="Retrieving issues",
            unit="issue",
            total=len(issues),
        ):
            result = self._retrieve_for_issue(issue, chunks, bundle)
            issue_results.append(result)
            write_json(output_root / "results" / f"issue_{issue.number}.json", result)

        manifest = {
            "repository": repo_ref.slug,
            "issue_count": len(issues),
            "snapshot_commit": snapshot_commit,
            "snapshot_dir": None,
            "index_dir": str(index_dir),
            "results_dir": str(output_root / "results"),
            "settings": {
                "embedding_model": self.settings.embedding_model,
                "top_n": self.settings.top_n,
                "bm25_top_k": self.settings.bm25_top_k,
                "dense_top_k": self.settings.dense_top_k,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "rrf_k": self.settings.rrf_k,
                "max_tokens": self.settings.max_tokens,
            },
        }
        write_json(output_root / "manifest.json", manifest)
        return {
            "manifest": manifest,
            "issues": issue_results,
        }

    def _find_resumable_index_dir(self, output_root: Path) -> Optional[Path]:
        manifest_path = output_root / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = read_json(manifest_path)
                snapshot_commit = str(manifest.get("snapshot_commit") or "").strip()
                if snapshot_commit:
                    manifest_index_dir = output_root / "indexes" / snapshot_commit
                    if self._is_usable_index_dir(manifest_index_dir):
                        return manifest_index_dir
            except Exception as exc:
                LOGGER.warning("failed to read cached manifest %s: %s", manifest_path, exc)

        indexes_root = output_root / "indexes"
        if not indexes_root.exists():
            return None

        candidates = [
            path
            for path in indexes_root.iterdir()
            if path.is_dir() and self._is_usable_index_dir(path)
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        LOGGER.warning(
            "multiple cached index directories found under %s; using most recent %s",
            indexes_root,
            candidates[0],
        )
        return candidates[0]

    def _is_usable_index_dir(self, index_dir: Path) -> bool:
        required_paths = (
            index_dir / "chunks.jsonl",
            index_dir / "bm25.pkl",
            index_dir / "embeddings.npy",
        )
        return all(path.exists() for path in required_paths)

    def _load_issues(self, issue_dir: Path) -> List[IssueRecord]:
        issues: List[IssueRecord] = []
        for path in sorted(issue_dir.glob("issue_*.json")):
            payload = read_json(path)
            issues.append(
                IssueRecord(
                    number=int(payload["number"]),
                    title=(payload.get("title") or "").strip(),
                    body=(payload.get("body") or "").strip(),
                    created_at=payload["createdAt"],
                    url=payload.get("url"),
                )
            )
        issues.sort(key=lambda issue: (issue.created_at, issue.number))
        return issues

    def _resolve_snapshot_commit(
        self,
        owner: str,
        repo: str,
        issues: Sequence[IssueRecord],
    ) -> str:
        earliest_issue_time = min(_parse_github_timestamp(issue.created_at) for issue in issues)
        default_branch = self._fetch_default_branch(owner, repo)
        params = parse.urlencode(
            {
                "sha": default_branch,
                "per_page": 1,
                "until": earliest_issue_time.isoformat().replace("+00:00", "Z"),
            }
        )
        commits = self._github_json(
            f"https://api.github.com/repos/{owner}/{repo}/commits?{params}"
        )
        if not commits:
            raise RuntimeError(
                f"Unable to find a commit on {owner}/{repo}@{default_branch} before "
                f"{earliest_issue_time.isoformat()}."
            )
        return str(commits[0]["sha"])

    def _fetch_default_branch(self, owner: str, repo: str) -> str:
        metadata = self._github_json(f"https://api.github.com/repos/{owner}/{repo}")
        branch = str(metadata.get("default_branch") or "").strip()
        if not branch:
            raise RuntimeError(f"GitHub did not return a default branch for {owner}/{repo}.")
        return branch

    def _ensure_repo_snapshot(self, owner: str, repo: str, commit_sha: str) -> Path:
        repo_cache_root = self.settings.repo_cache_root(owner, repo)
        snapshots_root = repo_cache_root
        snapshot_dir = snapshots_root / commit_sha
        marker_path = snapshot_dir / ".complete"
        if marker_path.exists():
            LOGGER.info("reusing cached snapshot %s", snapshot_dir)
            return snapshot_dir

        archive_path = repo_cache_root / "archives" / f"{commit_sha}.zip"
        ensure_directory(archive_path.parent)
        ensure_directory(snapshot_dir.parent)
        if not archive_path.exists():
            url = f"https://github.com/{owner}/{repo}/archive/{commit_sha}.zip"
            LOGGER.info("downloading snapshot %s", url)
            self._download_file(url, archive_path)

        temp_extract_dir = Path(tempfile.mkdtemp(prefix="repo_snapshot_", dir=str(snapshots_root)))
        try:
            with zipfile.ZipFile(archive_path) as archive:
                LOGGER.info("extracting %s", archive_path)
                archive.extractall(temp_extract_dir)
            children = [child for child in temp_extract_dir.iterdir() if child.is_dir()]
            if len(children) != 1:
                raise RuntimeError(
                    f"Expected a single top-level directory in {archive_path}, found {len(children)}."
                )
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            shutil.move(str(children[0]), str(snapshot_dir))
            self._prune_non_source_files(snapshot_dir)
            marker_path.write_text("ok\n", encoding="utf-8")
        finally:
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
        return snapshot_dir

    def _prune_non_source_files(self, snapshot_dir: Path) -> None:
        all_paths = list(snapshot_dir.rglob("*"))
        kept_files = 0
        removed_files = 0
        for path in self._progress_iter(
            all_paths,
            desc="Filtering source files",
            unit="path",
            total=len(all_paths),
        ):
            if path.is_dir():
                continue
            if path.suffix.lower() not in self.settings.source_extensions:
                path.unlink()
                removed_files += 1
            else:
                kept_files += 1
        for directory in sorted(snapshot_dir.rglob("*"), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        LOGGER.info(
            "source filtering kept %s files and removed %s non-source files",
            kept_files,
            removed_files,
        )

    def _load_or_build_chunks(self, snapshot_dir: Path, index_dir: Path) -> List[CodeChunk]:
        chunks_path = index_dir / "chunks.jsonl"
        if chunks_path.exists():
            LOGGER.info("reusing cached chunks from %s", chunks_path)
            return self._read_chunks(chunks_path)

        ensure_directory(index_dir)
        chunks: List[CodeChunk] = []
        files = sorted(path for path in snapshot_dir.rglob("*") if path.is_file())
        for file_path in self._progress_iter(
            files,
            desc="Chunking source files",
            unit="file",
            total=len(files),
        ):
            relative_path = str(file_path.relative_to(snapshot_dir))
            file_chunks = self._chunk_file(file_path, relative_path)
            chunks.extend(file_chunks)

        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_json(), sort_keys=True))
                handle.write("\n")
        return chunks

    def _read_chunks(self, chunks_path: Path) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                chunks.append(CodeChunk(**payload))
        return chunks

    def _chunk_file(self, file_path: Path, relative_path: str) -> List[CodeChunk]:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        lines = text.splitlines()
        language = file_path.suffix.lower().lstrip(".")
        structural_chunks = self._extract_structural_chunks(relative_path, text, lines)
        if structural_chunks:
            return structural_chunks
        return self._fallback_line_chunks(relative_path, language, lines)

    def _extract_structural_chunks(
        self,
        relative_path: str,
        text: str,
        lines: Sequence[str],
    ) -> List[CodeChunk]:
        if relative_path.endswith(".py"):
            return self._extract_python_chunks(relative_path, text)

        regex = LANGUAGE_SYMBOL_PATTERNS.get(Path(relative_path).suffix.lower())
        if not regex:
            return []

        matches = list(regex.finditer(text))
        if not matches:
            return []

        chunks: List[CodeChunk] = []
        line_offsets = _line_offsets(text)
        for index, match in enumerate(matches):
            start_char = match.start()
            end_char = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            start_line = _char_to_line(start_char, line_offsets)
            end_line = max(start_line, _char_to_line(end_char, line_offsets) - 1)
            chunk_text = "\n".join(lines[start_line - 1 : end_line]).strip()
            if not chunk_text:
                continue
            symbol_name = next((group for group in match.groups() if group), None)
            chunks.append(
                CodeChunk(
                    chunk_id=f"{relative_path}::{start_line}-{end_line}",
                    file_path=relative_path,
                    symbol_name=symbol_name,
                    language=Path(relative_path).suffix.lower().lstrip("."),
                    start_line=start_line,
                    end_line=end_line,
                    text=chunk_text,
                )
            )
        return chunks

    def _extract_python_chunks(self, relative_path: str, text: str) -> List[CodeChunk]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []

        chunks: List[CodeChunk] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start_line = getattr(node, "lineno", None)
            end_line = getattr(node, "end_lineno", None)
            if start_line is None or end_line is None:
                continue
            chunk_text = "\n".join(text.splitlines()[start_line - 1 : end_line]).strip()
            if not chunk_text:
                continue
            chunks.append(
                CodeChunk(
                    chunk_id=f"{relative_path}::{start_line}-{end_line}",
                    file_path=relative_path,
                    symbol_name=getattr(node, "name", None),
                    language="py",
                    start_line=start_line,
                    end_line=end_line,
                    text=chunk_text,
                )
            )
        chunks.sort(key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.symbol_name or ""))
        return chunks

    def _fallback_line_chunks(
        self,
        relative_path: str,
        language: str,
        lines: Sequence[str],
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        if not lines:
            return chunks

        size = max(1, self.settings.chunk_size)
        overlap = min(max(0, self.settings.chunk_overlap), size - 1)
        step = max(1, size - overlap)
        start = 0
        while start < len(lines):
            end = min(len(lines), start + size)
            chunk_text = "\n".join(lines[start:end]).strip()
            if chunk_text:
                chunks.append(
                    CodeChunk(
                        chunk_id=f"{relative_path}::{start + 1}-{end}",
                        file_path=relative_path,
                        symbol_name=None,
                        language=language,
                        start_line=start + 1,
                        end_line=end,
                        text=chunk_text,
                    )
                )
            if end >= len(lines):
                break
            start += step
        return chunks

    def _load_or_build_indexes(
        self,
        chunks: Sequence[CodeChunk],
        index_dir: Path,
    ) -> Dict[str, Any]:
        ensure_directory(index_dir)
        bm25_path = index_dir / "bm25.pkl"
        embeddings_path = index_dir / "embeddings.npy"
        metadata_path = index_dir / "index_metadata.json"

        tokenized_chunks = [tokenize_for_bm25(chunk.text) for chunk in chunks]
        if bm25_path.exists():
            with bm25_path.open("rb") as handle:
                bm25 = pickle.load(handle)
        else:
            LOGGER.info("building BM25 index for %s chunks", len(chunks))
            bm25 = self._build_bm25(tokenized_chunks)
            with bm25_path.open("wb") as handle:
                pickle.dump(bm25, handle)

        if embeddings_path.exists():
            LOGGER.info("reusing cached embeddings from %s", embeddings_path)
            embeddings = np.load(embeddings_path)
        else:
            LOGGER.info("computing dense embeddings for %s chunks", len(chunks))
            embeddings = self._embed_texts(
                [chunk.text for chunk in chunks],
                show_progress=True,
            )
            np.save(embeddings_path, embeddings)

        if not metadata_path.exists():
            write_json(
                metadata_path,
                {
                    "chunk_count": len(chunks),
                    "embedding_model": self.settings.embedding_model,
                    "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
                },
            )

        faiss_index = self._build_faiss_index(embeddings)
        return {
            "bm25": bm25,
            "tokenized_chunks": tokenized_chunks,
            "embeddings": embeddings,
            "faiss_index": faiss_index,
        }

    def _build_bm25(self, tokenized_chunks: Sequence[Sequence[str]]) -> Any:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for retrieval. Install the 'eval' extras."
            ) from exc
        return BM25Okapi(tokenized_chunks)

    def _embed_texts(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        tokenizer, model, device = self._load_embedding_model()
        vectors: List[np.ndarray] = []
        batch_size = 16
        batch_starts = range(0, len(texts), batch_size)
        total_batches = (len(texts) + batch_size - 1) // batch_size
        iterator: Iterable[int]
        if show_progress:
            iterator = self._progress_iter(
                batch_starts,
                desc="Embedding chunks",
                unit="batch",
                total=total_batches,
            )
        else:
            iterator = batch_starts
        for batch_start in iterator:
            batch = list(texts[batch_start : batch_start + batch_size])
            batch_vectors = self._encode_batch(
                tokenizer=tokenizer,
                model=model,
                device=device,
                texts=batch,
            )
            vectors.extend(batch_vectors)
        return np.vstack(vectors).astype(np.float32)

    def _cleanup_repo_snapshot(self, owner: str, repo: str, commit_sha: str) -> None:
        repo_cache_root = self.settings.repo_cache_root(owner, repo)
        snapshot_dir = repo_cache_root / commit_sha
        archive_path = repo_cache_root / "archives" / f"{commit_sha}.zip"

        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            LOGGER.info("removed extracted snapshot %s", snapshot_dir)
        if archive_path.exists():
            archive_path.unlink()
            LOGGER.info("removed snapshot archive %s", archive_path)

    def _load_embedding_model(self) -> Tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for dense retrieval. Install the 'eval' extras."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.settings.embedding_model)
        model = AutoModel.from_pretrained(self.settings.embedding_model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return tokenizer, model, device

    def _encode_batch(
        self,
        tokenizer: Any,
        model: Any,
        device: Any,
        texts: Sequence[str],
    ) -> List[np.ndarray]:
        import torch

        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.settings.max_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden * attention_mask
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return [row.detach().cpu().numpy() for row in pooled]

    def _build_faiss_index(self, embeddings: np.ndarray) -> Any:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for dense retrieval. Install the 'eval' extras."
            ) from exc

        if embeddings.size == 0:
            return None
        index = cast(Any, faiss.IndexFlatIP(int(embeddings.shape[1])))
        matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
        LOGGER.info("adding %s vectors to FAISS index", matrix.shape[0])
        index.add(matrix)
        return index

    def _retrieve_for_issue(
        self,
        issue: IssueRecord,
        chunks: Sequence[CodeChunk],
        bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        query = issue.query
        bm25_ranked = self._bm25_search(query, bundle["bm25"], chunks)
        dense_ranked = self._dense_search(query, bundle["faiss_index"], chunks)
        fused_ranked = reciprocal_rank_fusion(
            [bm25_ranked, dense_ranked],
            rrf_k=self.settings.rrf_k,
            top_n=self.settings.top_n,
        )

        results = []
        for rank, (chunk_index, score) in enumerate(fused_ranked, start=1):
            chunk = chunks[chunk_index]
            results.append(
                {
                    "rank": rank,
                    "rrf_score": score,
                    "file_path": chunk.file_path,
                    "chunk_id": chunk.chunk_id,
                    "symbol_name": chunk.symbol_name,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "text": chunk.text,
                }
            )

        return {
            "issue": issue.to_json(),
            "query": query,
            "bm25_top_k": self.settings.bm25_top_k,
            "dense_top_k": self.settings.dense_top_k,
            "top_n": self.settings.top_n,
            "results": results,
        }

    def _bm25_search(
        self,
        query: str,
        bm25: Any,
        chunks: Sequence[CodeChunk],
    ) -> List[Tuple[int, float]]:
        query_tokens = tokenize_for_bm25(query)
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[: min(self.settings.bm25_top_k, len(chunks))]

    def _dense_search(
        self,
        query: str,
        faiss_index: Any,
        chunks: Sequence[CodeChunk],
    ) -> List[Tuple[int, float]]:
        if faiss_index is None or not chunks:
            return []
        query_embedding = self._embed_texts([query], show_progress=False).astype(np.float32)
        scores, indices = faiss_index.search(query_embedding, min(self.settings.dense_top_k, len(chunks)))
        ranked: List[Tuple[int, float]] = []
        for chunk_index, score in zip(indices[0].tolist(), scores[0].tolist()):
            if chunk_index < 0:
                continue
            ranked.append((int(chunk_index), float(score)))
        return ranked

    def _github_json(self, url: str) -> Any:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "tbyc-dataset"}
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = request.Request(url, headers=headers)
        try:
            with request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"GitHub request failed for {url}: {exc.code} {detail}") from exc

    def _download_file(self, url: str, destination: Path) -> None:
        headers = {"User-Agent": "tbyc-dataset"}
        req = request.Request(url, headers=headers)
        try:
            with request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
                ensure_directory(destination.parent)
                with destination.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Download failed for {url}: {exc.code} {detail}") from exc

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
            return self._log_progress_iter(iterable, desc=desc, unit=unit, total=total)
        return tqdm(iterable, desc=desc, unit=unit, total=total)

    def _log_progress_iter(
        self,
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


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Tuple[int, float]]],
    rrf_k: int,
    top_n: int,
) -> List[Tuple[int, float]]:
    fused_scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]+|\d+", text.lower())


def _parse_github_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _line_offsets(text: str) -> List[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    return offsets


def _char_to_line(char_index: int, line_offsets: Sequence[int]) -> int:
    low = 0
    high = len(line_offsets) - 1
    while low <= high:
        mid = (low + high) // 2
        if line_offsets[mid] <= char_index:
            low = mid + 1
        else:
            high = mid - 1
    return max(1, low)


LANGUAGE_SYMBOL_PATTERNS = {
    ".c": re.compile(r"^[\w\s\*:&<>,\[\]]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{", re.MULTILINE),
    ".cc": re.compile(r"^[\w\s\*:&<>,~]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(const)?\s*\{", re.MULTILINE),
    ".cpp": re.compile(r"^[\w\s\*:&<>,~]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(const)?\s*\{", re.MULTILINE),
    ".cs": re.compile(r"^\s*(?:public|private|protected|internal|static|virtual|async|\s)+[\w<>\[\],]+\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    ".go": re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    ".h": re.compile(r"^[\w\s\*:&<>,~]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(const)?\s*(?:;|\{)", re.MULTILINE),
    ".hpp": re.compile(r"^[\w\s\*:&<>,~]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(const)?\s*(?:;|\{)", re.MULTILINE),
    ".java": re.compile(r"^\s*(?:public|private|protected|static|final|synchronized|abstract|\s)+[\w<>\[\],]+\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
    ".js": re.compile(r"^\s*(?:function\s+([A-Za-z_]\w*)\s*\(|(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\()", re.MULTILINE),
    ".jsx": re.compile(r"^\s*(?:function\s+([A-Za-z_]\w*)\s*\(|(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\()", re.MULTILINE),
    ".kt": re.compile(r"^\s*(?:class|fun)\s+([A-Za-z_]\w*)", re.MULTILINE),
    ".php": re.compile(r"^\s*(?:class|function)\s+([A-Za-z_]\w*)", re.MULTILINE),
    ".rb": re.compile(r"^\s*(?:class|module|def)\s+([A-Za-z_]\w*[!?=]?)", re.MULTILINE),
    ".rs": re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+([A-Za-z_]\w*)", re.MULTILINE),
    ".scala": re.compile(r"^\s*(?:class|object|trait|def)\s+([A-Za-z_]\w*)", re.MULTILINE),
    ".swift": re.compile(r"^\s*(?:class|struct|enum|protocol|func)\s+([A-Za-z_]\w*)", re.MULTILINE),
    ".ts": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface)\s+([A-Za-z_]\w*)", re.MULTILINE),
    ".tsx": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface)\s+([A-Za-z_]\w*)", re.MULTILINE),
}

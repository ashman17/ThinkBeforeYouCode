# Evaluation Retrieval Pipeline

`CodeRetrievalPipeline` builds a hybrid retrieval stack for GitHub issues:

1. Load curated raw issue snapshots from `data/raw/<owner>__<repo>/issues`.
2. Resolve the latest default-branch commit before the earliest issue timestamp.
3. Download and cache a repository zip snapshot for that commit.
4. Prune non-source files and chunk source code at symbol level when possible.
5. Build:
   - a BM25 index over chunk text
   - dense chunk embeddings with CodeBERT
   - a FAISS inner-product index over normalized embeddings
6. Retrieve BM25 and dense candidates for each issue query (`title + body`).
7. Fuse the rankings with Reciprocal Rank Fusion and write per-issue results.

Artifacts are written under `data/evaluation/<owner>__<repo>/`:

- `manifest.json`: run metadata and chosen snapshot commit
- `indexes/<commit>/chunks.jsonl`: chunk metadata and text
- `indexes/<commit>/bm25.pkl`: serialized BM25 index
- `indexes/<commit>/embeddings.npy`: dense embeddings
- `results/issue_<n>.json`: fused retrieval output for each issue

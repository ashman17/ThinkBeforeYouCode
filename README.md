# Think Before You Code Dataset Pipeline

This repository contains the dataset curation foundation for the "Think Before You Code" benchmark. The current implementation focuses on the first stage of the benchmark lifecycle:

- fetching issue-centered GitHub discussions via the GraphQL API,
- preserving raw snapshots for auditability and future reprocessing,
- normalizing those snapshots into factual records,
- structuring the codebase so later phases can add artifact extraction and LLM benchmarking cleanly.

## Why this structure

The benchmark is not just a scraper. It needs a reproducible data pipeline with clear boundaries between:

- ingestion of raw source-of-truth data,
- normalization into stable dataset records,
- extraction of deliberation artifacts from hidden discussion threads,
- evaluation of model behavior against human strategic judgment.

To keep those concerns separate, the repository is organized as a Python package with dedicated stage-specific modules.

The code is split by benchmark stage:

- `tbyc_dataset.dataset`: GitHub fetching, normalization, and dataset curation
- `tbyc_dataset.extraction`: comment-level entity extraction and prompt assets
- `tbyc_dataset.evaluation`: reserved for model-vs-human evaluation pipelines

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── tbyc_dataset/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── github.py
│       │   ├── normalize.py
│       │   ├── pipeline.py
│       │   └── queries.py
│       ├── evaluation/
│       │   └── __init__.py
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── discussion_entities_pipeline.py
│       │   └── discussion_entities_prompt.py
│       ├── models.py
│       └── storage.py
└── tests/
    ├── test_discussion_entities_prompt.py
    └── test_normalize.py
```

## Data layers

The pipeline writes data into two layers under `data/` by default:

- `data/raw/<owner>__<repo>/issues/*.json`
  - exact or near-exact GraphQL snapshots for each issue
- `data/processed/<owner>__<repo>/curated.jsonl`
  - normalized records used by downstream artifact extraction and benchmarking

Each curated record contains:

- input vector: original issue title and body,
- taxonomic metadata: labels, issue state, state reason,
- timeline events: normalized factual event data from the issue timeline,
- deliberation thread: chronological comments excluding the original issue body,
- resolution artifacts: linked pull requests and merged status,
- actor typology: participant associations and roles observed in the payload.

## Setup

The core data pipeline is lightweight. The project dependencies now include
`python-dotenv` and `langextract`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set a GitHub token with repository read access:

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

## Usage

Fetch raw issue snapshots:

```bash
python3 -m tbyc_dataset.cli fetch-repo --repo owner/name --max-issues 100
```

Curate normalized records from previously fetched raw snapshots:

```bash
python3 -m tbyc_dataset.cli curate-repo --repo owner/name
```

Run the full end-to-end pipeline:

```bash
python3 -m tbyc_dataset.cli build-dataset --repo owner/name --max-issues 100
```

Optional filters:

- `--states OPEN CLOSED`
- `--min-comments 2`
- `--max-comments 200`
- `--output-root data`

## Discussion Entity Extraction

The repository also includes a LangExtract-based extraction stage for turning
discussion threads into evaluable reasoning points.

The prompt and few-shot examples live in
`src/tbyc_dataset/extraction/discussion_entities_prompt.py` so they can be tuned
without touching the runtime pipeline.

LangExtract guidance from the upstream project strongly emphasizes that examples
should use exact verbatim spans, appear in source order, and avoid paraphrase. The
prompt/examples in this repository follow that pattern, and the extractor now sends
whole threads in `author: comment` form so cross-comment continuity is preserved.

Extracted spans are also mapped back to the originating GitHub comment so factual
speaker metadata such as author login and speaker role can be attached after
extraction.

Install LangExtract and ensure Ollama is serving your local model:

```bash
pip install langextract
ollama pull gemma3:4b
ollama serve
```

Then run extraction thread-by-thread over an existing curated dataset:

```bash
python3 -m tbyc_dataset.cli extract-discussion-entities \
  --repo bitcoin/bitcoin \
  --model-id gemma3:4b \
  --model-url http://localhost:11434
```

This writes:

- `data/extractions/<owner>__<repo>/discussion_entities.jsonl`
- `data/extractions/<owner>__<repo>/summary.json`

Optional:

- `--limit-threads 25`
- `--save-annotated`

## Future extensions

The current codebase is ready for the next two benchmark phases:

1. comment artifact extraction
   - trade-offs,
   - risk statements,
   - rejection rationales,
   - feasibility constraints,
   - alternative proposals
2. LLM benchmarking
   - prompting on issue-only input vectors,
   - comparing model triage decisions to human ground truth,
   - measuring implementation bias and deliberation coverage

Those additions should layer on top of `curated.jsonl`, without changing raw ingestion.

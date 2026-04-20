# Metrics README

This folder stores evaluation outputs comparing human-extracted artifacts vs LLM-derived artifacts.

## Folder Layout

Metrics are written under:

- `<output-root>/metrics/<model-id>/<repo-slug>/`

Example from this workspace:

- `data/metrics/qwen2.5:7b-instruct/kubernetes__kubernetes/`

Common files in each repo folder:

- `type_matching.json`
- `metadata_matching.json`
- `metadata_matching_all.json`
- `tag_matching.json`
- `summary_matching.json`

## Quick Reference

| File | What it measures (one line) | Main fields to watch | Better direction |
| --- | --- | --- | --- |
| `type_matching.json` | Whether the model predicts the right artifact types per issue. | `precision`, `recall`, `f1`, `jaccard` | Higher is better |
| `metadata_matching.json` | Whether metadata values match human values for each `(type, field)`. | Hard: `precision/recall/f1/jaccard`; Soft: `soft_precision/soft_recall/soft_f1` | Higher is better |
| `metadata_matching_all.json` | Side-by-side comparison of metadata similarity backends. | `summary_by_metric`, per-backend `overall`, per-backend `macro_average` | Higher is better |
| `tag_matching.json` | Tag overlap quality, conditioned on types that appear in both sides. | `precision`, `recall`, `f1`, `jaccard` | Higher is better |
| `summary_matching.json` | Semantic similarity of aggregated summaries for matching types. | `codebert.cosine`, `bertscore.f1`, `bleurt.score`, coverage and penalized blocks | Higher is better (`bleurt` may be negative; less negative is better) |

---

## 1) `type_matching.json`

### What it measures

How well the LLM predicts **which artifact types exist** (set membership), ignoring summary text and metadata details.

Example type names: `problem_statement`, `question`, `task_assignment`.

### Core idea

For each issue, compare two sets:

- Human type set: `H`
- LLM type set: `L`

Compute set overlap metrics.

### Equations

Let:

- `TP = |H ∩ L|`
- `FP = |L \ H|`
- `FN = |H \ L|`

Then:

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2 * Precision * Recall / (Precision + Recall)`
- `Jaccard = TP / (TP + FP + FN)`

### How to read it

- High precision, low recall: LLM predicts few types, but mostly correct.
- Low precision, high recall: LLM predicts many extra types.
- Jaccard is stricter than F1 for set overlap.

---

## 2) `metadata_matching.json`

### What it measures

How well LLM metadata values match human metadata values **within each type and field**, using fuzzy phrase similarity.

This goes beyond type presence and checks field-level content.

### Matching process (simplified)

For each `(type, field)` pair:

1. Build lists of human values and LLM values.
2. Compute phrase similarities between all pairs.
3. Greedily match pairs from highest similarity to lowest.
4. A pair is a hard match if similarity `>= threshold`.

### Hard metrics equations

Using matched pairs as `TP` and unmatched LLM/Human values as `FP/FN`:

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2PR / (P + R)`
- `Jaccard = TP / (TP + FP + FN)`

### Soft metrics equations

Soft metrics do not threshold; they average best similarities.

Let:

- `sim(a, b)` be phrase similarity in `[0, 1]`
- `best_LLM(x) = max_{h in Human} sim(x, h)`
- `best_HUM(y) = max_{l in LLM} sim(y, l)`

Then:

- `SoftPrecision = avg_{x in LLM} best_LLM(x)`
- `SoftRecall = avg_{y in Human} best_HUM(y)`
- `SoftF1 = 2 * SoftPrecision * SoftRecall / (SoftPrecision + SoftRecall)`

### Similarity backends

In `metadata_matching_all.json`, the same metric is run with multiple similarity functions:

- `token_f1`
- `sequence_ratio`
- `token_jaccard`
- `char_3gram_jaccard`
- `token_containment`
- `max_all` (max of the above)

### How to read it

- Hard metrics answer: "Did we match values above threshold?"
- Soft metrics answer: "How semantically close are values, even if not exact?"

---

## 3) `tag_matching.json`

### What it measures

How well tags overlap for **types present in both human and LLM** in each issue.

This is tag-set overlap conditioned on matching type presence.

### Equations

For each compared type in an issue:

- Human tag set `T_H`
- LLM tag set `T_L`

Let:

- `TP = |T_H ∩ T_L|`
- `FP = |T_L \ T_H|`
- `FN = |T_H \ T_L|`

Then:

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2PR / (P + R)`
- `Jaccard = TP / (TP + FP + FN)`

### How to read it

- High type score but low tag score means type detection is okay, but detail labeling is weak.

---

## 4) `summary_matching.json`

### What it measures

How similar human vs LLM summaries are for matching types.

For each issue and matching type:

1. Aggregate all human summaries for that type into one text.
2. Aggregate all LLM summaries for that type into one text.
3. Score text similarity with multiple scorers.

### Scorers

- `codebert.cosine`: cosine similarity of pooled CodeBERT embeddings.
  - Range roughly `[-1, 1]`, usually `[0, 1]` in practice.
- `bertscore.{precision, recall, f1}`: token-level contextual similarity.
  - Range `[0, 1]`.
- `bleurt.score`: learned quality score from BLEURT model.
  - Can be negative; higher is better.

### New aggregate blocks in `overall`

- `matched_only_macro`: average quality over issues with at least one matching type.
- `all_issues_macro_raw`: includes zero-score issues with no matching types.
- `all_issues_macro_with_unmatched_penalty`: per-issue scores multiplied by type-recall coverage.
- `support_weighted_matched_only`: matched-only average weighted by type support.
- `coverage`: macro average of type-level coverage per issue.
- `penalized_overall`: `matched_only_macro` scaled by macro type recall.
- `per_type_matched_only_macro`: macro over types with at least one matched issue.
- `per_type_macro_with_unmatched_penalty`: per-type macro after recall penalty.

### Coverage equations

For each issue with human type count `Hc`, LLM type count `Lc`, matched count `Mc`:

- `CoveragePrecision = Mc / Lc`
- `CoverageRecall = Mc / Hc`
- `CoverageF1 = 2 * CoveragePrecision * CoverageRecall / (CoveragePrecision + CoverageRecall)`

Penalty scaling uses:

- `PenalizedScore = RawScore * CoverageRecall`

### Why penalized summaries matter

A model can have high semantic similarity on the few types it matches, while still missing many human types.
Penalized aggregates combine quality + coverage into one view.

---

## Practical interpretation order

Use this order when comparing models:

1. `type_matching.json` -> Can the model find the right kinds of information?
2. `tag_matching.json` and `metadata_matching.json` -> Can it label details correctly?
3. `summary_matching.json` -> Is the generated summary text semantically aligned?
4. Coverage + penalized aggregates -> Is text quality still good after accounting for missed types?

---

## Notes and caveats

- BLEURT is model-dependent and not directly comparable across different BLEURT checkpoints.
- High BERTScore/CodeBERT with low coverage means over-optimistic quality if coverage is ignored.
- Macro averages treat each unit equally; weighted averages emphasize high-support units.
- Always compare metrics together, not in isolation.

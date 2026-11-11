# Full-text query refinement

The query optimizer can check each candidate Dimensions query against real full-text
evidence. The agent only **proposes** a refined query; a human approves it; a one-click
submit then creates the real extraction task.

## The loop (optimizer Phase 3)

Enabled with `DATASET_AGENT_FULLTEXT_REFINEMENT_ENABLED=true` (default `false`). When
disabled, no probes run and no proposal is built, but Phase 3 still differs from earlier
releases: the starting variant, the false-positive denominator and the Dimensions call
budget changed with this feature. Bounded by `optimize_max_iterations`
(default 3).

For each iteration the current candidate is mapped to an extraction seed and **probed**:

1. `POST /pipelines/start` with `probe=true` and `probe_pull_cap` (default 300): the
   extraction pipeline pulls the top-cited hits for the candidate (the probe's pool) and
   publishes nothing. `GET /pipelines/{id}/status` is polled until the search is done.
2. **Top-up batches.** The agent reads `GET /pipelines/{id}/llm-results` (every pool
   publication, in citation order, each with its `fulltext` outcome) and sends the next
   `fulltext_batch_size` (default 30) publications whose outcome is still `not_attempted`
   as explicit `publication_ids` to `POST /pipelines/{id}/full-publications/analyze`
   (`fulltext_source=open`, `steps={"validate_datasets": true}`). Open-access full text is
   fetched and only the dataset-validation LLM step runs. A publication without open full
   text is never analyzed, so the agent repeats with the next untried batch until
   `fulltext_target_delivered` (default 30) papers have full text, the pool is exhausted,
   or `fulltext_max_batches` (default 8) batches were sent. Each stop records its reason
   (`target_met`, `pool_exhausted`, `max_batches`); an OpenAlex budget miss is reported
   but does not stop the loop, because PMC and Europe PMC may still deliver.
3. A batch is done when every one of its publications has a full-text outcome in
   `llm-results` and `/status` `llm_analysis` is completed with nothing pending and
   `total_queued` (cumulative over the probe's batches) covering every delivered paper. `fulltext.selected` is not
   used: extraction overwrites it per analyze call while `attempted` accumulates.
   `probe_poll_timeout_seconds` applies to the search and to each batch.
4. The final `llm-results` gives per-publication `is_used`, `is_mentioned` and
   `text_quote`. `POST /pipelines/cleanup` removes the probe task once, also on failure.

When fewer than the target papers have full text, the iteration notes and the proposal
say so (for example `18/30 papers with full text (pool_exhausted; 212 attempted in 8
batch(es))`).

### Definitions

The probe judges the **search query**, so the question is "did the query find a paper
about this dataset?", not "does the paper use it?".

* match (genuine paper): the full text mentions the dataset, whether it uses it
  (`is_used == true`) or only mentions it (`is_mentioned == true`, `is_used == false`).
* false positive: the full text does not mention the dataset at all.
* Usage never decides a term change. A widely cited dataset is mentioned far more often
  than it is used; counting "mentioned, not used" as a false positive would drop the
  dataset's own official names.
* Papers with no delivered full text, not analyzed, or without a verdict are **excluded**
  from the full-text metrics; they only lower OA coverage.
* Primary metric: full-text precision = matches / evaluated. Also reported: used rate
  (used / evaluated), usage rate (used / matches), mention rate (equal to precision), OA
  coverage (delivered / attempted) and the abstract-based FP rate (secondary).
* A term may be changed only with full-text evidence. Terms with fewer than
  `refine_min_alias_papers` verified papers are `unverified` and never changed; an
  abstract-only FP signal can at most mark a term `suspect`.

### Pattern pass over all probe papers

The final `llm-results` read asks for `?contexts=true`: for each delivered paper,
extraction returns snippets (about 200 characters each side) around the query's aliases
and flag terms in the full text (title, text parts, reference titles) and the paper's
Dimensions fields of research. A paper that does not mention the dataset has no
validator quote, so these snippets are what show which other meaning of a term matched
it; they also count for per-term attribution (an alias is attributed to a paper when it
occurs in its quote, title or snippets). Multi-word terms match across punctuation and
hyphens (`WR. Dataset`), as a Dimensions phrase search does. When no alias occurs in a
paper's text at all, extraction returns `partial` snippets around the leading acronym of
each multi-word alias (`WR` for `WR dataset`), and the paper is attributed to that alias.

One LLM call sees the papers together, up to 40 per side with two snippets each. It
groups the false positives by the other meaning that pulled them in (for example "UI =
user interface"), naming the query term that caused each group and exclude phrases for
it, and reads the matching papers for shared context (flags) and alias forms. The
groups are written to the iteration notes and the proposal. Fields of research that
hold at least half of the false positives (and at least
`refine_category_min_fp_papers`) but at most 10% of the matches are reported as a
manual suggestion; they are never applied.

### Refinement actions

Deterministic rules (thresholds `refine_drop_fp_share`, `refine_demote_fp_share`):

* drop an alias whose verified FP share is at least the drop threshold (never the last alias);
* demote a safe alias (standalone) to risky (requires flag terms) at the demote threshold.

LLM proposals (validated against the quotes before acceptance):

* exclude terms: must occur in the title or full-text snippets of at least
  `refine_min_exclude_papers` false positives and in no matching paper (quote, title or
  snippets, singular forms included), and must remove at least one current result (a
  Dimensions count of the query AND the term; the diff states how many). At most
  `refine_max_exclude_checks` candidates are checked per pass, most false-positive hits
  first; checks stop when the call budget is down to `refine_impact_budget_reserve` (kept
  for the next gate evaluation), and an exclude whose impact is unknown is rejected. It
  must not overlap the query's own terms (an exclude inside or containing an alias, or inside
  a flag, is rejected because `NOT` would drop the query's own matches)
  (the diff reason states how many matches were quotes and how many title-only);
* flag terms: must occur in the quote or snippets of at least two matching papers and at
  least `refine_flag_min_share` of them (titles never count), and must be specific: a single
  lower-case word ("state", "wage") is rejected, and any other flag needs a Dimensions
  full-data count between 1 and `refine_flag_max_hits` (an unknown count rejects it; at
  most `refine_max_flag_checks` counts per pass);
* new aliases: must occur in a matching evidence quote (titles never count) and be absent
  from the query; Dimensions
  counts are re-probed (zero-hit aliases are rejected; very broad ones become risky).

Every rejected candidate is logged with its reason and summarized in the iteration
notes ("rejected exclude candidates: …").

The loop stops when full-text precision reaches `1 - optimize_fp_threshold_pct/100` on at
least `fulltext_min_evidence` evaluated papers, when nothing changes, after the iteration
cap, or when a gate evaluation or a probe fails (an evaluation error is not read as 0% FP;
it ends the loop and caps confidence at LOW). The proposal uses the iteration with the
best full-text precision among those with at least `fulltext_min_evidence` evaluated
papers (ties go to the earlier iteration); each iteration notes its `n_evaluated`. A
refinement produced on the last iteration is not probed and is therefore not proposed.

With full text enabled, the reported confidence never exceeds what the selected
iteration's full-text precision supports: HIGH needs precision at or above
`1 - optimize_fp_threshold_pct/100`, MEDIUM needs at least 0.5, anything lower is LOW. When
no iteration reaches `fulltext_min_evidence` (including a probe that failed on the first
iteration), the proposal equals the initial query, its notes say why, and confidence is LOW.

## Seed mapping

safe aliases -> `standalone_names`; risky aliases -> `dataset_names` (+ `flag_terms`);
excludes -> `exclude_terms`. Shapes the buckets cannot express (an explicit raw clause, or
safe-only candidates) use `dimensions_query`; every alias must appear in it, otherwise the
mapping fails before anything is sent. A term containing a double quote also fails the mapping
(the server rejects it).

## Proposal, approve, submit

The optimizer result carries `proposal` (also stored as `proposal-<id>.json` in
`output_dir`): initial vs proposed seed, per-term diff (action, status, reason, evidence
quotes), per-iteration metrics and the ready `/pipelines/start` payload (`probe=false`).

```
GET  /proposals/{id}
POST /proposals/{id}/approve      # optional body {"approved_by": "..."}
POST /proposals/{id}/submit       # 409 unless approved; one submit per proposal
```

The target extraction instance is `DATASET_AGENT_SUBMIT_TARGET` (`dev` by default).
`prod` has to be set explicitly and the proposal must still be approved.

## Settings (`DATASET_AGENT_` prefix)

| Setting | Default | Meaning |
|---|---|---|
| `fulltext_refinement_enabled` | `false` | Turn the loop on |
| `fulltext_target_delivered` | 30 | Papers with delivered full text a probe aims for |
| `fulltext_batch_size` | 30 | Publications per analyze batch (next untried, by citations) |
| `fulltext_max_batches` | 8 | Hard stop on batches per probe |
| `fulltext_top_n` | unset | Deprecated; when set (and `fulltext_batch_size` is not) it is the batch size |
| `probe_pull_cap` | 300 | Papers pulled by the probe search (the pool batches draw from) |
| `probe_poll_timeout_seconds` / `probe_poll_interval_seconds` / `probe_poll_max_interval_seconds` | 3600 / 3 / 20 | Polling; the timeout applies to the search and to each batch |
| `fulltext_min_evidence` | 3 | Min evaluated papers before anything changes |
| `refine_min_alias_papers` | 3 | Min verified papers per alias |
| `refine_drop_fp_share` / `refine_demote_fp_share` | 0.8 / 0.5 | Deterministic thresholds |
| `refine_max_new_aliases` | 3 | Cap on added aliases per iteration |
| `refine_min_exclude_papers` | 2 | Min false positives carrying a new exclude |
| `refine_max_exclude_checks` / `refine_impact_budget_reserve` | 5 / 8 | Impact checks per pass / Dimensions calls kept for the next gate evaluation |
| `refine_flag_min_share` | 0.2 | Min share of matching papers carrying a new flag |
| `refine_flag_max_hits` / `refine_max_flag_checks` | 500000 / 5 | Max Dimensions hits for a new flag / flag counts per pass |
| `optimize_suffix_short_acronym_safe` | `false` | Keep a suffixed short acronym ("WR dataset") safe instead of risky (flag-gated) |
| `refine_category_min_fp_papers` | 3 | Min false positives in a field of research before it is reported |
| `probe_target`, `submit_target` | `dev` | Extraction instance for probes / submit |
| `extraction_{dev,prod}_url`, `_user`, `_password` | empty (must be set; extraction requires HTTP Basic auth) | Per-target connection. Do not point it at dataset-agent's own port. |

## Benchmark

`scripts/benchmark_fulltext_refinement.py` compares initial and refined queries against
human `is_used` labels (precision, recall). It scores usage, which is stricter than the
loop's match precision; see its docstring for the input formats and how
to export human-validated data.

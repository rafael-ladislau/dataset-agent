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

1. `POST /pipelines/start` with `probe=true` and `probe_pull_cap` (default 200): the
   extraction pipeline pulls the top-cited hits for the candidate and publishes nothing.
2. When the search is done, `POST /pipelines/{id}/full-publications/analyze` with
   `fulltext_source=open`, `top_n` (default 30, flat by citations) and
   `steps={"validate_datasets": true}`: open-access full text is fetched and only the
   dataset-validation LLM step runs.
3. `GET /pipelines/{id}/status` is polled (backoff, `probe_poll_timeout_seconds`), then
   `GET /pipelines/{id}/llm-results` returns per-publication `is_used`, `is_mentioned`
   and `text_quote`.
4. `POST /pipelines/cleanup` removes the probe task, also on failure.

### Definitions

* genuine paper: `is_used == true`; false positive: `is_used == false`
  (`is_mentioned == false` is also a false positive).
* Papers with no delivered full text, not analyzed, or without a verdict are **excluded**
  from the full-text metrics; they only lower OA coverage.
* Primary metric: full-text precision = genuine / evaluated. Also reported: usage rate
  (genuine / mentioned-or-used), mention rate, OA coverage (delivered / sampled) and the
  abstract-based FP rate (secondary).
* A term may be changed only with full-text evidence. Terms with fewer than
  `refine_min_alias_papers` verified papers are `unverified` and never changed; an
  abstract-only FP signal can at most mark a term `suspect`.

### Refinement actions

Deterministic rules (thresholds `refine_drop_fp_share`, `refine_demote_fp_share`):

* drop an alias whose verified FP share is at least the drop threshold (never the last alias);
* demote a safe alias (standalone) to risky (requires flag terms) at the demote threshold.

LLM proposals (validated against the quotes before acceptance):

* exclude terms: must occur in a false-positive quote or title and in no genuine one
  (the diff reason states how many matches were quotes and how many title-only);
* flag terms: must occur in the evidence quotes of at least two genuine papers (titles
  never count); with fewer than two genuine papers no flag is added;
* new aliases: must occur in a genuine evidence quote (titles never count) and be absent
  from the query; Dimensions
  counts are re-probed (zero-hit aliases are rejected; very broad ones become risky).

The loop stops when full-text precision reaches `1 - optimize_fp_threshold_pct/100`, when
nothing changes, or after the iteration cap. The proposal uses the iteration with the best
full-text precision (ties go to the earlier iteration); a refinement produced on the last
iteration is not probed and is therefore not proposed.

## Seed mapping

safe aliases -> `standalone_names`; risky aliases -> `dataset_names` (+ `flag_terms`);
excludes -> `exclude_terms`. Shapes the buckets cannot express (an explicit raw clause, or
safe-only candidates) use `dimensions_query`; every alias must appear in it, otherwise the
mapping fails before anything is sent.

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
| `fulltext_top_n` | 30 | Top-cited papers analyzed per probe |
| `probe_pull_cap` | 200 | Papers pulled by the probe search |
| `probe_poll_timeout_seconds` / `probe_poll_interval_seconds` / `probe_poll_max_interval_seconds` | 900 / 3 / 20 | Polling |
| `fulltext_min_evidence` | 3 | Min evaluated papers before anything changes |
| `refine_min_alias_papers` | 3 | Min verified papers per alias |
| `refine_drop_fp_share` / `refine_demote_fp_share` | 0.8 / 0.5 | Deterministic thresholds |
| `refine_max_new_aliases` | 3 | Cap on added aliases per iteration |
| `probe_target`, `submit_target` | `dev` | Extraction instance for probes / submit |
| `extraction_{dev,prod}_url`, `_user`, `_password` | dev URL `http://127.0.0.1:8000` | Per-target connection |

## Benchmark

`scripts/benchmark_fulltext_refinement.py` compares initial and refined queries against
human `is_used` labels (precision, recall); see its docstring for the input formats and how
to export human-validated data.

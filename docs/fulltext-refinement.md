# Full-text query refinement

The query optimizer can check each candidate Dimensions query against real full-text
evidence. The agent only **proposes** a refined query; a human approves it; a one-click
submit then creates the real extraction task.

## The loop (optimizer Phase 3)

Enabled with `DATASET_AGENT_FULLTEXT_REFINEMENT_ENABLED=true` (default `false`). When
disabled, no probes run and no proposal is built, but Phase 3 still differs from earlier
releases: the starting variant, the false-positive denominator and the Dimensions call
budget changed with this feature. Bounded by `optimize_max_iterations`
(default 2).

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
   The cleanup answer (202) is a list with one item per task id; the agent treats the
   cleanup as failed when the call fails or any item has `status: "error"` or a
   `status_code` of 400 or more.

**AI-model probes (model task v2.1).** `run_probe_topup(..., entity="ai_model")` posts the
model seed plus `probe` and `probe_pull_cap` to `POST /ai-models/start` (unknown keys are
rejected with 422, so `seed_type` is never sent) and analyzes with
`steps={"validate_ai_models": true}`. Status, `llm-results` and cleanup use the same
`/pipelines/...` routes. The `llm-results` body carries `seed_type: "ai_model"` and per
publication `validated_ai_models` rows (`name`, `is_mentioned`, `is_used`, `text_quote`,
`match_type`, `usage_type`, and since v2.1 `mention_location` and `matched_name`); the
shared example is `tests/fixtures/llm-results-model-task-v2.1.json`. `build_evidence` reads these rows when
`seed_type` is `ai_model`: an empty list or only `validation_error` / `processing_error`
rows is no verdict; `usage_type: "cited"` is a mention. The references-only override below
does not apply to models (a model name found only in the reference list is a mention);
such papers are counted as `cited_only_count` / `cited_rate` in the iteration metrics
(null for datasets). A served `mention_location` of `text` or `references_only` decides
cited only; when it is missing or `unknown`, a served `usage_type` decides (`cited`), and
without one a genuine model paper whose name snippets are all in `references` is counted
as cited only; flag snippets are ignored. Model iteration metrics also carry
`paper_details`: one entry per evaluated paper with `id`, `title`, `verdict`, `location`,
`cited_only` and `matched_name` (null for datasets).

When fewer than the target papers have full text, the iteration notes and the proposal
say so (for example `18/30 papers with full text (pool_exhausted; 212 attempted in 8
batch(es))`).

### Definitions

The probe judges the **search query**, so the question is "did the query find a paper
about this dataset?", not "does the paper use it?".

* match (genuine paper): the full text mentions the dataset, whether it uses it
  (`is_used == true`) or only mentions it (`is_mentioned == true`, `is_used == false`).
* false positive: the full text does not mention the dataset at all.
* references-only override (datasets only): a name found only in the reference list is not
  a mention. When the served `validated_datasets` row carries `bibliography_excluded: true`
  the pipeline cut the reference list before the judge read the paper, and the judge's
  verdict stands. Otherwise the verdict becomes `not_mentioned` only when every non-flag
  snippet is in `references` and the validator quote is empty or lies inside one single
  references snippet, both normalized (case-folded; `|`, `…` and `...` as whitespace;
  citation markers such as `[62]`, `(ref 62)`, a leading `62.` and superscript digits
  removed; digits glued to a word's end removed; whitespace collapsed). Anything else keeps
  the verdict. The scope check's references-only rule uses the same quote test.
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
each multi-word alias (`WR` for `WR dataset`). The paper is attributed to the alias with
that acronym and the fewest words (the most ambiguous phrase), whichever alias extraction
tagged. Only an alias whose own leading word is an acronym (2-6 alphanumeric characters,
at least one letter, upper case or without vowels; `head_acronym()`, the same rule as in
the extraction pipeline) can own a `partial` snippet, so with the aliases `Add Health` and
`ADD Health Study` a snippet tagged `ADD Health Study` is credited only to the latter.

Matching normalizes text like extraction does: case, punctuation and `_` all separate
words (`WR_dataset` matches `WR dataset`). Extraction cuts snippets on word
boundaries, so the words next to a `…` are whole words and count like any other. A snippet whose part is `title`
counts for attribution but, like the paper title, never for flags or new aliases. The
evidence quote shown for a term is the validator quote, or, when that is empty, the
snippet around that term, so exclude and alias evidence is never empty.

One LLM call sees the papers together, up to 40 per side with two snippets each. It
groups the false positives by the other meaning that pulled them in (for example "UI =
user interface"), naming the query term that caused each group and exclude phrases for
it, and reads the matching papers for shared context (flags) and alias forms. The
groups are written to the iteration notes and the proposal, with the LLM's paper count
next to the number of false-positive papers whose title or snippets really contain the
group's cause term ("~6 per LLM; cause term 'NHIS' occurs in 4 false positive(s)": the
second number is how many false positives contain the cause term, not per-group evidence). The prompt asks for flag terms that are
proper names (agency, program, product), not lower-case context words. Fields of research that
hold at least half of the false positives (and at least
`refine_category_min_fp_papers`) but at most 10% of the matches are reported as a
manual suggestion, and only when at least that many matching papers exist to compare with; they are never applied.

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
  least `refine_flag_min_share` of them (titles never count), and must be a proper name
  (agency, program or product): a term with any lower-case word other than a joiner such as
  "of" or "the" is rejected ("state", "administrative data", "state UI"; "Bureau of Labor
  Statistics" passes), and a proper name needs a Dimensions
  full-data count between 1 and `optimize_flag_max_hits` (an unknown count rejects it; at
  most `refine_max_flag_checks` counts per pass); a bare short acronym is rejected when the
  query already has flags (`optimize_drop_short_acronym_flags`), and so is a flag that is a
  phrase of a flag-gated alias, current or accepted as risky in the same pass;
* proposals are deduplicated case-insensitively before any Dimensions count (a flag
  proposed as "Census Bureau" and "census bureau" costs one call), and proposals already
  in the query are skipped and noted as "already in the query";
* new aliases: must occur in a matching evidence quote (titles never count) and be absent
  from the query; Dimensions
  counts are re-probed (zero-hit aliases are rejected; very broad ones become risky). A
  `<short acronym> <generic suffix>` form ("UI data", "WR dataset"), a bare short acronym and a
  rhetorical phrase of ultra-common words ("All of Us") are never added.
* a term proposed both as a new alias and as a flag is judged as a flag only.

Flag and new-alias counts (and exclude impact checks) share the reserve: once the call
budget is down to `refine_impact_budget_reserve` they fail as "hit count unknown" instead
of reading as "0 Dimensions hits". Phase 1 counting is unchanged. When the shape rule
moves a safe `<short acronym> <generic suffix>` alias to risky, its `alias_counts` entry
is relabeled risky and the `suffix_risky` note appears once.

Every rejected candidate is logged with its reason and summarized in the iteration
notes ("rejected exclude candidates: …").

The loop stops when full-text precision reaches `1 - optimize_fp_threshold_pct/100` on at
least `fulltext_min_evidence` evaluated papers, when it meets the confidence bar
(`fulltext_recall_min_precision` on at least `fulltext_recall_min_evidence` evaluated papers,
0.80 on 10 by default: further refinement is not worth another probe), when nothing changes, after the iteration
cap, or when a gate evaluation or a probe fails (an evaluation error is not read as 0% FP;
it ends the loop and caps confidence at LOW). The proposal uses the iteration with the
best full-text precision among those with at least `fulltext_min_evidence` evaluated
papers (ties go to the earlier iteration); an iteration that meets the confidence bar
(`fulltext_recall_min_precision` on at least `fulltext_recall_min_evidence` papers) is
preferred over one that does not; each iteration notes its `n_evaluated`. A
refinement produced on the last iteration is not probed and is therefore not proposed.

With full text enabled, confidence comes from the selected iteration's full-text precision
and sample size only; the abstract FP rate stays in the notes and in the iteration metrics.
HIGH needs precision at or above `1 - optimize_fp_threshold_pct/100` (0.90), MEDIUM needs at
least `fulltext_recall_min_precision` (0.80), and both need at least
`fulltext_recall_min_evidence` (10) evaluated papers in the selected iteration. Confidence is
LOW on a gate error, when the first probe fails, or when no iteration reaches the evidence
floor (`fulltext_min_evidence`); in the last two cases the proposal equals the initial query
and its notes say why. A probe failure in a later iteration ends the loop: the best earlier
iteration is proposed with its own confidence, and the recall check is skipped. After a recall promotion the precision used is the lower of the
iteration's and the complement's.

For a dataset the resource map splits (the broad program query), the release bar is the
unrelated share instead of precision: MEDIUM, the early stop and the selection preference need
an unrelated share of at most `fulltext_split_unrelated_max` (0.20) on at least
`fulltext_recall_min_evidence` (10) evaluated papers. Each evaluated paper falls in one of four
classes (`PubVerdict.four_way`): `data` or `program_only` (genuine, by scope label),
`references_only` (turned to `not_mentioned` by the references-only override or the scope
check's references-only rule) and `unrelated` (the rest, including a paper the judge itself
called `not_mentioned` whose snippets all sit in the reference list). The split as it was
before (such a paper counted as references-only) is kept as `PubVerdict.four_way_legacy`,
and its unrelated share is reported as `unrelated_share_legacy` and in the release note. Data, program-only and
references-only papers all count toward the bar, because the judge removes citers and
program-only papers downstream. HIGH still needs precision of at least 0.90. Every other
dataset keeps the precision bar unchanged. The record states the bar (`release_bar`,
`release_bar_met`), and for a split dataset the four-way split with counts, shares and Wilson
95% intervals (`fulltext_four_way`) and the data-only precision (`fulltext_data_precision`);
the note `full-text release bar: ...` repeats them.

HIGH is capped at MEDIUM, with a note saying why, in two cases: the selected iteration's
probe pool was exhausted before the target was reached (for example 16 of 30 papers), or the
recall check rejected a candidate only on the Wilson lower bound although its point
precision met `fulltext_recall_min_precision` (a precise addition it could not make). LOW
and MEDIUM are never changed by this cap.

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

## Recall check

The loop measures precision only, so the selected query can be precise but tiny: a program's
full name that sits in the flag terms never matches on its own (flags only gate risky aliases).
After the loop selects its best iteration, a recall step proposes adding missing canonical
names as **safe aliases**, only with full-text evidence.

1. **Candidates** (deterministic, no LLM): flag terms that name the dataset, the main names,
   and known aliases that are not safe now (risky, or dropped in phase 1). A flag counts as
   naming the dataset when a main name sits inside it, when it contains a known multi-word
   alias, when it shares a distinctive phrase of two or more words with a name or such an alias,
   or when it is/starts with/has the initials of a dataset acronym (so "All of Us Research
   Program" and "Medical Imaging and Data Resource Center" pass while NIH, NSF and "National
   Cancer Institute" do not). Head-position acronyms only, so the alias "All of Us NIH" does
   not make NIH related, and the acronym head of a multi-word alias counts only when that
   alias is safe ("UNAVCO geodetic data" must not make "UNAVCO" a name). The existing safety
   bar applies: proper names only, no short acronym, no acronym plus generic suffix, no
   all-generic phrase such as "Data Commons", nothing the refiner already dropped or demoted
   on full text, no bare rhetorical phrase.
2. **Count**: the Dimensions hits each candidate adds beyond the current query (at most three
   times the candidate cap); fewer than the minimum evidence is rejected, as is a count above
   500,000.
3. **Complement probe**: top-cited papers matching the candidate but not the current query
   (`("term") NOT (<current query>)`), one probe per candidate within the usual probe limits.
   The candidate is sent as the alias so the validator looks for it. Promoted only with
   precision >= `fulltext_recall_min_precision` on >= `fulltext_recall_min_evidence` papers;
   otherwise the run notes say why (including free-full-text shortfalls). A promotion that
   adds more than `fulltext_recall_expansion_ratio` times the selected query's count must also
   have a Wilson 95% lower bound of at least `fulltext_recall_min_precision` minus
   `fulltext_recall_expansion_margin`; otherwise it is rejected with "too uncertain". The
   reason of a promotion also says how many papers use the data. Later candidates are
   measured beyond the ones already promoted. A flag-gated name is probed as a plain alias
   with the safe names, the flags and the excludes excluded; a flag or main-name candidate on
   a flag-gated query whose terms contain DSL special characters cannot be expressed and is
   reported as not evaluated, before any count call. When a candidate fails only on that
   lower bound (its point precision met the bar), the complement is probed once more with
   twice the target (about 40 papers) and judged on that probe alone, while the Dimensions
   call budget has room; the note reads `recall check: 'X' re-probed on N papers after a
   lower-bound rejection`. The bound and the margin are unchanged.
4. **Output**: a `TermDiff` (`alias`, `add`) with reason `recall: complement precision X on N
   papers; +K Dimensions hits` and evidence quotes; the final query, proposal and
   `selected_variant` (`...+recall`) include the alias. The abstract FP rate is not
   re-evaluated for the widened query. Confidence uses the lower of the
   iteration's and the complement's precision (a complement accepted at 0.8 cannot yield HIGH).

Caveat: the probe samples top-cited open-access papers while the hit count covers all years, so
the precision is an extrapolation from a small sample; the reason string states the sample size.

## Settings (`DATASET_AGENT_` prefix)

| Setting | Default | Meaning |
|---|---|---|
| `fulltext_refinement_enabled` | `false` | Turn the loop on |
| `fulltext_target_delivered` | 30 | Papers with delivered full text a probe aims for |
| `fulltext_batch_size` | 30 | Publications per analyze batch (next untried, by citations) |
| `fulltext_max_batches` | 8 | Hard stop on batches per probe |
| `probe_pull_cap` | 300 | Papers pulled by the probe search (the pool batches draw from) |
| `probe_poll_timeout_seconds` / `probe_poll_interval_seconds` / `probe_poll_max_interval_seconds` | 3600 / 3 / 20 | Polling; the timeout applies to the search and to each batch |
| `fulltext_recall_enabled` | `true` | Recall check after the precision loop (needs the loop) |
| `fulltext_recall_max_candidates` | 2 | Complement probes per run (0 disables) |
| `fulltext_recall_min_precision` / `fulltext_recall_min_evidence` | 0.8 / 10 | Bar a complement probe must meet (precision; evaluated papers and added Dimensions hits) |
| `fulltext_recall_expansion_ratio` / `fulltext_recall_expansion_margin` | 1.0 / 0.1 | Large-expansion rule: above ratio x the selected query's count, the Wilson lower bound must reach min precision minus margin |
| `fulltext_recall_target_delivered` | 20 | Full texts a complement probe aims for (at least the min evidence) |
| `fulltext_min_evidence` | 3 | Min evaluated papers before anything changes |
| `refine_min_alias_papers` | 3 | Min verified papers per alias |
| `refine_drop_fp_share` / `refine_demote_fp_share` | 0.8 / 0.5 | Deterministic thresholds |
| `refine_max_new_aliases` | 3 | Cap on added aliases per iteration |
| `refine_min_exclude_papers` | 2 | Min false positives carrying a new exclude |
| `refine_max_exclude_checks` / `refine_impact_budget_reserve` | 5 / 14 | Impact checks per pass / Dimensions calls kept for the next gate evaluation |
| `refine_flag_min_share` | 0.2 | Min share of matching papers carrying a new flag |
| `optimize_flag_max_hits` / `refine_max_flag_checks` | 500000 / 5 | Max Dimensions hits for a flag (research and refiner; 0 = off) / flag counts per pass |
| `refine_category_min_fp_papers` | 3 | Min false positives in a field, and min matching papers, before a field of research is reported |
| `probe_target`, `submit_target` | `dev` | Extraction instance for probes / submit |
| `extraction_{dev,prod}_url`, `_user`, `_password` | empty (must be set; extraction requires HTTP Basic auth) | Per-target connection. Do not point it at dataset-agent's own port. |

## Benchmark

`scripts/benchmark_fulltext_refinement.py` compares initial and refined queries against
human `is_used` labels (precision, recall). It scores usage, which is stricter than the
loop's match precision; see its docstring for the input formats and how
to export human-validated data.

# Dataset-research agent suffix evaluation
_Exported on 6/12/2026 at 11:14:02 GMT-3 from Cursor (3.7.27)_

---

**User**

is the dataset-research agent checking if it's a good idea to add suffix like data, dataset, survey? If yes, how is it being done?

---

**Cursor**

Let me explore the codebase to understand how the research agent handles query suffixes.

I can see there's a `check_suffix_necessity` function in `query_heuristics.py` and a `_SUFFIX_CANDIDATES` tuple. Let me read the relevant files in detail.

Now let me check where `check_suffix_necessity` is actually called.

Let me also check the query optimization use case to see if it's called there.

Now I have a clear picture. Here's the answer:

---

**Yes**, the codebase has a function specifically designed to check whether adding suffixes like "data", "dataset", "survey" etc. is beneficial — but **it is currently not wired into any active pipeline**. It exists as dead code.

The function is `check_suffix_necessity` in `src/dataset_agent/adapters/query_heuristics.py`:

```195:223:src/dataset_agent/adapters/query_heuristics.py
async def check_suffix_necessity(
    dsl_port: DimensionsDslPort,
    alias: str,
    *,
    search_in: str = "full_data",
    ratio_threshold: float = 100.0,
) -> str | None:
    """If count(bare)/count(bare + suffix) > *ratio_threshold*, return best suffixed form."""
    from dataset_agent.adapters.dimensions_dsl import run_alias_count

    bare = (alias or "").strip()
    if not bare or len(bare) > 120:
        return None
    c_bare = await run_alias_count(dsl_port, bare, search_in=search_in)
    if c_bare <= 0:
        return None

    best: str | None = None
    best_ratio = 0.0
    for suf in _SUFFIX_CANDIDATES:
        suffixed = f"{bare} {suf}".strip()
        c_s = await run_alias_count(dsl_port, suffixed, search_in=search_in)
        if c_s <= 0:
            continue
        ratio = float(c_bare) / float(c_s)
        if ratio > ratio_threshold and ratio > best_ratio:
            best_ratio = ratio
            best = suffixed
    return best
```

The candidate suffixes are defined on line 28:

```28:28:src/dataset_agent/adapters/query_heuristics.py
_SUFFIX_CANDIDATES: tuple[str, ...] = ("dataset", "survey", "program", "data", "study")
```

**How it works (in theory):**

1. It takes an alias (e.g. "NHANES") and queries Dimensions for the **bare** count — how many publications match just `"NHANES"`.
2. It then iterates over each suffix candidate, querying for the **suffixed** form (e.g. `"NHANES dataset"`, `"NHANES survey"`, etc.).
3. It computes the ratio: `count(bare) / count(bare + suffix)`.
4. If this ratio exceeds the `ratio_threshold` (default 100), it means the bare term is overwhelmingly noisy compared to the suffixed form — the bare term pulls in ~100x more results than the suffixed version, suggesting the suffix dramatically narrows to relevant publications.
5. It returns the best suffixed form (highest ratio above threshold), or `None` if no suffix qualifies.

**The problem: it's never called.**

Searching for `check_suffix_necessity` across `src/`, the only hit is the function definition itself. Neither `DatasetResearchUseCase` nor `QueryOptimizationUseCase` invokes it. The only other reference is in `tests/test_query_heuristics.py`, which has a unit test for it.

So the mechanism exists and has a clear design intent (detect when a bare alias is too generic and would benefit from a disambiguating suffix), but it's not integrated into either the research pipeline or the query optimization pipeline.

---

**User**

but is it worth being called? where would you suggest it being called?

---

**Cursor**

Good question. Let me re-examine the optimization pipeline to give you a concrete recommendation.

Here's my assessment:

**Yes, it's worth calling — but only in the query optimization pipeline, and with some constraints.**

### Why it's valuable

The current pipeline handles noisy aliases through V3/V4 variants, which AND the alias with `flag_terms`. But suffix disambiguation is a **complementary** and sometimes **superior** strategy because:

- Adding "dataset" or "survey" to a bare alias like `"NHANES"` keeps the query simple and readable (no dependency on having good flag terms)
- A suffixed alias that passes the ratio test is essentially "self-disambiguating" — it can be promoted from risky to safe, giving you more V1-eligible aliases and broader coverage
- It's especially powerful for rhetorical names (e.g. "All of Us" -> "All of Us dataset") where even V3/V4 may struggle

### Where to call it

The natural integration point is **Phase 1 of `QueryOptimizationUseCase.execute`**, right after an alias is classified as risky (lines 247-248). The flow would be:

```python
# Inside the alias counting loop, after risk classification:
if risk == "risky":
    # Try suffix disambiguation before giving up on bare form
    suffixed = await check_suffix_necessity(
        self._dsl, a, ratio_threshold=100.0
    )
    if suffixed:
        safe.append(suffixed)  # suffixed form is precise enough for V1
        notes_parts.append(f"suffix_promoted: {a!r} -> {suffixed!r}")
    else:
        risky.append(a)
```

This sits between the alias count (line 242) and the risky/safe bucketing (lines 247-250), inside the existing `for a in st0:` loop.

### Caveats and suggested guardrails

1. **API budget cost**: Each alias tested burns up to **5 extra Dimensions calls** (one per suffix candidate). With the current `optimize_max_dimensions_calls=64` budget, testing 3 risky aliases would consume ~15 calls — a significant chunk. I'd suggest:
   - Adding a setting like `optimize_suffix_check_max_aliases` (default 2-3) to cap how many risky aliases get suffix-tested
   - Early-exiting the suffix loop on the first match (the function already picks the best, but you could short-circuit after finding *any* passing suffix to save calls)

2. **Redundant bare count**: `check_suffix_necessity` re-fetches the bare count via `run_alias_count`, but Phase 1 already has it in variable `c`. To avoid burning an extra API call, either refactor the function to accept a pre-computed bare count, or accept the redundancy (one extra call per risky alias).

3. **Not in the research pipeline**: `DatasetResearchUseCase._run_pipeline_once` has no access to a `DimensionsDslPort` — it's pure LLM + web search. The suffix check is fundamentally a Dimensions signal, so it only belongs in the optimization path.

4. **Setting toggle**: Add something like `optimize_suffix_check: bool = True` to `Settings` so it can be disabled when API budgets are tight or the dataset names are already well-known.

### Summary

It fills a real gap: the pipeline currently has no mechanism to *rescue* a risky alias by appending a disambiguating keyword. V3/V4 handle risky aliases with AND-ed flag terms, but suffix promotion is cheaper (no flag terms needed) and produces simpler queries. The cost is moderate (up to 5 Dimensions calls per risky alias), which is worth it given it can shift aliases from the risky bucket into safe — directly improving V1 coverage.

---

**User**

is there any other dead code like this one wating to be integrated to the pipeline?

---

**Cursor**

Let me do a systematic scan of defined functions vs. their usage across the codebase.

Now let me verify which of these are actually called from the pipelines vs. only from tests.

Here's a clear picture. There are **8 dead functions** (besides `check_suffix_necessity`) that are defined but never called from any pipeline code. They're grouped into three thematic clusters:

---

### 1. FP Analysis enrichment (`fp_analysis.py`) — 4 functions

| Function | What it does |
|---|---|
| `identify_fp_domains` | LLM: maps collision domains and their indicator keywords for a risky alias (e.g. "NHIS" might collide with nursing/informatics domains) |
| `generate_flag_terms_for_alias` | LLM: produces context terms that confirm an alias genuinely refers to the target dataset |
| `generate_domain_positive_keywords` | LLM: title-level override keywords that strongly signal the target dataset |
| `web_search_alias_meanings` | Web search + LLM: discovers alternative real-world meanings of an alias to derive exclude terms |

**Where they'd fit:** All four belong in **Phase 3 of `QueryOptimizationUseCase`** (the FP analysis phase, ~lines 365–570). Right now, the FP analysis relies on keyword-hit heuristics (`classify_title`), abstract scanning (`scan_abstracts_for_aliases`), and LLM-based snippet classification (`classify_alias_mention`). These four functions would **enrich the FP keywords dictionary** that currently is naively built from the risky alias list itself (line 365: `fp_keywords = {"risky": [r.lower() for r in risky...]}`).

The integration would look like: for each risky alias, call `identify_fp_domains` to get a proper domain-to-keywords map (instead of just using the raw alias text), optionally call `web_search_alias_meanings` for additional web-grounded disambiguation, and use `generate_flag_terms_for_alias` / `generate_domain_positive_keywords` to build a positive-signal term set that could improve the V3/V4 flag term quality.

**Worth it?** `identify_fp_domains` and `web_search_alias_meanings` are the most valuable — they'd replace the crude `fp_keywords` dictionary with something semantically meaningful. `generate_flag_terms_for_alias` overlaps with the existing `refine_flag_terms_with_llm` from the research pipeline but is alias-specific, which could add precision for ambiguous acronyms. These are all LLM calls though, so budget-awareness matters.

---

### 2. Query enrichment (`query_heuristics.py`) — 3 functions

| Function | What it does |
|---|---|
| `detect_sub_products` | LLM over homepage HTML: discovers sub-product names (e.g. QWI, LODES, PUMS from a Census page) |
| `generate_multilingual_aliases` | LLM: alternate spellings, unaccented forms, translations of the dataset title |
| `strip_accents` | Pure utility: ASCII-folds accented characters |

**Where they'd fit:** `detect_sub_products` belongs in **Phase 1 of `QueryOptimizationUseCase`** — the discovered sub-products could be added to the alias list before counting. `generate_multilingual_aliases` could sit either in the research pipeline's Step 5 (alias generation in `_run_pipeline_once`) or in Phase 1 of optimization. `strip_accents` is a helper for `generate_multilingual_aliases` deduplication.

**Worth it?** `generate_multilingual_aliases` is very valuable for non-English datasets (think Brazilian IBGE surveys, French INSEE, etc.) where authors cite the dataset in its original language. `detect_sub_products` is niche but useful for umbrella programs like Census LEHD or NCHS — it would expand the alias pool with names that the LLM alias step might miss because they require parsing the actual homepage.

---

### 3. Dimensions-based alias risk classification (`dataset_aliases.py`) — 1 function

| Function | What it does |
|---|---|
| `classify_alias_risk` | Async: queries Dimensions to compare `count(alias)` vs `count(full_name)` with a 10x ratio threshold |

**Where it'd fit:** This is an interesting one — `QueryOptimizationUseCase` already has `_alias_risk_label` (line 55), which is a **synchronous reimplementation** of the same logic using counts it already fetched. The docstring on line 56 even says: "Same rules as `classify_alias_risk` without extra Dimensions round-trips." So the optimization pipeline deliberately inlined a cheaper version to avoid redundant API calls.

**Worth it?** No — this one is genuinely superseded. The inlined `_alias_risk_label` does the same job without extra API calls. The async version would only matter if you needed standalone alias risk classification outside the optimization pipeline (e.g. in the research pipeline), but research doesn't talk to Dimensions during alias generation.

---

### 4. Query builder (`query_builder.py`) — 1 function

| Function | What it does |
|---|---|
| `build_readable_for_clause` | Returns a human-friendly version of the for-clause (no DSL escape backslashes) |

**Worth it?** This is a display utility, not a pipeline function. It would be useful if you ever add a UI/report view showing the selected query in readable form. Low priority but harmless to keep.

---

### Summary ranking by integration value

1. **`check_suffix_necessity`** — High value, discussed earlier (Phase 1, risky alias rescue)
2. **`identify_fp_domains`** — High value (replaces crude `fp_keywords` dict in Phase 3)
3. **`web_search_alias_meanings`** — High value (web-grounded FP disambiguation, complements `identify_fp_domains`)
4. **`generate_multilingual_aliases`** — Medium-high for non-English datasets (research Step 5 or optimization Phase 1)
5. **`detect_sub_products`** — Medium for umbrella programs (optimization Phase 1)
6. **`generate_flag_terms_for_alias`** — Medium, partially overlaps existing flag term refinement
7. **`generate_domain_positive_keywords`** — Low-medium, niche use in title-level override scoring
8. **`classify_alias_risk`** — Not worth integrating (superseded by `_alias_risk_label`)
9. **`build_readable_for_clause`** / **`strip_accents`** — Utility/display, integrate when needed

---

**User**

Plan how to implement 1, 2, 3, and 4.

---

**Cursor**

Let me re-read the key integration points in detail to build an accurate plan.

Now let me check the existing test patterns to know what to target.

Now let me check how the optimization use case receives `self._agent` — needed for the LLM-based functions.

Now let me check how the research pipeline passes `self._agent` — needed for #4 (multilingual aliases).

I have all the context I need. Let me create the plan.

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

I'll implement the plan step by step, starting with the suffix check integration.


Updating tests: disabling new features in existing tests for stability, then adding integration tests.


Fixing the two failing tests.


The 3 API test failures are pre-existing (mocked literature module missing `promote_dimensions_metrics_from_detail`) — unrelated to our changes. All new and updated tests pass.


All four integrations are implemented. Summary:

### 1. Suffix check (`check_suffix_necessity`)
- Added optional `bare_count` to skip a redundant Dimensions call
- **Phase 1** of `QueryOptimizationUseCase`: risky aliases can be promoted to `safe` with a suffixed form (e.g. `"All of Us dataset"`)
- Settings: `optimize_suffix_check_enabled`, `optimize_suffix_check_max_aliases`

### 2. FP domains (`identify_fp_domains`)
- **Phase 3**: builds `fp_keywords` from LLM domain analysis instead of only lowercased risky aliases
- Falls back to the old dict when the setting is `0` or LLM returns nothing
- Setting: `optimize_fp_domains_max_aliases`

### 3. Web disambiguation (`web_search_alias_meanings`)
- **Phase 3**: runs after FP domains; enriches `fp_keywords` and seeds `exclusion_terms`
- Setting: `optimize_web_disambig_max_aliases`

### 4. Multilingual aliases (`generate_multilingual_aliases`)
- **Research Step 5**: appends alternate spellings/translations after alias filtering
- Setting: `research_multilingual_aliases`

### Tests
- Updated `tests/test_query_heuristics.py` (including `bare_count`)
- Added integration tests in `tests/test_query_optimization.py` and `tests/test_research_multilingual.py`
- **22/22** targeted tests pass; **122/125** full suite pass (3 existing API mock failures unrelated to this work)

---

**User**

test it O*NET again with the changes that were applied

---

**Cursor**

Running O*NET through the pipeline with the new integrations. Checking how to invoke it and the current environment.


Running the full optimize flow for O*NET (research + optimization with the new integrations). This will call the LLM and Dimensions API.


O*NET ran end-to-end successfully (~5 minutes). Here’s what the new integrations did:

### Research (`output/onet_research.json`)
- **Multilingual aliases** fired: `3 -> 5 names` (added variants like `O*NET OnLine`, `O-NET`)
- Final aliases: `O*NET`, `ONET`, `Occupational Information Network`, `O*NET OnLine`, `O-NET`
- Flag terms: ETA, Employment and Training Administration, U.S. Department of Labor, USDOL
- Dimensions preview: **5,463** publications (hybrid query with all aliases + flags)

### Optimization (selected **V1**)
| Metric | Value |
|---|---|
| Expected count | **27,018** articles |
| FP rate (V1) | **0%** (keyword proxy) |
| Confidence | **low** (scope ratio drove this) |
| Dimensions calls | **11** |

**Safe aliases (OR’d in V1):**
- `O*NET` (25,898)
- `Occupational Information Network` (6,860)
- `O*NET OnLine` (1,233)
- `O-NET` (25,898)

**Risky (kept out of V1):**
- `ONET` — only used in V3 with flag-term AND

**New Phase 3 features (from notes):**
- `fp_domains: 12 keywords across 3 domains` — `identify_fp_domains` ran for `ONET`
- `web_disambig: enriched fp_keywords from web search (2 aliases)` — `web_search_alias_meanings` ran
- Exclusion terms seeded: `database`, `job`, `occupation`, `labor`, `employment`

**Selected query:**
```
("O*NET" OR "Occupational Information Network" OR "O*NET OnLine" OR "O-NET")
```

### Suffix check did not run for `ONET`
`ONET` is 4 characters, so `apply_short_acronym_heuristic` moves it to **risky before counting**. Suffix promotion only runs for aliases classified risky *during* the count loop, not for pre-bucketed short acronyms. That’s why there’s no `suffix_promoted` note in the log.

### Quality note
V1 top titles still include obvious false positives (Alzheimer’s facts/figures, face-detection papers) — the 0% keyword FP rate is misleading; Signal 3 on V3 scored **60%** FP. V1 was still chosen because it had a better count/FP tradeoff under the current scoring.

If you want `ONET` to get suffix disambiguation too, the next step would be extending suffix check to aliases in the pre-risky short-acronym bucket. I can implement that if you’d like.

---

**User**

The number of publications are almost the same on v1 and v3 strategies, why v1 has 0% fp rate? It looks wrong. how could I fix it?

And why in v3 "ONET" shows up in the query out of the parenthesis of the alias? it also looks wrong, how cound I fix it?

and why no suffix got added to the aliases? What was the veridict?

---

**Cursor**

Tracing how FP rate is computed, how V3 builds its query, and why suffix promotion didn't apply.


Here are direct answers to all three points, tied to how the code actually behaves.

---

## 1. Why V1 shows 0% FP when counts are almost the same as V3

**The counts are close on purpose; the FP scores are not measuring the same thing reliably.**

| Variant | Count | Reported `fp_rate_pct` |
|--------|-------|-------------------------|
| V1 | 27,018 | **0%** |
| V3 | 28,008 | **60%** (~990 extra hits) |

V3 is only slightly broader because it adds  
`(ONET AND (ETA OR Employment and Training Administration OR …))`  
on top of the same safe OR block.

### Why V1 got 0% (buggy proxy, not “no false positives”)

Phase 3 uses a **keyword hit** proxy in `_fp_rate_and_noise_terms`:

```70:88:src/dataset_agent/application/query_optimization.py
def _fp_rate_and_noise_terms(
    titles: list[str],
    fp_keywords: dict[str, list[str]],
) -> tuple[float, list[str]]:
    ...
        m = classify_title(title, fp_keywords)
        risky_kw = m.get("risky") or []
        if risky_kw:
            hits += 1
```

After your new integration, `fp_keywords` comes from `identify_fp_domains` with keys like `"nursing"`, `"technology"`, etc. — **not** `"risky"`.  
`classify_title` can find matches under those keys, but the FP counter **only increments when the key is literally `"risky"`**. So keyword FP is always **0%** whenever domain-based keywords are used.

Your run’s notes confirm that:  
`signal3_fp_pct=60.0 keyword_fp_pct=0.0`

### Why V3 got 60% but V1 was still chosen

- **Signal 3** (LLM title scoring) runs only on the **primary** variant (highest count) — here **V3** — so V3 gets the real 60% estimate.
- **V1** never gets Signal 3; it only gets the broken keyword proxy → **0%**.
- Selection drops variants with `fp_rate_pct > 5%`:

```578:592:src/dataset_agent/application/query_optimization.py
            if fp > 5.0:
                continue
            score = float(cnt) * (1.0 - fp / 100.0)
...
        if not candidates:
            best = tested[0]
```

V3 is excluded (60% > 5%). V1 survives with a falsely perfect score and wins despite titles like Alzheimer’s and face-detection papers in `top_titles`.

So: **V1’s 0% is wrong** — it’s a measurement/selection bug, not evidence that V1 is clean.

### How to fix it (conceptually)

1. **Keyword FP:** Treat *any* domain match as an FP hit, e.g. sum matches across all keys in `fp_keywords`, not only `m.get("risky")`.
2. **Signal 3:** Run it on **every** variant you compare (or at least on V1 and V3), not only the highest-count one.
3. **Selection:** Don’t pick a variant whose keyword FP is 0% only because the dict has no `"risky"` key; require Signal 3 (or abstract sampling) before trusting low FP.
4. **Stricter gate:** If top titles on V1 clearly look off-topic, lower confidence or force V4 with NOT terms even when count is high.

---

## 2. Why `ONET` is “outside” the alias parentheses in V3

That’s **intentional V3 design**, not a DSL typo.

V3 is built as:

```75:89:src/dataset_agent/adapters/query_builder.py
    # V3 / V4: safe OR hybrid
    safe_expr = _or_group(st)
    ...
    if rk and fl:
        hybrid = f"({_or_group(rk)} AND {_or_group(fl)})"
        chunks.append(hybrid)
    body = " OR ".join(c for c in chunks if c)
```

So you get:

```text
( safe aliases OR … )  OR  ( ( ONET ) AND ( flag terms ) )
```

`ONET` is in **`risky`**, not **`safe`**, because of the short-acronym rule (≤4 characters must not appear as a bare OR term):

```69:87:src/dataset_agent/adapters/query_heuristics.py
def apply_short_acronym_heuristic(...):
    """Move short acronyms from *safe* into *risky* so V1 never ORs them alone."""
    ...
        if is_short_acronym(s):
            rk.append(s)
```

So `ONET` is **deliberately excluded** from the first parenthesis group and only allowed as `(ONET AND flags)`.

### If that looks wrong to you

- **Semantically:** It matches the spec: ambiguous short tokens need context (flags), not bare OR.
- **If you want `ONET` inside the main OR group:** That’s closer to **V2** (safe + risky OR together) — usually **more** noise, not less.
- **Better fix for O*NET:** Promote a suffixed safe form like `"ONET database"` / `"O*NET data"` into **safe** (see suffix section), so V1 can use it without the hybrid arm.

---

## 3. Why no suffix was added — what was the verdict?

**Suffix check never ran for `ONET`.** There was no “rejected suffix” verdict in the log.

Flow:

1. `apply_short_acronym_heuristic` moves `ONET` into `risky` **before** the counting loop.
2. Suffix promotion only runs inside the loop for aliases in `st0` that are classified **risky during counting**:

```252:271:src/dataset_agent/application/query_optimization.py
                if risk == "risky":
                    ...
                        suffixed = await check_suffix_necessity(...)
```

3. `ONET` is never in `st0` (it’s already in `rk0`), so `check_suffix_necessity` is **never called** for it.
4. No `suffix_promoted: ...` line appears in your logs → **no promotion attempted**.

`check_suffix_necessity` only returns a suffixed form when  
`count(bare) / count(bare + suffix) > 100` (default). Even if it had run, `ONET` might still fail unless something like `"ONET dataset"` is ~100× narrower than bare `ONET`.

### How to fix suffix behavior for cases like O*NET

1. **Run suffix check on the initial `rk0` list** (short-acronym risky aliases), not only on loop-classified risky aliases.
2. **Lower `ratio_threshold`** for short acronyms (e.g. 10 instead of 100) — spec mentions ~9× for `"ONET"` vs `"O*NET data"`.
3. **After promotion**, add `"ONET survey"` / `"O*NET data"` to **safe** and optionally remove bare `ONET` from **risky**.

---

## Summary

| Issue | Root cause | Fix direction |
|-------|------------|---------------|
| V1 @ 0% FP | Keyword scorer only looks at `fp_keywords["risky"]`; domain keys ignored; Signal 3 only on primary | Score all domains; run Signal 3 on all finalists |
| V1 chosen over V3 | V3 correctly 60% FP → excluded; V1 falsely 0% → wins | Same as above |
| `ONET` outside parens | V3 hybrid: risky ∧ flags, by design | Promote suffixed safe alias; don’t bare-OR `ONET` |
| No suffix | `ONET` pre-bucketed as short acronym; suffix path never invoked | Suffix-check `rk0` aliases; tune ratio for acronyms |

I’m in **Ask mode**, so I can’t change the code here. If you want these fixes implemented, switch to **Agent mode** and we can patch FP scoring, suffix coverage for short acronyms, and optionally variant selection logic.

---

**User**

implement the proposed fix

---

**Cursor**

Implementing the three fixes: FP scoring across all domain keys, Signal 3 on all finalist variants, and suffix checks for short-acronym risky aliases.


Implementing the three fixes: updating FP scoring, extending suffix checks to short-acronym aliases, and running Signal 3 on all finalist variants.


Implemented the three fixes from the O*NET analysis:

### 1. Keyword FP rate counts all domains
`_fp_rate_and_noise_terms` now treats a title as a false positive if **any** domain in `fp_keywords` matches (not only the `"risky"` key). Noise terms for NOT refinement are collected from all matched domains.

### 2. Signal 3 on every finalist variant
Signal 3 runs for **each** variant in the top-2 comparison (`top_idx`), not only the highest-count one. Notes look like `signal3_V1=… keyword_fp_pct=…` and `signal3_V3=…`.

### 3. Suffix check for short acronyms (`ONET`, etc.)
- Short acronyms in `rk0` are counted and run through suffix promotion **before** the main alias loop.
- Shared helper `_try_suffix_promote` uses:
  - **ratio 10** for acronyms ≤4 chars (`optimize_suffix_ratio_short_acronym`)
  - **ratio 100** for longer risky aliases (`optimize_suffix_ratio_threshold`)
- Promoted forms go to **safe**; bare forms stay in **risky** only if promotion fails.

### 4. Abstract FP sample gate
`fetch_top_n` for abstract scanning now runs when **any** FP keywords exist (`_fp_keywords_active`), not only when `fp_keywords["risky"]` is set.

### New settings
- `optimize_suffix_ratio_threshold` (default `100`)
- `optimize_suffix_ratio_short_acronym` (default `10`)

### Tests
- `test_fp_rate_counts_all_fp_domains`
- `test_optimize_short_acronym_suffix_via_rk0`
- **24/24** optimization/heuristics tests pass; **124/127** full suite (3 pre-existing API mock failures)

Re-run O*NET with:

```bash
dataset-research optimize "O*NET" --url "https://www.onetonline.org/"
```

You should see `suffix_promoted: 'ONET' -> …` when Dimensions supports it, non-zero keyword FP on V1 when domain keywords hit top titles, and Signal 3 scores on both V1 and V3 so selection is less likely to pick a high-FP variant because another one falsely shows 0%.

---

**User**

rerun it 

---

**Cursor**

O*NET rerun finished (~5 min). Here’s how it compares to the first run:

### Optimization result

| | Before fixes | After fixes |
|---|---|---|
| **Selected** | V1 | V1 |
| **Count** | 27,018 | **27,162** |
| **FP rate (V1)** | 0% (misleading) | **40%** (Signal 3) |
| **Risky aliases** | `ONET` | **none** |
| **V3 built?** | Yes (28,008 hits) | **No** (no risky aliases) |

**Selected query:**
```
("O*net" OR "Occupational Information Network" OR "Www.onetonline.org" OR "O*NET OnLine" OR "O-NET")
```

**Safe aliases:** `O*net`, `Occupational Information Network`, `Www.onetonline.org`, `O*NET OnLine`, `O-NET`  
**Risky:** empty

### What the fixes changed

1. **FP scoring is more honest** — V1 now reports **40%** FP via `signal3_V1=40.0` instead of a false **0%**. Notes also say: `No variant under 5% proxy FP; picked first probe result` and **confidence: low**, which matches the noisy top titles (Alzheimer’s, face detection, etc.).

2. **No V3 / no `ONET` outside parentheses** — This run’s research pass did not keep bare `ONET` as an alias (only `O*net`, `O-NET`, etc.). With **risky=0**, only **V1** was built, so there is no hybrid `(ONET AND flags)` arm.

3. **Suffix promotion** — No `suffix_promoted` in the log because **`ONET` never entered the risky bucket** (it wasn’t in the final alias list). Suffix logic only runs on aliases that exist and are short/risky.

4. **Phase 3 FP domains / web disambig** — Skipped when `risky` is empty, so `keyword_fp_pct=0.0` is expected here.

### Research (`output/onet_research.json`)
- Multilingual expansion: **3 → 5** names  
- Dimensions preview (hybrid with flags): **5,292** publications  
- Terms evaluation: **effective**, names 9 / flags 10  

### Takeaway
The scoring fix is working (40% FP on V1, low confidence). To fully exercise suffix + V3 for O*NET, the research step needs to retain **`ONET`** as an alias again (this run dropped it during refine/heuristic strip). If you want, we can adjust alias refinement so `ONET` is preserved for optimize, or re-run with explicit `dataset_names` including `ONET`.

---

**User**

Is it clear that the query with flag terms returns fewer false positives? Is the agent considering this case? Why is it not the winner query?

---

**Cursor**

Tracing how the optimizer compares V1 vs V3 and selects the winner.


Here’s how the agent actually treats “queries with flag terms” versus what you might expect from O*NET.

## Two different “flag term” queries

The codebase uses flags in **two different shapes**:

| Query | Shape | O*NET-ish effect |
|--------|--------|------------------|
| **Research / literature** | `(all dataset names) AND (all flag terms)` | Stricter, **fewer** hits (e.g. ~5.3k in the last research preview) |
| **Optimize V3** | `(safe aliases OR …) OR ((risky aliases) AND (flag terms))` | Broader: safe terms stay **unfiltered**; flags only gate **risky** tokens |

Literature DSL is built here:

```777:790:src/dataset_agent/adapters/literature.py
    dataset_clause = " OR ".join(f'\\"{t}\\"' for t in dataset_terms_query)
    if flag_terms_query:
        flag_clause = " OR ".join(f'\\"{t}\\"' for t in flag_terms_query)
        search_inner = f"(({dataset_clause}) AND ({flag_clause}))"
```

V3 is built only when **both** `risky` and `flag_terms` exist:

```173:186:src/dataset_agent/adapters/query_builder.py
    if rk and fl:
        out.append(
            (
                "V3",
                build_for_clause(
                    safe=st,
                    variant="V3",
                    risky=rk,
                    flag_terms=fl,
```

So: **yes**, “everything AND flags” usually reduces noise and count. **No**, that is not the same as the optimize winner (V1), which is **safe-only OR** with no flags on the main aliases.

The spec is explicit that V3’s job is to add precision **only for risky aliases**, not to replace the safe OR block (`dataset_query_agent_spec.md` § V3).

---

## Does the optimizer “consider” that V3 is more precise?

**Partially, but not the way you’re thinking.**

1. **V3 is probed** when there are risky aliases + flag terms.
2. **FP is scored on the whole variant’s top titles** (keyword domains + Signal 3), not on “papers that came only from `(risky AND flags)`”.
3. **Selection** uses one score for the full query:

```699:713:src/dataset_agent/application/query_optimization.py
        for ent in tested:
            cnt = ent.expected_count or 0
            fp = float(ent.fp_rate_pct or 0.0)
            if fp > 5.0:
                continue
            score = float(cnt) * (1.0 - fp / 100.0)
            pref = _VARIANT_ORDER.get(ent.label, 99)
            candidates.append((score, -pref, cnt, ent))

        if not candidates:
            best = tested[0]
            notes_parts.append("No variant under 5% proxy FP; picked first probe result.")
        else:
            best = max(candidates, key=lambda x: (x[0], x[1], x[2]))[3]
```

There is **no** step that says: “compare V1 vs V3 on FP **for the risky alias only**” or “prefer V3 when hybrid arm is cleaner.”

So for O*NET:

- Top titles on **both** V1 and V3 can still be dominated by **broad safe** matches (`O*net`, `O-NET`, etc.) — Alzheimer’s, face detection, employment chapters, etc.
- V3 can add ~1k extra hits via `(ONET AND flags)` while the **top 10 by citations** stay similar → similar-looking FP on the sample.
- The agent does **not** prove “flags reduce FPs” for the full result set; it only infers FP from a small title sample of whatever the full query returns.

Tie-break **does** prefer hybrid variants **among survivors** (`V4` > `V3` > `V1` via `_VARIANT_ORDER`), but only if `fp_rate_pct ≤ 5%`.

---

## Why V3 was not the winner

### Latest rerun (after fixes)

- **`risky: []`** — no `ONET` (or other short/risky token) in the final alias list.
- **Only V1 was built** (`n_variants=1`).
- V3 never entered the race.

So the optimizer never compared “with flags on risky” vs “safe only.”

### First rerun (before / around fixes)

- V1 ~27k, V3 ~28k (V3 slightly **larger**, as expected: safe OR **plus** hybrid arm).
- V1 showed **0%** FP (bug: keyword scorer ignored domain keys).
- V3 showed **~60%** FP on Signal 3 (only scored on the primary / higher-count variant then).
- V3 was **dropped** (`fp > 5%`); V1 was kept and won via fallback: *“No variant under 5% proxy FP; picked first probe result.”*

So V3 lost because **measured FP on the full query looked worse**, not because the agent measured “hybrid arm precision” and ignored it.

---

## Is it “clear” that flag terms reduce false positives?

**In design docs: yes.** V3/V4 are described as higher precision for risky acronyms.

**In the O*NET runs: not demonstrated.**

- Research **AND** query: fewer pubs → likely **stricter**, but that query is **not** what optimize selects.
- Optimize **V1**: no flags on safe aliases → **highest recall**, FP can still be high on top titles (40% Signal 3 on the last run).
- Optimize **V3**: only helps the **risky** slice; doesn’t clean the safe OR slice that drives most volume and top titles.

The agent is **not** currently answering: “Does `(dataset names AND flags)` beat `safe OR only` on precision?” It answers: “Among V1/V3/V4 probes, which has acceptable FP on top titles and highest count?”

---

## What would need to change (conceptually)

If you want “flag-gated query” to win when it’s truly cleaner:

1. **Always keep a risky token** (e.g. `ONET`) through research → optimize so V3/V4 are built.
2. **Score variants fairly** — Signal 3 on **all** finalists (you added this); ensure keyword FP uses domain keys (you added this).
3. **Compare the right queries** — e.g. probe or score:
   - V1 (safe only),
   - research-style `(all names AND flags)`,
   - V3 / V4 as today.
4. **Optional: FP only on risky-sourced hits** — e.g. titles from pubs matching risky alias in abstract, or a Dimensions query for just `(ONET AND flags)` vs bare `ONET`.
5. **When all variants fail the 5% gate**, don’t default to `tested[0]` (V1); pick **lowest FP** per spec § Phase 4 rule 3.

---

**Short answers**

| Question | Answer |
|----------|--------|
| Is it clear flags reduce FPs? | Clear in **spec**; **not** clear in O*NET runs — V1 safe-OR noise dominates; research AND-flags query isn’t the selected winner. |
| Is the agent considering this? | It **builds** V3 when `risky + flag_terms` exist and **scores** it, but does **not** isolate “hybrid arm” precision or prefer the research-style AND query. |
| Why isn’t it the winner? | Last run: **no V3** (`risky` empty). First run: V3 **failed FP gate** (60%) while V1 looked 0%; fallback chose V1. |

I’m in Ask mode — if you want selection logic or alias retention changed so V3/V4 or an AND-flags variant can win when appropriate, switch to Agent mode and we can implement that.

---

**User**

it looks like v3 queries are wrong. it should not be (safe aliases OR …) OR ((risky aliases) AND (flag terms)) it should be this instead: ((safe aliases OR …) OR (risky aliases)) AND (flag terms) what do you think?

---

**Cursor**

Checking the spec and how V3 is documented versus the research query shape.


Your proposed shape is **not wrong** for datasets like O*NET — but it is a **different variant** than what the code and spec call V3 today. Whether it should *replace* V3 depends on what you optimize for.

## What you have today

**Implemented V3** (`build_for_clause`, variant V3):

```text
(safe₁ OR safe₂ OR …)  OR  ((risky₁ OR …) AND (flag₁ OR flag₂ OR …))
```

**Your proposal:**

```text
((safe₁ OR safe₂ OR …) OR (risky₁ OR …))  AND  (flag₁ OR flag₂ OR …)
```

**Research / literature gate** (already closer to your idea):

```788:788:src/dataset_agent/adapters/literature.py
        search_inner = f"(({dataset_clause}) AND ({flag_clause}))"
```

There, `dataset_clause` is **all** names (safe + risky), and everything must also hit a flag term. That’s why O*NET research preview was ~**5.3k** pubs while optimize V1 was ~**27k**: flags constrain the **whole** query in research, but **not** the safe arm in V3.

---

## Why the current V3 exists (and when it’s right)

The spec models V3 after cases like **ADP**, where some aliases are already very specific:

```191:198:dataset_query_agent_spec.md
#### V3: Hybrid (Safe OR + Risky AND Flags)
...
("ADP payroll data" OR "ADP payroll" OR ...)
OR (("ADP") AND ("payroll" OR "employment report" OR ...))
```

The design assumption:

- **Safe** = long / product-specific strings → can OR alone without pulling in biology, etc.
- **Risky** = short homonyms → must be `(alias AND domain flags)`.

For ADP, bare `"ADP payroll data"` rarely means adenosine diphosphate; bare `"ADP"` does. So gating only the short token is enough.

---

## Why that feels wrong for O*NET

For O*NET, many “safe” aliases are **not** that safe in `full_data`:

- `O*net` / `O-NET` still match huge unrelated literature.
- `Occupational Information Network` is broad.
- So current V3 still behaves almost like **V1 OR a small hybrid extra**:

  - Noisy safe OR drives count and top titles.
  - `(ONET AND flags)` only adds a slice; it does **not** clean the safe OR arm.
  - FP scoring on the **full** result set still sees Alzheimer’s, CV papers, etc.

So your diagnosis is right: **for O*NET, current V3 does not mean “dataset names in flag context.”** It means “permissive name OR + optional gated acronym.”

---

## What your formula would do

**Pros**

1. **Aligns optimize with literature** — same boolean idea as `build_dimensions_literature_query`.
2. **Flags apply to every hit** — should cut FPs from broad safe strings, not only from `ONET`.
3. **Clearer story** — “publications that mention these names **and** sponsor/org context.”
4. **Matches intuition** — “query with flag terms should be stricter” becomes true for the variant you select.

**Cons**

1. **Recall loss** — papers that cite “O*NET” or “Occupational Information Network” without naming DOL/ETA/USDOL in searchable fields may drop out. That may be acceptable or not depending on corpus.
2. **Flag quality is load-bearing** — weak or missing `flag_terms` → empty or useless query; today, safe aliases still work without flags.
3. **Changes the V1–V4 ladder** — you’d effectively get:
   - **V1**: names only (max recall),
   - **Your shape**: names AND flags (precision),
   - **V2**: names including risky, no flags (diagnostic),
   - **V4**: names AND flags NOT exclusions.
   
   That’s coherent, but it’s **not** the spec’s current V3; it’s closer to “validated literature query” as an optimize variant.
4. **May over-filter ADP-style datasets** where safe phrases are already precise — global AND might be unnecessary and shrink good recall.

---

## Recommendation

I would **not** silently replace current V3 in `build_for_clause` with global AND for everyone. I would treat your formula as a **first-class variant** (or a conditional branch), e.g.:

| Variant | Shape | Role |
|--------|--------|------|
| **V3** (keep) | `safe OR (risky AND flags)` | When safe aliases are truly specific (ADP-style). |
| **V3′ / V6 / “V3G”** | `(safe OR risky) AND flags` | When safe aliases are still noisy (O*NET-style); matches literature gate. |

Then let Phase 2 probe **both** and Phase 4 pick by count + FP (with Signal 3 on all finalists), instead of assuming one hybrid shape fits all datasets.

**Heuristic trigger** (optional): if any “safe” alias still has very high Dimensions count vs full name (e.g. ratio &gt; 10×, same rule as risky), build the global-AND variant even for names in the safe bucket.

---

## Direct answers

| Question | Answer |
|----------|--------|
| Is current V3 “wrong”? | Wrong **for O*NET-like noise**; still **right for ADP-like** highly specific safe phrases per spec. |
| Should it be `((safe OR risky)) AND flags`? | **Yes, as an additional (or alternate) variant**, and it should match what literature already does. |
| Why isn’t that the winner today? | Optimize doesn’t build that shape; literature does, but optimize selects V1 safe-OR only. |

If you want this implemented, switch to Agent mode — the change is mainly in `build_for_clause` / `default_variant_build_order`, plus tests and a spec note that V3 split vs global-AND serve different alias-quality regimes.

---

**User**

but it looks like the safe aliases are not safe, if they are the same as v1 variant, ant he fp rate is 40%, it should not be considered safe.

---

**Cursor**

Tracing how "safe" is assigned today — it's likely not based on measured FP rate.


You’re right to push back on the label **“safe.”** In this codebase it does **not** mean “low false-positive rate.” It means something much narrower — and for O*NET that gap shows up exactly as you describe.

## What “safe” actually means today

Phase 1 only uses **Dimensions hit counts**, not measured FP:

```60:68:src/dataset_agent/application/query_optimization.py
def _alias_risk_label(c_alias: int, c_full: int, has_full_name: bool) -> Literal["safe", "risky"]:
    ...
    if c_alias > 10.0 * float(c_full):
        return "risky"
    return "safe"
```

Plus: tokens with **length ≤ 4** go to `risky` via `apply_short_acronym_heuristic` (e.g. `ONET`), regardless of FP.

So **“safe” ≈ “not a 10× count outlier vs the main dataset name”** (and not a tiny acronym). It is **not** “validated as low FP.”

## Why O*NET aliases stayed “safe” while V1 had ~40% FP

From your last run, counts were roughly:

| Alias | Count | Label |
|--------|------:|--------|
| `O*net` / `O-NET` | ~25,898 | safe |
| `Occupational Information Network` | ~6,860 | safe |
| `O*NET OnLine` | ~1,233 | safe |

Compared to anchor `O*NET` (~25k), none of these exceed **10×** the full-name count, so they all pass `_alias_risk_label` → **safe**.

That can all be true **and** V1’s top titles can still be mostly off-topic, because:

1. **High count ≠ on-topic** — Dimensions `full_data` matches substrings/homonyms (employment, “network”, occupational health, etc.).
2. **The anchor itself is huge** — if `O*NET` already matches ~26k docs, every similar alias looks “proportionally fine” vs that anchor even when many hits are noise.
3. **FP is measured later, on V1 only** — 40% Signal 3 on the **combined** V1 query is never fed back into re-labeling aliases as non-safe.
4. **V1 = safe bucket** — so “safe” aliases are exactly what V1 ORs; 40% FP on V1 is direct evidence that **the safe bucket isn’t safe for this dataset**.

So your logic holds: **if the only thing you build from “safe” aliases (V1) has 40% FP, those aliases should not be treated as safe for selection purposes.**

## Why the agent doesn’t act on that yet

The pipeline order is:

1. Phase 1: count-based safe / risky  
2. Phase 2: build V1 from “safe”, V3 from safe OR (risky AND flags)  
3. Phase 3: score FP on variants  
4. Phase 4: pick variant — but **not** reclassify aliases  

There is **no step** that says: “V1 FP &gt; X% ⇒ move all current safe aliases to risky (or require flags globally).”

So the system can simultaneously:

- Report `fp_rate_pct: 40%` and `confidence: low` on V1, and  
- Still keep those terms in the **safe** list and still prefer V1 when every variant fails the 5% gate.

That’s inconsistent in product terms, even if it’s consistent with the current code.

## What would be more coherent

Conceptually, you want **safe** to mean at least one of:

- **Empirically low FP** on a small title sample for that alias or for V1, or  
- **Structurally specific** (long product phrase), not just “count ratio OK vs a noisy anchor.”

Concrete directions (for when you implement):

1. **Post–Phase 2 relabeling** — If V1 (safe-only) FP &gt; threshold (e.g. 15–20%), treat **all** current safe aliases as needing flags → build `(safe OR risky) AND flags` (your proposed shape) or push everything into “needs context.”
2. **Don’t call them safe in the API** — Expose `aliases.safe` / `aliases.risky` as `aliases.bare_or` vs `aliases.gated` (or similar) so “safe” isn’t misleading.
3. **Stricter count rule for short/generic names** — e.g. multi-word but generic phrases (“Occupational Information Network”) get a lower count ceiling or always require flags.
4. **Align with literature** — Research already uses `(all names) AND (flags)` (~5k for O*NET); optimize could treat that as the precision variant when V1 FP is high, instead of trusting count-based “safe.”

## Bottom line

- **You’re correct:** for O*NET, “safe” aliases behave like V1, V1 has ~40% FP, so they **should not** be considered safe in the sense users care about.  
- **The bug is naming + classification logic**, not your reading of the results: “safe” is count-ratio-based, and FP is never used to downgrade aliases.  
- **Your global AND idea** (`((safe OR risky)) AND flags`) is one good fix when the whole name set is noisy; another is relabeling after V1 proves the “safe” set isn’t safe.

I’m in Ask mode, so I can’t change the code here; in Agent mode we could wire FP feedback into alias classification and/or add a global-AND variant.


# Spec Gap Analysis: What Still Needs to Be Built

Gap analysis comparing the two specification documents against the current codebase, identifying every unimplemented feature and mapping each to concrete files/modules in the project structure.

**Date:** 2025-05-03  
**Baseline commit:** current `main`

---

## 1. Context

The project has two specification documents that describe planned capabilities:

| Document | Scope | Implementation Status |
|---|---|---|
| `dataset_query_agent_spec.md` | A new **Query Optimization Agent**: takes `(dataset_name, dataset_url)` → produces an optimized Dimensions DSL query via a 4-phase pipeline (alias counting → variant building → FP verification → selection) | **Not started** — this is a net-new system |
| `docs/white-paper-dataset-research-agent.md` | 3 improvements for the existing Dataset Research Agent (LoRA fine-tuning and entity-type templates excluded from scope) | **Structured Output** ✅ Done; **Deterministic Search** and **Multi-Stage Verification** absorbed into §2 below |

### What IS already built

The current codebase implements:

- **6-step research pipeline** (`application/research.py` → `DatasetResearchUseCase`) — description, URLs, organizations, flag terms, aliases, refinement
- **Pydantic domain models** (`domain/models.py`) — `DatasetRecord`, `ResearchRequest`, `ValidationRequest`, `TermsEvaluation`, `LiteratureValidation`
- **Dimensions literature gate** (`adapters/literature.py` → `DimensionsLiteratureGate`) — queries Dimensions via `dimcli`, lexical prefilter, LLM-based publication relevance scoring
- **LLM-based alias & flag-term refinement** (`adapters/dataset_aliases.py`) — structured output via `emit_dataset_names` / `emit_flag_terms` tools
- **Terms evaluation** (`adapters/literature.py` → `evaluate_terms_with_llm`) — post-validation quality check with `suggested_exclude_terms`
- **Async API + CLI** (`interfaces/api.py`, `interfaces/cli.py`) — `POST /tasks`, `GET /tasks/{id}`, `POST /validate`, CLI `run` command
- **Multi-provider LLM** (`adapters/agent_anthropic.py`, `settings.py`) — Ollama, LMStudio support

---

## 2. Query Optimization Spec — Gap Analysis

The `dataset_query_agent_spec.md` describes the Query Optimization Agent. **None of its core functionality exists in the codebase.** This section also absorbs two white-paper improvements that are superseded by the query optimizer's structured pipeline: **Deterministic Web Search** (see §2.4) and **Multi-Stage Verification** (see §2.7). Below is a detailed breakdown.

### 2.1 §2 Input/Output Contract

| Item | Status | Detail |
|---|---|---|
| `dataset_url` input parameter | ❌ Missing | `ResearchRequest` only has `dataset_name`. No URL input. |
| `QueryOptimizationRecord` output model | ❌ Missing | Fields needed: `selected_variant`, `for_clause`, `dsl_query`, `expected_count`, `fp_rate_pct`, `confidence`, `aliases.safe`, `aliases.risky`, `alias_counts[]`, `flag_terms`, `exclusion_terms`, `all_variants_tested[]`, `notes` |
| Confidence levels (`high`/`medium`/`low`) | ❌ Missing | Based on FP rate + scope ratio thresholds |

**How to implement:**
- Add `dataset_url: Optional[str] = None` to `ResearchRequest` in `domain/models.py`
- Create a new `QueryOptimizationRecord(BaseModel)` in `domain/models.py` with all spec fields
- Add a `ConfidenceLevel` enum: `HIGH`, `MEDIUM`, `LOW`

### 2.2 §3 Dimensions DSL Reference

| Item | Status | Detail |
|---|---|---|
| `sanitize_alias()` | ❌ Missing | Escape DSL special characters: `^ : ~ \ [ ] { } ( ) ! \| & +` |
| `_q()` inner-quoted term builder | ❌ Missing | Wraps sanitized alias in escaped quotes for DSL for-clause |
| DSL pagination (`limit 1000 skip N`) | ❌ Missing | Loop to fetch all results in pages of 1000 |
| Rate limiting (2.1s between calls) | ❌ Missing | Handle 429 responses with `Retry-After` header |

**How to implement:**
- New file: `src/dataset_agent/adapters/dimensions_dsl.py`
- Reuse `dimcli` auth from `adapters/literature.py` (extract shared `_get_dsl_client()` helper)
- Add `dimensions_rate_limit_seconds: float = 2.1` to `Settings`

### 2.3 §4 Query Pattern Catalog (V1–V4)

| Item | Status | Detail |
|---|---|---|
| `build_for_clause()` | ❌ Missing | Assembles V1 (safe OR), V3 (safe OR + risky AND flags), V4 (hybrid + NOT exclusions), two-level hybrid |
| `run_variant()` | ❌ Missing | Tests a for-clause against Dimensions: `limit 20` → count + top titles |
| V1–V4 decision matrix | ❌ Missing | Logic to decide which variants to build based on dataset characteristics |

**How to implement:**
- New file: `src/dataset_agent/adapters/query_builder.py`
- `build_for_clause(aliases, hybrid=None, tier_hybrid=None, excludes=None) → str`
- `build_readable_for_clause()` for human-friendly display
- `run_variant(label, for_clause, dsl_client) → dict` using `dimensions_dsl.py`

### 2.4 §5 Alias Generation Strategy

*Absorbs white-paper §4 (Deterministic Web Search): the structured LLM prompts and Dimensions API counting below replace dynamic agent-directed web searches, achieving the original goals of reproducibility, debuggability, and training data creation through a data-driven pipeline rather than a standalone search cache.*

| Item | Status | Detail |
|---|---|---|
| Per-alias counting via Dimensions | ❌ Missing | `run_alias_count(alias)` → `limit 1` query to get publication count |
| Risk classification (10× ratio rule) | ❌ Missing | `count(bare_acronym) > 10 × count(full_name)` → risky |
| 7 alias filtering rules | ⚠️ Partial | Current `_filter_alias_entries()` in `research.py` does basic length/junk/URL filtering. **Missing**: length > 80 cap, version string removal (`^\d.v\-]+$`), generic word blocklist (model, base, small, large, etc.), sentence-pattern removal, case-insensitive canonical-form dedup |
| `dataset_url` in web search + alias list | ❌ Missing | Use `dataset_url` in alias-discovery web searches and add the URL domain (e.g., `onetonline.org`) as a safe alias when distinctive. Currently `aliases_prompt()` only passes `home_url` as context, not as a search term or candidate alias. |
| Sub-dataset / version alias removal | ❌ Missing | Detect aliases that are versions (e.g., NLSY79, NLSY97) or sub-products (e.g., QWI under LEHD) of the main dataset → remove from alias list (they belong to separate dataset records). **Note:** this reverses the original spec §9 guidance which says to include version variants. |

**How to implement:**
- Add `count_alias(alias, dsl_client) → int` to `adapters/dimensions_dsl.py`
- Add `classify_alias_risk(alias, full_name, dsl_client) → Literal["safe", "risky"]` to `adapters/dataset_aliases.py`
- Extend `_filter_alias_entries()` with remaining 4 rules from spec §5
- In `aliases_prompt()`, inject `dataset_url` into the web_search query so alias discovery includes URL-based results
- Extract domain from `dataset_url` (e.g., `onetonline.org`) and add as candidate safe alias; validate via `run_alias_count()`
- Add `detect_subdataset_aliases(aliases, dataset_name, agent) → tuple[list[str], list[str]]` to `adapters/dataset_aliases.py` — uses LLM to identify aliases that are versions/sub-products of the main dataset → returns `(clean_aliases, removed_aliases)`. Called after alias generation, before risk classification.

### 2.5 §6 False Positive Identification

| Item | Status | Detail |
|---|---|---|
| Per-alias FP domain analysis via LLM | ❌ Missing | Prompt: *"What other meanings does `{risky_alias}` have in scientific literature? List FP indicator terms."* |
| Flag term identification per risky alias | ❌ Missing | Prompt: *"What context terms indicate a paper is genuinely about `{dataset_name}` rather than `{risky_alias}`?"* |
| `classify_title()` | ❌ Missing | Keyword scanning of publication titles against FP keyword dict |
| Domain-positive override keywords | ❌ Missing | LLM-generated per dataset; if title contains override keyword → skip FP check |
| Scope comparison metric | ❌ Missing | `full_data` count vs `title_abstract_only` count; ratio > 10× → high FP risk |
| `exclude_terms` integration | ⚠️ Partial | `DatasetRecord.exclude_terms` exists and `_filter_publications_by_exclude_terms()` in `literature.py` does post-fetch filtering; but these terms are not computed via the spec's per-alias FP pipeline |
| Abstract-level alias scanning | ❌ Missing | From top-N pubs (configurable via `fp_sample_size`, default 1k, max 10k), find which have any alias as a substring in the **abstract**. From those, check if alias mentions are genuine or FP. This is the **primary FP detection method** — title scanning is a secondary fast signal. |
| Empirical exclude-term derivation | ❌ Missing | From confirmed abstract FPs, extract context terms that characterize the FP domain → feed into DSL NOT clause. This derives exclude terms from **observed data** rather than relying solely on LLM-predicted FP domains. |
| Web-search alias disambiguation | ❌ Missing | For each risky alias, run `web_search("{alias}")` → LLM analyzes search results to identify alternative real-world meanings → derive candidate exclude terms (e.g., NLX → "cells", "neurology", "receptor"). Complements LLM-only FP domain analysis with current web evidence. |

**How to implement:**
- New file: `src/dataset_agent/adapters/fp_analysis.py`
  - `identify_fp_domains(dataset_name, risky_alias, agent) → dict[str, list[str]]`
  - `generate_flag_terms_for_alias(dataset_name, risky_alias, agent) → list[str]`
  - `classify_title(title, fp_keywords) → dict`
  - `generate_domain_positive_keywords(dataset_name, agent) → list[str]`
  - `run_scope_comparison(for_clause, dsl_client) → float` (returns ratio)
- Add `EMIT_FP_ANALYSIS` and `EMIT_FLAG_TERMS_FOR_ALIAS` tool definitions to `adapters/tools.py`
- **Abstract-level FP scanning (revised Phase 3 flow):**
  - `scan_abstracts_for_aliases(publications, aliases) → list[AliasHit]` — substring search for each alias in each abstract; returns `(pub_id, matched_alias, snippet)` tuples
  - `classify_alias_mention(pub, alias, dataset_name, agent) → bool` — LLM classifies whether the alias mention in context is genuinely about the dataset (`True`) or a false positive (`False`)
  - `derive_exclude_terms_from_fps(fp_hits, agent) → list[str]` — clusters confirmed FP contexts and extracts domain-specific exclude terms
  - Title-only `classify_title()` remains as a fast secondary signal for sanity checks
- **Web-search disambiguation:**
  - `web_search_alias_meanings(alias, agent) → dict[str, list[str]]` — runs `web_search()` for the alias, LLM analyzes results to identify alternative meanings, returns `{domain: [candidate_exclude_terms]}`
  - Called during Phase 1 (after risk classification) for each risky alias; results feed into Phase 3 FP keyword dict
- Add `fp_sample_size: int = 1000` to `Settings` (configurable, max 10000)

### 2.5a FP Identification Flow (Post-Amendment Summary)

If all amendments are implemented, the complete FP identification flow is:

1. **Phase 1 — Alias counting + risk classification:** For each alias, `run_alias_count()` via Dimensions. If `count(alias) > 10× count(full_name)` → risky.
2. **Per-risky-alias LLM analysis:** Ask LLM *"What other meanings does `{risky_alias}` have in scientific literature?"* → FP indicator terms.
3. **Per-risky-alias web search (§2.5 amendment):** Run `web_search("{alias}")` → LLM analyzes real search results to discover alternative meanings not in training data → derive candidate exclude terms (e.g., NLX → "cells", "neurology").
4. **Phase 2 — Variant building:** Construct V1/V3/V4 for-clauses using safe/risky classification + flag terms + exclude terms from steps 2–3.
5. **Phase 3 — Abstract-level FP verification (§2.5 amendment):**
   - Fetch top-N publications (configurable: `fp_sample_size`, default 1000, max 10000)
   - Scan each **abstract** for alias substrings → produces `(pub, matched_alias, snippet)` tuples
   - From pubs with alias hits, LLM classifies each mention: genuine dataset reference or FP?
   - From confirmed FPs, cluster by FP domain and **empirically derive exclude terms** from observed context
   - Title-only keyword scanning (`classify_title()`) remains as a fast secondary signal
6. **Iteration:** If FP rate > 3%, refine exclusions using empirically-derived terms, re-test (max 2 iterations).
7. **Verification:** Signals 1–3 (FP rate, scope ratio, top-10 relevance).

### 2.6 §7 Optimization Pipeline (4 Phases)

This is the core orchestration layer. **Nothing exists.**

| Phase | Status | What it does |
|---|---|---|
| **Phase 1**: Alias counting | ❌ Missing | For each alias → `run_alias_count()` → classify safe/risky → remove zero-count aliases. Budget: ~10-20 API calls. |
| **Phase 2**: Variant building + testing | ❌ Missing | Construct V1/V3/V4 for-clauses → `run_variant()` with `limit 20` → quick sanity check on top 10 titles. Budget: ~3-4 API calls. |
| **Phase 3**: FP verification | ❌ Missing | For 2 best variants: `fetch_top_n(fp_sample_size)` (configurable, default 1000, max 10000) → **primary**: scan abstracts for alias substrings, LLM-classify each mention as genuine or FP, derive exclude terms from observed FPs → **secondary**: `classify_title()` keyword scan on titles → compute FP rate. If FP > 3%: add exclusions, re-test (max 2 iterations). Budget: ~4-8+ API calls per variant. |
| **Phase 4**: Selection | ❌ Missing | Score: `clean_count × (1 − fp_rate/100)`. Tie-break: V4 > V3 > V1. Set confidence level. Budget: 0 API calls. |

**Total API budget per dataset:** ~23-44 Dimensions calls, ~1-2 minutes.

**How to implement:**
- New file: `src/dataset_agent/application/query_optimization.py`
- Class: `QueryOptimizationUseCase`
  - `__init__(agent, dsl_client, settings)`
  - `execute(dataset_name, dataset_url) → QueryOptimizationRecord`
  - `_phase1_alias_counting(aliases) → dict[str, AliasInfo]`
  - `_phase2_variant_building(safe, risky, flags, excludes) → list[VariantResult]`
  - `_phase3_fp_verification(variants) → list[VerifiedVariant]`
  - `_phase4_selection(verified) → QueryOptimizationRecord`
- Wire into `bootstrap.py` alongside existing `DatasetResearchUseCase`

### 2.7 §8 Verification Criteria

*Absorbs white-paper §5.2 (Multi-Stage Verification): the data-driven signals below supersede the proposed two-pass LLM self-review. Per-alias API counting validates aliases with hard numbers (not LLM self-audit), FP rate measurement validates queries against actual publications, and Signal 3 provides focused relevance review of final results.*

| Signal | Status | Thresholds |
|---|---|---|
| Signal 1: FP rate on top-1000 | ❌ Missing | < 1% → high, 1–3% → medium, 3–5% → low, > 5% → reject |
| Signal 2: Scope ratio | ❌ Missing | < 5× → normal, 5–10× → moderate risk, > 10× → restrict hybrid to `title_abstract_only` |
| Signal 3: Top-10 title LLM scan | ❌ Missing | LLM scores each of top 10 most-cited titles using `mention_score` (0–10) + `context_score` (0–10); `final_score = min(mention, context)`. **Relevant** = final_score ≥ 7, **borderline** = 5–6, **irrelevant** = < 5. Thresholds: 8–10/10 relevant → good, 5–7 → moderate, < 5 → reject. |
| Composite verification | ❌ Missing | All pass → accept; 1 fail → lower confidence; 2+ fail → reject variant |
| Scoring rubric specificity | ⚠️ Under-specified | Signals 1 & 2 are objective (computed metrics). **Signal 3** needs a formal rubric: what constitutes "relevant" must be defined as *"the paper uses, analyzes, or directly references data from this specific dataset"*. The composite rule does not weight signals — consider weighted scoring (e.g., Signal 1=0.5, Signal 2=0.2, Signal 3=0.3) vs. simple pass/fail. Edge case: `classify_title()` finds 0 FP keywords but titles look wrong — needs a fallback rule. |

**How to implement:**
- Add `_verify_variant(variant, dsl_client, agent) → VerificationResult` method to `QueryOptimizationUseCase`
- Use `AgentPort.get_structured()` with a new `EMIT_TITLE_RELEVANCE` tool for Signal 3
- Define `SIGNAL_3_RELEVANCE_RUBRIC` prompt template in `fp_analysis.py` — reuses the `mention_score` + `context_score` pattern from `literature.py` `_llm_relevance_prompt()` with dataset-specific context
- Define "relevant" = *"the paper uses, analyzes, or directly references data from this specific dataset"*; relevant = final_score ≥ 7
- Consider weighted composite: `composite = 0.5×signal1_pass + 0.2×signal2_pass + 0.3×signal3_pass` where each signal_pass is 0.0 or 1.0; overall pass if composite ≥ 0.7
- Document edge-case handling: if `classify_title()` finds 0 FP keywords but > 40% of top-10 titles score < 5 on Signal 3, flag variant for manual review

### 2.8 §9 Edge Cases & Heuristics

| Heuristic | Status | Detail |
|---|---|---|
| Rhetorical phrase detection | ❌ Missing | If dataset name is common English phrase (< 4 words, all common words) → auto-add rhetorical NOT exclusions ("affects all of us", "for all of us", etc.) |
| Short acronym rule | ❌ Missing | Any alias < 5 chars → always route to hybrid clause, never direct OR |
| Sub-product detection | ❌ Missing | Scrape `dataset_url` homepage → identify sub-product names (e.g., QWI, LODES under LEHD) → add as safe aliases |
| Non-English name handling | ❌ Missing | Include accented and unaccented forms; LLM prompted to generate both language variants |
| Platform-as-data heuristic | ❌ Missing | If dataset is from a tech platform (e.g., LinkedIn) → add NOT clauses for platform-study and social media terms |
| Version-specific references | ❌ Missing | Auto-expand version variants (e.g., NLSY → NLSY79, NLSY97) via LLM. **Amendment note (§2.4):** sub-dataset/version aliases are now **removed** from the parent dataset's alias list and tracked as separate dataset records. This heuristic changes from "expand" to "detect and separate". |

**How to implement:**
- New file or extension: `src/dataset_agent/adapters/query_heuristics.py`
- `detect_rhetorical_name(name) → bool` + `rhetorical_exclusions(name) → list[str]`
- `is_short_acronym(alias) → bool` (len < 5)
- `detect_sub_products(url, agent) → list[str]` (reuses `make_request` tool)
- `generate_multilingual_aliases(name, agent) → list[str]`
- These are called during Phase 1/2 of the optimization pipeline

### 2.9 Cross-Validation Rules

Three validation rules ensuring consistency between aliases, flag terms, and query construction. **None exist in the original spec.** These run after alias generation + flag-term refinement, before Phase 1 alias counting.

#### Rule 1: Flag-term ≠ alias check

| Item | Status | Detail |
|---|---|---|
| Flag-term / alias overlap detection | ❌ Missing | After alias + flag_term generation, verify no flag_term appears in the alias list (case-insensitive). If overlap found: remove from aliases (flag_terms gate the hybrid clause and should not also appear in the OR clause). |

**How to implement:**
- Add `validate_no_flag_alias_overlap(aliases, flag_terms) → tuple[list[str], list[str], list[str]]` to `adapters/dataset_aliases.py` — returns `(clean_aliases, clean_flags, conflicts_removed)`
- Called after `refine_dataset_names_with_llm()` and `refine_flag_terms_with_llm()` in the pipeline

#### Rule 2: Compound alias check (alias ∉ another alias)

| Item | Status | Detail |
|---|---|---|
| Compound alias detection | ⚠️ Inversely handled | Current `filter_aliases_by_substrings()` in `text_processing.py` keeps the **longer** alias and drops the shorter substring. This is the **opposite** of what’s needed. Example: `"Occupational Information Network - O*NET"` contains `"O*NET"` → the compound form is redundant and should be removed; keep `"Occupational Information Network"` and `"O*NET"` as separate valid aliases. |

**How to implement:**
- Add `remove_compound_aliases(aliases) → list[str]` to `adapters/text_processing.py` (or modify `filter_aliases_by_substrings()`):
  - For each alias, check if it **contains** another alias from the list as a substring (case-insensitive)
  - If so, the containing (compound) alias is removed; the constituent parts remain
  - Edge case: only trigger if the contained alias is a separate entry in the list (not just a common word)
- Called during `_filter_alias_entries()` in `research.py`, after dedup

#### Rule 3: Suffix necessity check

| Item | Status | Detail |
|---|---|---|
| Suffix necessity heuristic | ❌ Missing | For aliases that are common English phrases or very short/generic terms, test whether adding a qualifying suffix ("dataset", "survey", "program", "data") significantly reduces noise. Compare `count("All of Us")` vs `count("All of Us dataset")` — if ratio > 100×, recommend the suffixed form as safe alias and gate the bare form as risky. |

**How to implement:**
- Add `check_suffix_necessity(alias, dsl_client) → Optional[str]` to `adapters/query_heuristics.py` — returns recommended suffixed alias or `None`
- Tests common suffixes: `"dataset"`, `"survey"`, `"program"`, `"data"`, `"study"`
- Decision rule: if `count(bare) / count(suffixed) > 100`, recommend the suffixed form as safe and classify bare form as risky
- Called during Phase 1 for aliases flagged as common phrases by `detect_rhetorical_name()` or with very high individual counts

---

## 3. New Files/Modules Required

| File | Purpose | Depends on |
|---|---|---|
| `adapters/dimensions_dsl.py` | DSL utilities: `sanitize_alias()`, `_q()`, `run_alias_count()`, `run_variant()`, `fetch_top_n()`, pagination, rate limiting | `dimcli`, `settings.py` |
| `adapters/query_builder.py` | `build_for_clause()` with V1/V3/V4 and two-level hybrid logic | `dimensions_dsl.py` |
| `adapters/fp_analysis.py` | FP domain identification, title classification, scope comparison, domain-positive keywords, **abstract-level alias scanning** (`scan_abstracts_for_aliases`, `classify_alias_mention`, `derive_exclude_terms_from_fps`), **web-search disambiguation** (`web_search_alias_meanings`), Signal 3 relevance rubric (`SIGNAL_3_RELEVANCE_RUBRIC`) | `dimensions_dsl.py`, `AgentPort`, `tools.py` |
| `adapters/query_heuristics.py` | Rhetorical phrase detection, short acronym routing, sub-product detection, multilingual aliases, **suffix necessity check** (`check_suffix_necessity`) | `AgentPort`, `make_request`, `dimensions_dsl.py` |
| `application/query_optimization.py` | `QueryOptimizationUseCase` — orchestrates the 4-phase pipeline | All adapters above |

### Existing Files to Update

| File | Changes |
|---|---|
| `domain/models.py` | Add `dataset_url: Optional[str]` to `ResearchRequest`; add `QueryOptimizationRecord`, `ConfidenceLevel`, `AliasInfo`, `VariantResult` models |
| `domain/ports.py` | (Optional) Add `DimensionsDslPort` abstract class if you want to decouple the DSL client |
| `adapters/tools.py` | Add `EMIT_FP_ANALYSIS`, `EMIT_FLAG_TERMS_FOR_ALIAS`, `EMIT_TITLE_RELEVANCE`, `EMIT_QUERY_VARIANTS` tool definitions |
| `adapters/dataset_aliases.py` | Add `classify_alias_risk()`; extend `_filter_alias_entries()` with remaining spec rules; add `detect_subdataset_aliases()` for version/sub-product removal; add `validate_no_flag_alias_overlap()` for cross-validation Rule 1 |
| `adapters/text_processing.py` | Add `remove_compound_aliases()` for cross-validation Rule 2 (compound alias check) — or modify existing `filter_aliases_by_substrings()` |
| `application/prompts.py` | Update `aliases_prompt()` to inject `dataset_url` into web_search query |
| `interfaces/api.py` | New endpoint: `POST /optimize` → accepts `{dataset_name, dataset_url}`, returns `QueryOptimizationRecord` |
| `interfaces/cli.py` | New CLI command: `dataset-research optimize "Name" --url "..."` |
| `settings.py` | Add `dimensions_rate_limit_seconds: float = 2.1`; add `fp_sample_size: int = 1000` (max 10000) |
| `bootstrap.py` | Wire `QueryOptimizationUseCase` with dependencies |

---

## 4. Implementation Priority Suggestion

A recommended build order based on dependency chains:

| Priority | Module | Why first |
|---|---|---|
| **P0** | `adapters/dimensions_dsl.py` | Foundation — everything else depends on DSL utilities and alias counting |
| **P0** | `domain/models.py` updates | Define output schema before building the pipeline |
| **P1** | `adapters/query_builder.py` | V1–V4 variant construction; needed before testing/verification |
| **P1** | `adapters/dataset_aliases.py` updates | Risk classification feeds into variant building |
| **P1** | `adapters/dataset_aliases.py` cross-validation | `validate_no_flag_alias_overlap()` + `detect_subdataset_aliases()` — needed before risk classification |
| **P1** | `adapters/text_processing.py` update | `remove_compound_aliases()` — reverses current substring logic |
| **P2** | `adapters/fp_analysis.py` | FP identification: abstract scanning, web-search disambiguation, empirical exclude derivation, Signal 3 rubric |
| **P2** | `adapters/query_heuristics.py` | Edge-case handling + suffix necessity check; can be added incrementally |
| **P3** | `application/query_optimization.py` | Orchestration — wire everything together |
| **P3** | `interfaces/api.py` + `cli.py` updates | Expose the new pipeline to users |

---

## 5. Architecture Recommendations (Anthropic SDK Best Practices)

Reviewed against Anthropic's *Building Effective Agents* (Dec 2024), the *Claude Agent SDK* engineering blog (2025), and current Anthropic API documentation for structured output and strict tool use. These recommendations apply to both the existing research pipeline and the new Query Optimization Agent.

### 5.1 Pattern Classification: Prompt Chaining + Parallelization

The Query Optimization pipeline should use **prompt chaining with parallelization (sectioning)**, not hub-and-spoke or orchestrator-workers.

**Rationale** (from Anthropic):

> *"Orchestrator-workers is well-suited for complex tasks where you can't predict the subtasks needed. The key difference from parallelization is its flexibility — subtasks aren't pre-defined, but determined by the orchestrator based on the specific input."*

The 4-phase pipeline has **fixed, predictable steps**: the alias list determines Phase 1 calls, the variant catalog (V1/V3/V4) determines Phase 2, and so on. All subtasks are derivable from data, not from LLM judgment. This means:

- **Phase 1** (alias counting): Pure code loop — `run_alias_count()` per alias. These are independent API calls → **parallelize with `asyncio.gather()`**.
- **Phase 2** (variant building): Pure code — `build_for_clause()` + `run_variant()` for V1/V3/V4. Sequential (few calls, fast).
- **Phase 3** (FP verification): Structured LLM calls for FP analysis. The 2 best variants are independent → **parallelize with `asyncio.gather()`**.
- **Phase 4** (selection): Pure code — scoring formula `score = clean_count × (1 − fp_rate/100)`. No LLM.

The orchestrator is `QueryOptimizationUseCase.execute()` — a Python method with `if/for` statements, not an LLM deciding what to do next. This gives determinism, debuggability, and reproducibility.

**Implementation note**: `application/query_optimization.py` should be `async` from the start. Wire `_phase1_alias_counting()` and `_phase3_fp_verification()` to use `asyncio.gather()` for concurrent Dimensions API calls (respecting the 2.1s rate limit via `asyncio.Semaphore`).

### 5.2 Deterministic vs. Agentic Classification

Every new component from §2–§2.9 classified by its correct execution pattern. Components that are arithmetic, string matching, or template building should **not** use LLM calls.

| Component | Pattern | Rationale |
|---|---|---|
| `sanitize_alias()`, `_q()` | **Pure code** | String manipulation — no judgment needed |
| `run_alias_count()` | **Pure code** (Dimensions API) | Deterministic DSL query |
| `classify_alias_risk()` (10× ratio) | **Pure code** | `count > 10 × count(full_name)` is arithmetic |
| `build_for_clause()` V1/V3/V4 | **Pure code** | Template assembly from classified aliases |
| `run_variant()` | **Pure code** (Dimensions API) | Deterministic DSL query |
| `classify_title()` | **Pure code** | Keyword scanning — no LLM needed |
| `_phase4_selection()` scoring | **Pure code** | Formula: `clean_count × (1 − fp_rate/100)` |
| `detect_rhetorical_name()` | **Pure code** | Word-count + common-word check |
| `is_short_acronym()` | **Pure code** | `len(alias) < 5` |
| `validate_no_flag_alias_overlap()` | **Pure code** | Case-insensitive set intersection |
| `remove_compound_aliases()` | **Pure code** | Substring containment check |
| `check_suffix_necessity()` | **Pure code** (Dimensions API) | Count comparison: `count(bare) / count(suffixed)` |
| FP domain identification per alias | **Structured LLM** (`get_structured`) | LLM identifies collision domains |
| Flag term generation per alias | **Structured LLM** (`get_structured`) | LLM generates domain-positive terms |
| Signal 3 top-10 relevance | **Structured LLM** (`get_structured`) | LLM scores publication titles |
| `derive_exclude_terms_from_fps()` | **Structured LLM** (`get_structured`) | LLM clusters FP contexts |
| `classify_alias_mention()` (abstract) | **Structured LLM** (`get_structured`) | LLM classifies genuine vs. FP mention |
| `detect_subdataset_aliases()` | **Structured LLM** (`get_structured`) | LLM identifies version/sub-product aliases |
| `generate_multilingual_aliases()` | **Structured LLM** (`get_structured`) | LLM generates language variants |
| Web-search alias disambiguation | **Agentic loop** (`get_information`) | LLM decides search queries, analyzes results |
| Sub-product detection from URL | **Agentic loop** (`get_information`) | LLM scrapes page, interprets content |

### 5.3 Add `strict: true` to All Emit Tool Schemas

Anthropic now supports `strict: true` on tool definitions, which **guarantees** the model output conforms to the JSON Schema. This eliminates malformed responses.

**Existing tools to update** (in `adapters/tools.py`):

| Tool | Current | Change |
|---|---|---|
| `EMIT_DATASET_NAMES` | No `strict` | Add `"strict": True` |
| `EMIT_FLAG_TERMS` | No `strict` | Add `"strict": True` |
| `EMIT_RELEVANCE_SCORE` | No `strict` | Add `"strict": True` |
| `EMIT_TERMS_EVALUATION` | No `strict` | Add `"strict": True` |

**New tools** (to be created per §3):

| Tool | Purpose |
|---|---|
| `EMIT_FP_ANALYSIS` | FP domain identification output |
| `EMIT_FLAG_TERMS_FOR_ALIAS` | Per-alias flag term generation output |
| `EMIT_TITLE_RELEVANCE` | Signal 3 top-10 relevance scoring output |
| `EMIT_QUERY_VARIANTS` | Variant test results output |
| `EMIT_ALIAS_MENTION_CLASSIFICATION` | Abstract-level genuine-vs-FP classification |
| `EMIT_EXCLUDE_TERMS` | Empirically derived exclude terms |

All new tools must include `"strict": True`, `"required"` listing all fields, and `"additionalProperties": False`.

**Compatibility note**: `strict: true` is an Anthropic API feature. When using LM Studio or Ollama backends via the Anthropic-compatible endpoint, this flag may be ignored. The `required` + `additionalProperties: false` fields still provide schema enforcement at the prompt level. Test with your local provider and fall back to the existing `tool_choice` pattern if `strict` is not supported.

### 5.4 Convert Research Steps 1–3 to Two-Phase Pattern

Steps 1–3 of the existing research pipeline (`_run_pipeline_once()` in `application/research.py`) use `get_information()` (agentic loop) and then parse free-text `===SECTION===` blocks via regex (`extract_sections()`). This is fragile — the model can format sections inconsistently, causing parse failures.

**Recommended pattern**: Two-phase calls — agentic search → structured emit.

| Step | Current (fragile) | Recommended |
|---|---|---|
| Step 1: Description + URL | `get_information()` → `extract_sections()` regex | Phase A: `get_information()` with `web_search` tool to gather info. Phase B: `get_structured()` with `EMIT_DESCRIPTION_AND_URL` tool to format findings |
| Step 2: URLs + Access | `get_information()` → `extract_sections()` regex | Phase A: `get_information()` with `web_search` + `make_request`. Phase B: `get_structured()` with `EMIT_URLS_AND_ACCESS` tool |
| Step 3: Organizations | `get_information()` → `extract_list()` regex | Phase A: `get_information()` with `web_search`. Phase B: `get_structured()` with `EMIT_ORGANIZATIONS` tool |

**New emit tools needed:**

- `EMIT_DESCRIPTION_AND_URL`: `{"description": str, "home_url": str}`
- `EMIT_URLS_AND_ACCESS`: `{"data_url": str, "schema_url": str, "documentation_url": str, "access_type": str}`
- `EMIT_ORGANIZATIONS`: `{"organizations": [str]}`

This eliminates the `TextExtractorPort.extract_sections()` dependency for these steps and makes output parsing deterministic.

**Note**: Steps 4–6 (alias refinement, flag term refinement, dataset name refinement) already use the structured `get_structured()` pattern correctly — no changes needed.

### 5.5 Enrich Tool Descriptions per Anthropic Appendix 2

Anthropic's tool engineering guidance:

> *"A good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools."*
> *"Poka-yoke your tools. Change the arguments so that it is harder to make mistakes."*

Current tool descriptions in `adapters/tools.py` are functional but minimal. Improvements:

**`web_search`:**
- Add example: `"Example: query='Current Population Survey official website census bureau'"`
- Add edge case: `"Returns 'no results' if nothing found. Retry with simpler query if needed."`
- Add constraint: `"Max 6 results returned. Query should be < 120 chars for best results."`

**`make_request`:**
- Add example: `"Example: url='https://www.census.gov/programs-surveys/cps.html'"`
- Add edge case: `"Returns error string on network failure or timeout (30s). Only GET requests."`
- Add boundary: `"Use web_search to FIND URLs, then make_request to VALIDATE them. Do not guess URLs."`

**All new tools** (`EMIT_FP_ANALYSIS`, `EMIT_FLAG_TERMS_FOR_ALIAS`, etc.):
- Each `description` field should include: (1) what the tool returns, (2) when to call it ("Call exactly once"), (3) what NOT to include
- Each property `description` should include value constraints (e.g., "0–10 integer", "max 5 items", "empty string if not found")

This is especially important when using local models (LM Studio, Ollama) which may need more guidance than Anthropic's hosted models.

# Dataset Query Optimization Agent — Technical Specification

## 1. Overview

### Purpose

An AI agent that receives a **dataset name** and **dataset URL** as input and returns an **optimal Dimensions DSL query** to find academic publications that mention or use the dataset. The agent automates the manual query optimization methodology developed across 11 rounds of team discussions (Jan–Mar 2026) and validated on 24+ workforce datasets.

### Design Philosophy

**Precision over recall.** The agent should return fewer, higher-quality publications rather than a larger set with significant false positives. The target false positive rate on a top-1000 sample is **< 3%**. This threshold was validated across all 24 production datasets — every one achieved < 3% FP after optimization.

### Why This Agent Exists

Previously, optimizing a query for a single dataset required:
- Manual alias brainstorming and expansion
- Individual alias counting via Dimensions API
- Building 3–5 query variants with progressively refined boolean logic
- Fetching top-1000 publications and scanning for false positives
- Iterating on exclusion terms and flag terms

This process took 1–3 hours per dataset and required deep knowledge of DSL syntax, common FP domains, and alias quality heuristics. The agent encodes all accumulated knowledge into an autonomous pipeline.

---

## 2. Input/Output Contract

### Input

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dataset_name` | string | Yes | Canonical name, e.g. `"American Community Survey (ACS)"` |
| `dataset_url` | string | Yes | Homepage or documentation URL, e.g. `"https://www.census.gov/programs-surveys/acs"` |

### Output

```json
{
  "dataset_name": "American Community Survey (ACS)",
  "dataset_url": "https://www.census.gov/programs-surveys/acs",
  "selected_variant": "V4: Hybrid + NOT cardiology",
  "for_clause": "(\"American Community Survey\" OR \"ACS data\" OR ...) OR ((\"ACS\") AND (\"Census Bureau\" OR ...)) NOT (\"coronary\" OR ...)",
  "dsl_query": "search publications in full_data for \"(...)\" where year in [2015:2025] and type=\"article\" and research_org_countries=\"US\" return publications[id+title+doi+year+times_cited] sort by times_cited",
  "expected_count": 3200,
  "fp_rate_pct": 1.2,
  "confidence": "high",
  "aliases": {
    "safe": ["American Community Survey", "ACS data", "ACS PUMS"],
    "risky": ["ACS"],
    "all": ["American Community Survey", "ACS data", "ACS PUMS", "ACS"]
  },
  "alias_counts": [
    {"alias": "American Community Survey", "count": 15000},
    {"alias": "ACS", "count": 500000}
  ],
  "flag_terms": ["Census Bureau", "demographics", "household survey"],
  "exclusion_terms": ["coronary", "cardiac", "acute coronary syndrome"],
  "all_variants_tested": [
    {"label": "V1", "count": 15200, "fp_rate_pct": 0.0},
    {"label": "V3", "count": 18500, "fp_rate_pct": 2.1},
    {"label": "V4", "count": 17800, "fp_rate_pct": 1.2}
  ],
  "notes": "The bare acronym 'ACS' was gated with Census/demographics context terms due to collision with Acute Coronary Syndrome in medical literature."
}
```

### Confidence Levels

| Level | Criteria |
|-------|----------|
| `high` | FP rate < 1% AND scope ratio < 5x |
| `medium` | FP rate 1–3% OR scope ratio 5–10x |
| `low` | FP rate 3–5% OR scope ratio > 10x |

The **scope ratio** is `full_data count / title_abstract_only count`. A high ratio indicates most matches are in body text only, which correlates with higher false positive risk (validated via UI Wage Records analysis: 71.9% of body-text-only matches were false positives).

---

## 3. Dimensions DSL Query Reference

### Anatomy of a Query

```
search publications in full_data for "<FOR_CLAUSE>"
where year in [2015:2025] and type="article" and research_org_countries="US"
return publications[id+title+doi+year+times_cited] sort by times_cited
```

| Component | Description |
|-----------|-------------|
| `full_data` | Searches title + abstract + indexed full text (~70% of pubs). Case-insensitive, stemmed. |
| `title_abstract_only` | Searches only title and abstract. Higher precision, lower recall. |
| `FOR_CLAUSE` | Boolean expression with OR, AND, NOT operators (must be UPPERCASE) |
| `where` | Filters: year range, publication type, country of research organizations |
| `return` | Fields to include in results |
| `sort by times_cited` | Returns highest-impact publications first (critical for FP sampling) |
| `limit N skip M` | Pagination — max 1000 per page, max ~50,000 total |

### Search Scopes

| Scope | Description | When to Use |
|-------|-------------|-------------|
| `full_data` | Title + abstract + full text. Case-insensitive, stemmed. | Default for all queries. Maximizes recall. |
| `title_abstract_only` | Title + abstract only. | Comparison metric for FP risk estimation. |
| `full_data_exact` | Full text, case-sensitive, no stemming. | Short/ambiguous terms like "T5", "Yi" where stemming causes false matches. |

### DSL Special Character Escaping

Characters that must be backslash-escaped inside for-clauses:

```
^ : ~ \ [ ] { } ( ) ! | & +
```

Python implementation (from `optimize_queries_wave4.py:43-56`):

```python
DSL_SPECIAL_CHARS = set('^:~\\[]{}()!|&+')

def sanitize_alias(alias: str) -> str:
    """Escape DSL special characters."""
    return ''.join(f'\\{ch}' if ch in DSL_SPECIAL_CHARS else ch for ch in alias).strip()

def _q(term: str) -> str:
    """Inner-quoted term: \\\"term\\\" for use inside single-quote for-clause."""
    return f'\\"{sanitize_alias(term)}\\"'
```

### Quoting and Escaping Layers

There are three escaping layers in a complete DSL query string:

1. **DSL special chars** → backslash-escaped via `sanitize_alias()`
2. **Phrase quoting** → wrapped in escaped quotes via `_q()` to produce `\\\"term\\\"`
3. **Python string** → the outer `f'search ... for "{for_clause}" ...'` uses regular quotes

Example for the term `O*NET data`:
- After `sanitize_alias()`: `O\\*NET data`
- After `_q()`: `\\\"O\\*NET data\\\"`
- In DSL string: `search publications in full_data for "\\\"O\\*NET data\\\"" where ...`

### Pagination

```python
# Fetch all results in pages of 1000
skip = 0
while skip < total_count:
    dsl = f'... limit 1000 skip {skip}'
    results = run_query(dsl)
    skip += len(results)
```

Rate limit: 2.1 seconds between API calls. Handle 429 responses with `Retry-After` header.

---

## 4. Query Pattern Catalog

### The V1–V4 Progression

The core methodology builds 4 query variants of increasing sophistication. Each variant adds a layer of complexity. The agent tests each against the Dimensions API and selects the one with the best precision/recall balance.

#### V1: Safe Aliases Only (OR)

Only uses specific, multi-word aliases that have low false positive risk.

```
("ADP payroll data" OR "ADP payroll" OR "ADP National Employment Report" OR "ADP Research Institute")
```

- **Precision**: Very high (0% FP typical)
- **Recall**: Low — misses publications that use shorter/informal references
- **When to prefer**: When all aliases are specific enough (e.g., Revelio Labs: 44 pubs, 0% FP)

**Production example**: Revelio Labs — `"Revelio Labs data" OR "Revelio data" OR "Revelio Labs"` → 44 pubs, 0% FP

#### V2: All Aliases (OR)

Adds risky aliases (short acronyms, ambiguous terms) to the OR clause.

```
("ADP payroll data" OR "ADP payroll" OR "ADP National Employment Report" OR "ADP Research Institute" OR "ADP")
```

- **Precision**: Low — short aliases match unrelated domains
- **Recall**: High
- **When to prefer**: Almost never in production. Used as a diagnostic to measure the FP inflation from risky aliases.

**Production example**: ADP V2 = 851 pubs, but top results include "Mitochondrial membrane potential" and "Biocatalysis" — FP from Adenosine Diphosphate.

#### V3: Hybrid (Safe OR + Risky AND Flags)

Short/ambiguous aliases are gated by AND-qualifying flag terms that indicate the paper is about the dataset's domain.

```
("ADP payroll data" OR "ADP payroll" OR "ADP National Employment Report" OR "ADP Research Institute")
OR (("ADP") AND ("payroll" OR "employment report" OR "workforce" OR "wages" OR "labor market"))
```

- **Precision**: High — flag terms filter out most FPs from the risky alias
- **Recall**: Good — captures papers that use the short form in a relevant context
- **When to prefer**: When risky aliases have known collision domains and good flag terms exist

**Production example**: ADP V3 = 569 pubs, 0% FP after NOT biology exclusions.

#### V4: Hybrid + NOT Exclusions

Adds explicit NOT clauses to exclude known false positive domains.

```
("ADP payroll data" OR "ADP payroll" OR "ADP National Employment Report" OR "ADP Research Institute")
OR (("ADP") AND ("payroll" OR "employment report" OR "workforce" OR "wages" OR "labor market"))
NOT ("adenosine diphosphate" OR "ATP" OR "phosphorylation" OR "kinase" OR "mitochondrial" OR "platelet" OR "ribosylation" OR "enzyme")
```

- **Precision**: Very high — excludes known FP domains
- **Recall**: Slightly reduced from V3 (some edge-case papers excluded)
- **When to prefer**: Default choice when FP rate of V3 exceeds 1%

**Production example**: ADP V4 = 569 pubs, 0% FP. LinkedIn V4 = 813 pubs, 0.86% FP.

#### V5: Conservative Baseline (Optional)

Ultra-specific: only the full official name and/or grant numbers. Used as a floor comparison.

```
("All of Us Research Program" OR "Ul1tr001855")
```

**Production example**: All of Us V5 — only used as a diagnostic baseline.

### Two-Level Hybrid Pattern (Advanced)

For datasets with multiple tiers of ambiguity, the `build_for_clause()` function supports a two-level hybrid (from `optimize_allofus_query.py:224-259`):

```python
build_for_clause(
    aliases=SAFE_ALIASES,          # Direct OR
    hybrid={"broad_term": "All of Us", "flags": HYBRID_FLAGS},  # Broad term AND flags
    tier_hybrid={"terms": TIER_TERMS, "qualifier": "All of Us"},  # Tier terms AND qualifier
    excludes=EXCLUDE_TERMS,        # NOT clause
)
```

Produces:
```
(safe1 OR safe2) OR (("All of Us") AND ("precision medicine" OR "biobank" OR ...))
OR (("Controlled Tier" OR "Registered Tier") AND ("All of Us"))
NOT ("affects all of us" OR "benefits all of us" OR ...)
```

### Decision Matrix: When to Use Each Pattern

| Dataset Characteristic | Recommended Pattern | Example |
|-----------------------|--------------------:|---------|
| Unique, multi-word name | V1 (safe OR only) | NLSY, Revelio Labs, ORS |
| Short acronym with FP collision | V4 (hybrid + NOT) | ADP, HRS, CPS, NLx |
| Common English phrase | V4 with rhetorical exclusions | All of Us |
| Sub-products with own names | V1 with expanded alias list | LEHD (QWI, LODES) |
| Very short acronym (2-3 chars) | V3 with tight flag terms | UI (wage records) |
| No common acronym | V1 with full name only | Multipurpose Occupational Systems Analysis Inventory (7 pubs) |

---

## 5. Alias Generation Strategy

### Goal

Generate all terms researchers might use to refer to a dataset. The alias quality determines both recall and precision — aliases that are too generic inflate FPs, while missing aliases cause recall gaps.

### Critical Insight: The Terminology Gap

**Round 08 finding**: For UI Wage Records, LLM full-text extraction revealed 1,663 unique mention forms used by researchers, but only 7 of our 20 search aliases actually appeared in the extracted terminology. Researchers use different names than dataset administrators expect.

This means alias generation must think from the **researcher's perspective**, not the **dataset provider's perspective**.

### Alias Categories

| Category | Risk Level | Example | Handling |
|----------|-----------|---------|----------|
| Full official name | Safe | "Occupational Information Network" | Direct OR |
| Name + "data"/"survey" | Safe | "O*NET data", "ACS survey" | Direct OR |
| Specific acronym compound | Safe | "CPS-ASEC", "NLSY79", "PSID-CDS" | Direct OR |
| Publisher-qualified | Safe | "Census ACS", "BLS Time Use Survey" | Direct OR |
| Sub-product name | Safe | "Quarterly Workforce Indicators (QWI)" | Direct OR |
| Bare acronym (distinctive) | Safe | "NLSY", "LEHD" | Direct OR (if count < 10x full name) |
| Bare acronym (ambiguous) | Risky | "ADP", "HRS", "CPS", "UI" | AND flag terms + NOT exclusions |
| Single generic word | Exclude | "data", "survey", "records" | Never use |
| Common English phrase | Risky | "All of Us" | AND flag terms + NOT rhetorical |

### Risk Classification Heuristic

An alias is **risky** if its individual Dimensions count is > 10x the count of the full official name. This was validated across multiple datasets:

| Alias | Count | Full Name Count | Ratio | Classification |
|-------|------:|----------------:|------:|---------------|
| "ADP" | 85,172 | "ADP payroll data": 5 | 17,034x | Risky |
| "HRS" | 812,496 | "Health and Retirement Study": 17,065 | 47x | Risky |
| "CPS" | 578,036 | "Current Population Survey": 14,071 | 41x | Risky |
| "ONET" | 50,000+ | "O*NET data": 5,373 | 9x | Borderline |
| "NLSY" | 2,078 | "National Longitudinal Survey of Youth": 4,140 | 0.5x | Safe |
| "LEHD" | 1,574 | "Longitudinal Employer Household Dynamics": 1,574 | 1x | Safe |

**Rule**: If `count(bare_acronym) > 10 * count(full_name)`, the acronym must be gated with AND flag_terms + NOT exclusions.

### Alias Filtering Rules

Derived from `select_discriminating_aliases()` in `search_publications_dsl.py:197-236`:

1. **Remove aliases < 4 characters** (unless they are known distinctive acronyms like "QWI")
2. **Remove aliases > 80 characters** (likely LLM-generated sentence fragments)
3. **Remove pure version strings** (regex: `^[\d.v\-]+$`)
4. **Remove generic words**: model, base, small, large, tiny, medium, mini, data, survey, records, information
5. **Remove sentence-like patterns**: aliases containing "the", "is", "are", "was", "which", "used", "based"
6. **Deduplicate case-insensitively** — Dimensions `full_data` is case-insensitive, so "O*NET" and "o*net" return identical results
7. **Keep the most canonical form** — prefer "RoBERTa" over "roberta", "O*NET" over "onet"

### Alias Generation via LLM

The agent should generate aliases by sending the dataset context to an LLM. The prompt should request:

1. Full official name and all known variants
2. Common acronyms and abbreviations
3. Acronym + descriptor compounds (e.g., "ACS data", "ACS survey", "ACS PUMS")
4. Publisher-qualified forms (e.g., "Census ACS", "BLS ATUS")
5. Sub-product names (e.g., "CPS-ASEC", "QWI", "LODES")
6. Version/year variants (e.g., "NLSY79", "NLSY97")
7. URL if it is a well-known domain (e.g., "onetonline.org")
8. Informal names researchers commonly use

The LLM should explicitly be told to **NOT** include:
- Generic terms like "survey data" or "government data"
- Terms shorter than 4 characters unless they are well-known acronyms
- Sentence fragments or descriptions

### LLM Configuration

Existing scripts use LM Studio local model at `http://192.168.62.9:1234` with model `gpt-oss-120b`. The agent should support configurable LLM endpoints (local or cloud API).

---

## 6. False Positive Identification

### The FP Problem

Many dataset names and acronyms collide with terms from unrelated scientific domains:

| Dataset | Acronym | FP Domain | FP Terms |
|---------|---------|-----------|----------|
| ADP Payroll | ADP | Biochemistry | adenosine diphosphate, ATP, phosphorylation, kinase |
| Health & Retirement Study | HRS | Medical | hazard ratio, cox regression, tumor, carcinoma, survival analysis |
| Current Population Survey | CPS | Engineering | cyber-physical, IoT, embedded system |
| Current Population Survey | CPS | Social work | child protective, child welfare, foster care |
| Longitudinal Business Database | LBD | Biochemistry | ligand-binding domain, receptor, crystal structure |
| Panel Study of Income Dynamics | PSID | Engineering | power system, voltage, inverter, photovoltaic |
| National Labor Exchange | NLx | Pharmacology | opioid, receptor, naloxone, fentanyl, morphine |
| FRBNY Consumer Credit Panel | CCP | Chemistry/Politics | cyclopentadienyl, catalytic, Chinese Communist Party |
| Burning Glass | BG | Materials science | stained glass, glass fiber, glass transition, borosilicate |
| LinkedIn | LinkedIn | Social media | social media marketing, social network analysis, sentiment analysis |
| All of Us | "All of Us" | Rhetoric | "affects all of us", "benefits all of us", "for all of us" |
| O*NET | ONET | Technology | optical network, neural network, sensor network |

### FP Identification via LLM

For each risky alias, the agent should ask the LLM:

> "The dataset '{dataset_name}' uses the acronym '{risky_alias}'. What other meanings does this acronym/term have in scientific literature? List domain-specific indicator terms that would signal a false positive (the paper is NOT about this dataset)."

Expected output: a list of FP domains with indicator terms (matching the structure of `FP_KEYWORDS` in `optimize_queries_wave4.py:175-303`).

### Flag Term Identification via LLM

For each risky alias, the agent should also ask:

> "What context terms would indicate a paper is genuinely about the dataset '{dataset_name}' rather than a different meaning of '{risky_alias}'? List terms that researchers using this dataset would likely mention."

Expected output: a list of domain-positive flag terms (matching the structure of `HYBRID_FLAGS` in `optimize_allofus_query.py:201-209`).

### Scope Comparison Technique

Compare `full_data` vs `title_abstract_only` counts for the same query. A high ratio indicates many body-text-only matches, which have higher FP risk.

From `validate_ui_wage_records.py` analysis:
- UI Wage Records V6: `full_data` = 4,398, `title_abstract_only` = 30 → ratio 147x
- True positive rate on body-text-only matches: **4.7%** (vs ~16% for alias-only matches)
- This validated that a high scope ratio is a reliable FP risk indicator

**Agent threshold**: If scope ratio > 10x, flag as high FP risk and consider restricting hybrid portion to `title_abstract_only`.

### FP Detection via Title Scanning

After fetching top-1000 publications, scan titles for FP keywords. Implementation from `optimize_queries_wave4.py:310-318`:

```python
def classify_title(title: str, fp_keywords: dict) -> dict:
    """Check title against FP keywords. Return matched categories."""
    title_lower = title.lower()
    matches = {}
    for category, keywords in fp_keywords.items():
        matched = [kw for kw in keywords if kw.lower() in title_lower]
        if matched:
            matches[category] = matched
    return matches
```

The agent uses LLM-generated FP keywords (from Stage 3) instead of the hardcoded `FP_KEYWORDS` dict.

### Domain-Positive Override

Some keywords in a title definitively indicate a true positive, overriding any FP keyword matches. From `optimize_allofus_query.py:393-407`:

```python
DOMAIN_POSITIVE_KEYWORDS = [
    "All of Us Research Program",
    "Researcher Workbench",
    "precision medicine initiative",
    ...
]
# If title contains any domain-positive keyword → skip FP check
```

The agent should generate domain-positive keywords for each dataset alongside FP keywords.

---

## 7. Optimization Pipeline

The agent follows a 4-phase pipeline, matching the methodology from `optimize_queries_wave4.py`:

### Phase 1: Individual Alias Counting

For each alias, run a `limit 1` query to get the publication count. This is fast (~2.1s per alias) and reveals:
- Which aliases are inflation sources (bare "ADP" → 85,172 vs "ADP payroll data" → 5)
- Which aliases return zero results (can be removed)
- The safe/risky classification based on count ratios

Implementation from `optimize_queries_wave4.py:132-143`:
```python
def run_alias_count(alias, headers):
    for_clause = _q(alias)
    dsl = f'search publications in full_data for "{for_clause}" {WHERE} return publications[id+title+times_cited] limit 1'
    count, titles = query_count_and_titles(dsl, headers)
    return {"alias": alias, "count": count}
```

**Budget**: ~10-20 API calls (one per alias at 2.1s = ~20-40 seconds)

### Phase 2: Variant Building and Testing

Build V1, V3, V4 variants (skip V2 — it's diagnostic only). For each variant:
1. Construct the for-clause using `build_for_clause()`
2. Run the query with `limit 20` to get count + top titles
3. Quick sanity check: are the top 10 titles relevant?

Implementation from `optimize_queries_wave4.py:146-168`:
```python
def run_variant(label, for_clause, headers, n_titles=20):
    dsl = f'search publications in full_data for "{for_clause}" {WHERE} return publications[{FIELDS}] sort by times_cited limit {n_titles}'
    count, titles = query_count_and_titles(dsl, headers)
    return {"label": label, "for_clause": for_clause, "count": count, "top_titles": titles}
```

**Budget**: ~3-4 API calls (one per variant)

### Phase 3: FP Verification

For the 2 most promising variants, fetch top 1000 publications sorted by citations and scan titles for FP keywords.

Implementation from `optimize_queries_wave4.py:102-129` and `321-374`:
```python
pubs, total_count = fetch_top_n(for_clause, headers, n=1000)
fp_analysis = analyze_fps(pubs, dataset_key, label)
# Returns: total_scanned, clean_count, fp_count, fp_rate_pct, fp_by_category
```

**If FP rate > 3%**: Add more exclusion terms and re-test (max 2 iterations). This follows the pattern from `optimize_allofus_query.py:664-708` where V4 exclusions are built dynamically from V3's observed FPs.

**Budget**: ~4-8 API calls per variant for pagination (1000 pubs at 1000/page)

### Phase 4: Selection

Score each variant and select the best one.

**Selection rules** (from `optimize_allofus_query.py:712-728`):
1. Eliminate variants with FP rate > 5%
2. Among remaining, pick the one with the highest total count
3. If no variant is under 5% FP, pick the lowest FP rate and report `confidence: "low"`
4. If V1 returns 0 results, report that no publications were found

**Scoring formula**: `score = clean_count * (1 - fp_rate / 100)`

**Tie-break preference**: V4 > V3 > V1 (hybrid with exclusions is the sweet spot, per the results across 24 datasets).

### Total API Budget Per Dataset

| Phase | API Calls | Time (at 2.1s/call) |
|-------|----------:|--------------------:|
| Phase 1: Alias counting | 10-20 | 21-42s |
| Phase 2: Variant testing | 3-4 | 6-8s |
| Phase 3: FP verification | 8-16 | 17-34s |
| Phase 3b: Scope comparison | 2-4 | 4-8s |
| **Total** | **23-44** | **~1-2 minutes** |

Plus LLM calls: ~4 calls, ~2000 tokens total. Negligible time if using local model.

---

## 8. Verification Criteria

The agent self-verifies using three signals, all proven across 11 rounds:

### Signal 1: FP Rate on Top-1000

The primary metric. Fetch the top 1000 publications by citation count and scan titles for FP keywords.

| Threshold | Action |
|-----------|--------|
| < 1% | High confidence. Accept query. |
| 1-3% | Medium confidence. Accept with note. |
| 3-5% | Low confidence. Try adding more exclusions. |
| > 5% | Reject variant. Fall back to V1 or iterate. |

### Signal 2: Scope Comparison Ratio

Compare `full_data` count vs `title_abstract_only` count.

| Ratio | Interpretation |
|------:|---------------|
| < 5x | Normal. Most matches are in title/abstract. |
| 5-10x | Moderate risk. Many body-text-only matches. |
| > 10x | High risk. Consider restricting hybrid to `title_abstract_only` scope. |

### Signal 3: Top-Title Relevance

Quick LLM scan of the top 10 most-cited titles: are they about the dataset?

| Relevant / 10 | Interpretation |
|--------------:|---------------|
| 8-10 | Good — query is on target |
| 5-7 | Moderate — some noise, but acceptable |
| < 5 | Poor — variant is too broad, reject |

### Composite Verification

If all three signals pass → accept variant with reported confidence.
If one fails → accept with lower confidence and explanatory note.
If two or more fail → reject variant, iterate or fall back to V1.

---

## 9. Edge Cases & Heuristics

### Highly Ambiguous Names

**Case study**: "All of Us" (NIH Research Program)

The phrase "all of us" appears rhetorically in thousands of scientific papers ("this affects all of us", "benefits all of us"). Solution from `optimize_allofus_query.py`:
- Gate with biobank/precision medicine flag terms
- Exclude rhetorical phrases (21 patterns like "affects all of us", "for all of us")
- Use two-level hybrid: safe aliases OR (broad term AND flags) OR (tier terms AND qualifier) NOT exclusions
- Result: V3 achieved viable count; V4 with exclusions provided clean results

**Agent heuristic**: If the dataset name is a common English phrase (< 4 words, all common words), automatically add rhetorical exclusion patterns.

### Short Acronyms with Medical/Science Collision

**Affected datasets**: ADP, HRS, CPS, NLx, PSID, LBD, ATUS, CCP

The FP_KEYWORDS dict in `optimize_queries_wave4.py:175-303` documents 15+ collision domains. Common pattern:
- Biomedical: adenosine diphosphate (ADP), hazard ratios (HRS), ligand-binding domain (LBD)
- Engineering: cyber-physical systems (CPS), power systems (PSID)
- Pharmacology: naloxone (NLx)

**Agent heuristic**: For any alias < 5 characters, always generate FP domain analysis via LLM and include the alias only in a hybrid clause.

### Datasets with No Common Acronym

**Example**: "Multipurpose Occupational Systems Analysis Inventory" → 7 pubs

When the full name is long and specific enough, V1 with just the full name is sufficient. No hybrid or exclusion logic needed.

**Agent heuristic**: If V1 returns < 50 pubs and FP rate = 0%, accept V1 without building V3/V4.

### Datasets with Sub-Products

**Example**: LEHD program → Quarterly Workforce Indicators (QWI), LEHD Origin-Destination Employment Statistics (LODES)

Each sub-product needs its own aliases in the OR clause. From `wave4_reviewed_queries.json`:
```
"LEHD data" OR "LEHD" OR "Quarterly Workforce Indicators (QWI)" OR
"LEHD Origin-Destination Employment Statistics (LODES)" OR
"Linked longitudinal employer-employee data"
```

**Agent heuristic**: The dataset URL scraping (Stage 1) should identify sub-products from the homepage. The LLM alias generation should explicitly ask for sub-product names.

### Platform Names Used as Data

**Example**: LinkedIn → researchers may study the platform itself OR use LinkedIn data as a workforce data source.

Solution from `optimal_queries_all_datasets.json`:
- Specific aliases: "LinkedIn data", "LinkedIn employee data", "LinkedIn workforce data"
- Hybrid: ("LinkedIn") AND ("workforce data" OR "labor market data" OR "employment data")
- NOT: "social media marketing", "social network analysis", "sentiment analysis"

**Agent heuristic**: If the dataset is from a technology platform, add NOT clauses for platform-study terms and social media analysis terms.

### Non-English Dataset Names

**Example**: RAIS — Relacao Anual de Informacoes Sociais (Brazilian)

Include both the original language name and English translation. The agent should ask the LLM to generate aliases in both languages. Include accented and unaccented forms (e.g., "Relacao Anual de Informacoes Sociais" and "Relacao Anual de Informacoes Sociais").

**Dramatic case**: Bare "RAIS" matched 88,191 publications. After removing the bare acronym and keeping only specific aliases ("Relacao Anual de Informacoes Sociais", "Brazilian Employer-Employee Dataset", "RAIS-CAGED", etc.), the count dropped to **64 pubs** at 1.6% FP — a 1,378x reduction. This is the most extreme example of alias inflation in the project.

### Datasets with Version-Specific References

**Example**: NLSY79 vs NLSY97 — researchers often reference specific cohorts.

Both version-specific ("NLSY79") and version-agnostic ("NLSY") forms should be included. From `optimal_queries_all_datasets.json`:
```
"National Longitudinal Survey of Youth" OR "NLSY" OR "NLSY79" OR "NLSY97" OR
"NLS Youth" OR "NLS-Y" OR "National Longitudinal Surveys" OR "NLSY data" OR
"NLSY Children" OR "NLSY-CY"
```

All of these are safe aliases — "NLSY" has a 0.5x ratio to the full name, making it safe for direct OR inclusion.

---

## 10. Reference Data

### Ground Truth Queries

The following files contain optimized queries validated across all rounds. The agent should be able to reproduce queries of comparable quality for these known datasets:

| File | Path | Description |
|------|------|-------------|
| `optimal_queries_all_datasets.json` | `data/` | 24+ datasets with recommended variants, FP rates, for-clauses |
| `wave4_reviewed_queries.json` | `data/` | 14 datasets with Julia's corrections applied |
| `top1000_fp_verification.json` | `data/` | FP verification results per dataset (top-1000 scan) |
| `optimize_queries_wave3_report.md` | `data/` | Narrative report for UI Wage, ADP, LinkedIn, Lightcast |
| `optimize_queries_wave4_report.md` | `data/` | Narrative report for 19 Tier 1/2/3 datasets |
| `optimize_allofus_report.md` | `data/` | All of Us edge case with rhetorical FP handling |

### FP Rate Achieved Across All Datasets

| Dataset | Count | FP Rate | Pattern Used |
|---------|------:|--------:|-------------|
| UI Wage Records | 84 | 1.19% | V4: 18 specific + NOT medical |
| ADP Payroll | 569 | 0.0% | V3: Hybrid + NOT biology |
| LinkedIn Data | 813 | 0.86% | V4: Hybrid + NOT social media |
| Lightcast / Burning Glass | 425 | 0.0% | V4: Hybrid + NOT materials science |
| O*NET | 2,319 | 0.8% | V1: 4 specific aliases |
| LEHD | 1,574 | 0.0% | V1: 10 specific aliases |
| Revelio Labs | 44 | 0.0% | V1: 6 specific aliases |
| NLx | 219 | 0.0% | V1: specific aliases |
| HRS | 113,696 | 0.2% | V4: Hybrid + NOT medical/time |
| CPS | 57,329 | 0.8% | V4: Hybrid + NOT cyber/child |
| LBD | 564 | 1.6% | V4: Hybrid + NOT biochemistry |
| Decennial Census | 8,596 | 0.0% | V4: Hybrid + NOT redistricting |
| NLSY | 7,359 | 0.0% | V1: 10 specific aliases |
| PSID | 4,515 | 0.0% | V4: Hybrid + NOT power systems |
| ATUS | 2,346 | 0.0% | V4: + NOT non-US time use |
| USPTO Patents | 356,633 | 0.1% | V1: specific aliases |
| Swedish Registers | 3,129 | 0.2% | V1: specific aliases |
| PIAAC | 369 | 0.0% | V1: specific aliases |
| CCP / FRBNY | 23,540 | 0.4% | V1: specific aliases |
| Dingel & Neiman | 31 | 0.0% | V1: specific aliases |
| ORS | 8 | 0.0% | V1: specific aliases |
| MOSAIC | 7 | 0.0% | V1: specific aliases |
| RAIS | 64 | 1.6% | V1: specific only (bare "RAIS" removed — reduced from 88,191) |

### Existing Scripts to Reference

| Script | Path | Key Functions |
|--------|------|--------------|
| `optimize_queries_wave4.py` | `scripts/` | `sanitize_alias()`, `_q()`, `get_token()`, `run_query()`, `fetch_top_n()`, `run_alias_count()`, `run_variant()`, `classify_title()`, `analyze_fps()` |
| `optimize_allofus_query.py` | `scripts/` | `build_for_clause()`, `build_readable_for_clause()`, two-level hybrid, rhetorical FP patterns |
| `validate_ui_wage_records.py` | `scripts/` | Scope comparison, LLM classification, stratified sampling |
| `generate_model_aliases.py` | `scripts/` | Two-layer alias generation (rule + LLM), merge/dedup |
| `search_publications_dsl.py` | `scripts/` | `select_discriminating_aliases()`, alias filtering heuristics |

---

## 11. Appendix: Known FP Collision Domains

Reference table of known acronym/term collisions discovered across 24 datasets (from `optimize_queries_wave4.py:175-303` and `optimize_allofus_query.py:354-390`):

| Dataset Acronym | Collision Domain | FP Indicator Terms |
|----------------|-----------------|-------------------|
| ADP | Biochemistry | adenosine diphosphate, ATP, phosphorylation, kinase, mitochondrial, platelet, ribosylation, enzyme |
| HRS | Medical statistics | hazard ratio, hazard ratios, cox regression, proportional hazard, survival analysis |
| HRS | Medical treatment | tumor, carcinoma, chemotherapy, radiotherapy, incubation, cell line, in vitro |
| HRS | Time units | 48 hrs, 72 hrs, 24 hrs, 12 hrs |
| CPS | Engineering | cyber-physical, cyber physical, IoT, internet of things, embedded system |
| CPS | Social work | child protective, child welfare, foster care, child abuse, child maltreatment |
| CPS | Other | characters per second, cardiopulmonary, capsule, nanoparticle |
| LBD | Biochemistry | ligand-binding domain, ligand binding domain, receptor, crystal structure, binding affinity, nuclear receptor, estrogen, androgen, protein, mutation, transcription |
| PSID | Power engineering | power system, voltage, inverter, photovoltaic, solar cell, grid-connected, electric drive, motor drive |
| NLx | Pharmacology | opioid, receptor, naloxone, fentanyl, morphine, peptidomimetic, analgesic, mu-opioid, delta-opioid, kappa-opioid, nociceptin, antagonist, agonist |
| NLx | NLP | natural language, NLP |
| CCP | Politics | Chinese Communist Party, Communist Party of China, Xi Jinping, Politburo |
| CCP | Chemistry | cyclopentadienyl, catalytic |
| ONET | Technology | optical network, neural network, sensor network, mesh network |
| USPTO | Medical | patent ductus, patent foramen, patent airway, patent leather |
| ATUS | Non-US surveys | European, HETUS, MTUS, Eurostat, UK time use, British, Australian, Canadian time use |
| Decennial Census | Legal/political | redistricting, gerrymandering, reapportionment, electoral, congressional district, voting rights, political representation |
| MOSAIC | Biology | fluorescence, microscopy, mosaic virus, genetic mosaic, somatic mosaic |
| RAIS | English words | Rais, raising, raised |
| LinkedIn | Social media | social media marketing, social network analysis, online dating, fake news, misinformation, sentiment analysis |
| Burning Glass | Materials science | stained glass, glass fiber, glass transition, borosilicate, optical glass, ceramic, silica |
| All of Us | Rhetoric | affects all of us, benefits all of us, for all of us, concerns all of us, impacts all of us, all of us need, all of us can, all of us should, teach all of us |
| UI (wage records) | Medical | urine, urinary, ultrasound, imaging |
| UI (wage records) | Computing | user interface, GUI, mobile app |

This table serves as a reference knowledge base. When the agent encounters a new dataset with a similar acronym pattern (e.g., a 2-4 letter acronym that could match a biomedical term), it should proactively check for the same class of collision domains via LLM.

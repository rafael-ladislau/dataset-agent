# Agent functionality with dataset
_Exported on 6/12/2026 at 11:14:13 GMT-3 from Cursor (3.7.27)_

---

**User**

give a descripton of how the agent works given a dataset name and dastase url.

---

**Cursor**

Let me explore the codebase to understand the agent's architecture and flow.

Here's a comprehensive description of how the **dataset-agent** works when given a dataset name and dataset URL.

---

## High-Level Purpose

The agent automates the process of **researching a dataset** — gathering structured metadata, aliases, organizational associations, and search terms — and optionally **optimizing a Dimensions (bibliometric database) query** to find publications that use or reference that dataset.

---

## Entry Points

You can provide a dataset name and URL through:

| Surface | Name | URL |
|---|---|---|
| **CLI** `dataset-research run` | Positional arg | Not wired (only name) |
| **CLI** `dataset-research optimize` | Positional arg | `--url` flag |
| **API** `POST /tasks` | `ResearchRequest.dataset_name` | `ResearchRequest.dataset_url` |
| **API** `POST /optimize` | `OptimizeRequest.dataset_name` | `OptimizeRequest.dataset_url` |

---

## The Research Pipeline (6-Step LLM Process)

When you give the agent a dataset name (e.g. `"NHANES"`) and optionally a URL (e.g. `"https://www.cdc.gov/nchs/nhanes/"`), the `DatasetResearchUseCase` runs a **6-step pipeline**, each step powered by an LLM (Anthropic-compatible endpoint via LM Studio or Ollama) with tool-use:

### Step 1 — Description & Home URL
The LLM generates a **natural-language description** of the dataset and identifies its **official home URL**, using the provided `dataset_url` as a reference if available.

### Step 2 — Data, Schema & Documentation URLs + Access Type
The LLM (assisted by web search tools) discovers URLs for the dataset's **data download**, **schema/documentation**, and determines the **access type** (open, restricted, etc.).

### Step 3 — Organizations & Flag Terms
The LLM identifies **organizations** associated with the dataset (producers, funders, distributors). These are processed into **flag terms** — keywords useful for filtering or flagging relevant publications.

### Step 4 — Refine Flag Terms
An LLM refinement pass cleans and deduplicates the flag terms.

### Step 5 — Aliases (Dataset Names)
The LLM generates **aliases** — all the different names, abbreviations, and acronyms by which the dataset is known. The hostname from the `dataset_url` or `home_url` may be appended. Optionally, **multilingual aliases** are generated if configured.

### Step 6 — Refine Dataset Names & Build Record
A final LLM pass refines aliases, checks for overlaps, filters sub-datasets, and assembles the canonical **`DatasetRecord`** — a structured JSON object containing all gathered metadata plus default search parameters (engine, years, publication types, etc.).

The record is **saved to disk** as `<sanitized_name>_research.json` in the configured output directory.

---

## Literature Gate (Optional Validation)

After the pipeline, an optional **literature gate** runs to validate that the gathered terms actually find relevant publications:

- **`noop` mode** — Skips validation; may still fetch a Dimensions publication count preview.
- **`dimensions` mode** — Queries the Dimensions API, samples publications, and scores their **relevance** (lexical + optional LLM scoring). If the terms don't pass the threshold, the pipeline **retries** (up to `literature_max_attempts`), refining terms each iteration. On success, the record is enriched with `literature_validation` (pass/fail, counts, FP rate) and `terms_evaluation` (LLM assessment of term quality).

---

## Query Optimization (Optional, Separate Flow)

If you hit `POST /optimize` or `dataset-research optimize`, the agent runs a **4-phase optimization** to produce the best Dimensions DSL query:

1. **Alias Counting** — Each alias is tested against Dimensions for publication counts. Aliases are bucketed as safe or risky; short acronyms get special handling.
2. **Query Variant Building** — Multiple `for` clause variants are constructed (different combinations of aliases, flags, exclude terms) and probed for counts.
3. **False-Positive Proxy** — The LLM identifies likely FP domains, web disambiguation is attempted, keyword FP rates are computed on sampled titles, and exclusion terms are derived.
4. **Best Variant Selection** — The variant with the best `count × (1 − FP%)` score wins. A **confidence level** is assigned, and the final `dsl_query` is assembled.

If `dataset_names` / `flag_terms` weren't provided, the optimizer **runs the research pipeline first** to generate them.

---

## External Services

| Service | Purpose |
|---|---|
| **LLM** (LM Studio / Ollama, Anthropic-compatible API) | All reasoning, structured extraction, relevance scoring |
| **Web Search** (DuckDuckGo or Tavily) | Discovering URLs, disambiguation, FP domain analysis |
| **Dimensions** (dimcli + API key) | Publication counts, sampling, DSL query execution |
| **HTTP** (httpx) | Fetching web pages, webhook delivery |
| **SQLite** | Async task queue state |

---

## What the Agent Produces

1. **`DatasetRecord`** (JSON) — Description, home URL, data/schema/doc URLs, access type, aliases (`dataset_names`), `flag_terms`, `exclude_terms`, organizations, default search filters, and optionally `literature_validation`, `terms_evaluation`, `publications_total`, and a `dsl_query`.

2. **`QueryOptimizationRecord`** (JSON) — The best Dimensions `for` clause, full `dsl_query` string, expected publication count, proxy FP rate, confidence level, safe/risky alias breakdown with counts, exclusion terms, and an audit trail of all variants tested.

---

## In Summary

Given `dataset_name="NHANES"` and `dataset_url="https://www.cdc.gov/nchs/nhanes/"`:

1. The LLM researches the dataset across 6 steps, using web search and the URL as anchors.
2. It builds a structured metadata record with all known aliases, organizations, and flag terms.
3. Optionally, the literature gate validates these terms against real Dimensions publications.
4. Optionally, the optimizer refines terms into a precision-tuned Dimensions DSL query.
5. The final JSON artifacts are saved to disk and returned via CLI or API.

---

**User**

in step 1, the agent should mandatorily use web search to search the description in a search engine and use web fetch to call the dataet url, is it working like that?

In the research phase, I need a step where I search the web to see if the aliases selected for the database are also name of other things like a pharmacy remedy, animal name,biological name, and etc. basically any other entity that has the same name of the dataset in context and can jeorpadize the finding of dataset mentions in a publication catalog

In the research phase, I need a step to check if there is any suffix that usually comes with the dataset name: suffixes like survey, data or dataset for example. 

The query optimization flow and the literature gate flow are not optional. They shoud run after the reasearch pipeline.

In the query optiomation flow, on each iteration I will create a candidate DSL query . This DSL query should go through the literature gate where a configurated number of publications should be pulled from dimensions, by default a thousand. The title, abastract and reference should be downloaded, and the script will try to find the cadidate dataset aliases stems on the title, abstract and references. From the findings, it will check which of them were mentioning the candidate were really mentioning the caindadate datset or were false positives, if false positives from the findings are more than 10%, try to understand the reasons of why the tested dsl query has so many falase positives and create a new dsl to be tested. Try to add exclude terms, flag_terms, add suffixes, remove wrong aliases from the original dsl to create a better dsl query with less false positives. do that at most 3 times and return the dsl query variant with the best false positive rate according with the findings in the title, abastract and references.

plan how to get it done


---

**Cursor**

I'll research the current implementation in detail before planning. Let me examine the key files.

Now let me examine the prompts, tools, literature gate, query heuristics, and settings.

Let me examine the query heuristics, settings, dimensions DSL, FP analysis, and the wiring (bootstrap, CLI, API).

Now let me examine the agent tool loop, FP analysis, and the wiring layer (bootstrap, CLI, API).

Let me look at the CLI, API, and query builder to complete the picture of how flows are wired and what DSL fields are fetched.

I now have a thorough understanding of the codebase. Let me answer your direct question first, then confirm a couple of design points before writing the plan.

## Is Step 1 working like that today?

**No.** Here's what actually happens in Step 1 (`_run_pipeline_once` → `description_and_home_url_prompt`):

- The call passes `TOOL_DEFINITIONS` (so `web_search` + `make_request` are *available*), but nothing forces their use. In [agent_anthropic.py](src/dataset_agent/adapters/agent_anthropic.py) the loop returns immediately if the model's first reply has `stop_reason == "end_turn"` with no tool call. So `web_search` is only *suggested* by the prompt text ("Use web_search to find the official website"), never mandatory.
- The `dataset_url` is injected only as `Reference URL (if provided): {url}` in [prompts.py](src/dataset_agent/application/prompts.py). There is **no instruction to fetch it**. `make_request` is only mentioned later in Step 2 to validate discovered URLs. So the agent does **not** reliably web-fetch the dataset URL in Step 1.

So both behaviors you want (mandatory search + mandatory fetch of the dataset URL) are currently absent.

## What I found relevant to your other requests

- **Homonym/collision check**: Logic already exists (`web_search_alias_meanings` in [fp_analysis.py](src/dataset_agent/adapters/fp_analysis.py)) but only runs inside query optimization Phase 3, not in the research phase.
- **Suffix check**: `check_suffix_necessity` exists in [query_heuristics.py](src/dataset_agent/adapters/query_heuristics.py) but is Dimensions-count-based and lives only in optimization Phase 1.
- **"Not optional"**: Research (`/tasks`, CLI `run`) and optimization (`/optimize`, CLI `optimize`) are separate entry points; the literature gate defaults to `noop`. Nothing chains research → optimize → gate.
- **Iterative DSL + gate**: Today optimization uses a *proxy* FP rate (title keyword hits + optional LLM signal), fetches only `basics+title+abstract+times_cited` (**no references**), iterates max 2x, and gates at 3%/5% — not your "pull 1000, match alias stems in title/abstract/references, FP>10% → refine, max 3x" design.

Before I write the plan, two design points materially change the implementation:

**Design decisions for the redesigned pipeline**

> **You want FP detection to scan title, abstract, AND references. Dimensions DSL exposes reference_ids (lists of Dimensions IDs), not reference text. How should 'references' be handled?**
> - Rely on Dimensions 'full_data' search scope (already indexes full text incl. reference/acknowledgement sections); match alias stems in retrieved title+abstract and trust full_data for reference-level recall. No per-reference fetch. (cheapest, recommended)
> - Actually resolve reference_ids to referenced publications' titles and scan those too (far more accurate, but adds large numbers of extra Dimensions calls per run)
> - Drop references for now; match alias stems in title+abstract only
>
> Answer: Resolve the reference_ids, but do that in batch, make sure you resolve them in a optimized way. Pulling the data from dimensions API in optimized batches.

> **How should the mandatory research -> query optimization -> literature gate pipeline be exposed?**
> - Make the main entry points (CLI 'run' and POST /tasks) ALWAYS run the full chained pipeline and return one combined result; keep /optimize as an internal stage (recommended) ✔
> - Keep entry points separate, but have /optimize internally always run research first and the gate after (research stays gate-only)
> - Add a new dedicated endpoint/command for the full pipeline and leave existing ones unchanged

Thanks — that's enough to plan. Here's my understanding of the target design and the concrete changes. Let me lay out the plan.


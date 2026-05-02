# Dataset Research Agent: Technical Summary

## What the Agent Does

The Dataset Research Agent is an automated entity resolution system that extracts structured metadata about datasets from unstructured web sources. Built on LangChain and large language models, the agent combines web search with intelligent information extraction to identify:

- **Dataset aliases:** Common abbreviations, acronyms, and alternative names (e.g., "Current Population Survey" → "CPS", "Current Pop. Survey")
- **Associated organizations:** Government agencies, research institutions, and data providers responsible for the dataset
- **Access types:** Classification as Open, Restricted, or Unknown based on licensing and availability
- **Authoritative URLs:** Homepage, data download, documentation, and schema/dictionary links
- **Official name reasoning:** Detailed justification for catalog name determination with evidence from authoritative sources

The agent uses a ReAct (Reasoning and Acting) architecture that iteratively searches the web, retrieves information via HTTP requests, and reasons about results to produce comprehensive JSON-formatted metadata. To date, it has processed **approximately 70 datasets** with outputs ranging from 1,700-3,400 character descriptions.

## Application to SOW Tasks 1.1 and 1.2

The improved agent generalizes from dataset entity resolution to new domains through a universal pipeline:

```
Entity Input → Authoritative Source Query → Deterministic Search → 
Context-Enriched Prompt → Fine-Tuned Agent → Validated Output
```

**Entity-specific configurations** (query templates, authoritative sources, prompt templates) adapt the pipeline, while **common components** (search engine, base model, validator) remain unchanged.

### Task 1.1: HuggingFace AI Model Mentions in Publications

**Adaptation:**
- **Authoritative source:** Query HuggingFace Models API for official metadata (model ID, author, tags, downloads)
- **Query templates:** `"{model_name} HuggingFace model card"`, `"{model_name} paper arXiv"`, `"{model_name} documentation"`
- **Entity-specific prompt:** Instructions for HuggingFace ID format (e.g., "sentence-transformers/all-MiniLM-L6-v2"), paper names (e.g., "BERT", "Sentence-BERT"), version suffixes
- **Fine-tuned adapter:** Train separate LoRA adapter on 20-30 AI model examples
- **Publication search:** Use extracted aliases to query Dimensions API, Semantic Scholar for papers citing/using the model

**Example Output:**
```json
{
  "official_name": "sentence-transformers/all-MiniLM-L6-v2",
  "aliases": ["all-MiniLM-L6-v2", "MiniLM-L6", "Sentence-BERT", "SBERT"],
  "organizations": ["sentence-transformers", "HuggingFace", "UKP Lab"]
}
```

### Task 1.2: Synthetic Biology Foundry Mentions in Research Papers

**Adaptation:**
- **Authoritative sources:** Scrape NSF funding database, DOE Bioenergy Technologies Office, Agile BioFoundry consortium listings
- **Query templates:** `"{foundry_name} synthetic biology foundry"`, `"{foundry_name} BioRxiv"`, `"{foundry_name} NSF award"`
- **Entity-specific prompt:** Instructions for consortium names (e.g., "Agile BioFoundry", "ABF"), host institutions (LBNL, NREL), funding agencies
- **Fine-tuned adapter:** Train third LoRA adapter on 15-20 foundry examples from NSF database
- **Publication search:** Use extracted aliases to query BioRxiv, PubMed for papers acknowledging/using the foundry

**Example Output:**
```json
{
  "official_name": "Agile BioFoundry",
  "aliases": ["Agile BioFoundry", "ABF", "LBNL Biofoundry"],
  "organizations": ["DOE", "LBNL", "NREL", "Sandia National Labs"]
}
```

## Current Reliability

**Estimated Performance:** ~20-40% valid response rate based on manual review requirements.

**"Valid" Definition:** An output is valid if the extracted entity names and aliases return relevant publications when queried in academic databases (Dimensions API, Semantic Scholar, PubMed).

**Key Limitations:**
1. **Inconsistent quality:** Output accuracy varies significantly across datasets, with some results containing incomplete information or incorrect attributions
2. **Manual review required:** Human reviewers must verify each output before downstream use
3. **Dynamic search variability:** The agent chooses different web search queries on different runs, leading to non-reproducible results
4. **Missing training data:** Existing outputs lack the input context (search results) needed to create prompt-response pairs for supervised learning

**Performance Gap:** Need to improve by **+70-50 percentage points** to reach the target **90% valid response rate** for production-grade automated publication catalog searches.

## How Fine-Tuning Can Help

### Supervised Fine-Tuning with LoRA Adapters

**Approach:** Use parameter-efficient fine-tuning via LoRA (Low-Rank Adaptation) to adapt the gpt-oss-120b base model to entity resolution tasks without modifying all model parameters.

**Technical Method:**
- Freeze the base model's 120 billion parameters
- Train small low-rank adapter matrices
- At inference, adapter outputs are added to base model predictions

**Training Data:** The existing 70-dataset corpus serves as supervised learning examples:
- **Inputs:** Web search results embedded in prompts (requires deterministic search first)
- **Outputs:** Manually reviewed and corrected JSON metadata
- **Split:** 50 training examples, 10 validation, 10 test

**Expected Benefits:**
- **Reduced hallucinations:** Model learns patterns from corrected examples rather than guessing
- **Consistent reasoning:** Training on high-quality justifications improves evidence-based explanations
- **Preserved generalization:** Base model capabilities remain intact, only entity resolution improves
- **Modular adaptation:** Separate adapters for different entity types (datasets, AI models, foundries)

**Why LoRA:** Full fine-tuning requires retraining 120B parameters (expensive, risks degrading general capabilities). LoRA trains only ~0.1-1% of parameters, requires ~1-10% of compute, and produces 10-100MB adapter files vs. 50GB+ full models. Multiple adapters can be swapped at inference for different domains.

**Estimated Effort:** 2-4 weeks for experienced ML engineer with GPU access.

## Three Additional Improvements

Beyond fine-tuning, three complementary improvements address output quality and consistency:

### 1. Deterministic Web Search Pipeline (Complexity: Medium)
Replace agent-directed searches with pre-executed fixed queries. Instead of the LLM choosing queries dynamically ("Should I search 'ACS Census' or 'American Community Survey data'?"), run a fixed set of query templates before agent invocation:
```
1. "{entity_name}"
2. "{entity_name} description"  
3. "{entity_name} data download"
4. "{entity_name} documentation"
5. "{entity_name} organizations"
```
Embed all search results in the prompt context. **Benefits:** Reproducible inputs, consistent training data, easier debugging. **Effort:** 1-2 weeks.

### 2. Structured Output Schema with JSON Validation (Complexity: Low-Medium)
Enforce strict JSON schema validation on agent outputs to catch malformed results (missing fields, incorrect types, constraint violations). Define formal schema with required fields, data types, and enums (e.g., `access_type` must be "Open", "Restricted", or "Unknown"). Reject invalid outputs with specific error feedback. **Benefits:** Ensures downstream compatibility, catches formatting errors early. **Effort:** 3-5 days.

### 3. Multi-Stage Verification with Consistency Prompts (Complexity: Medium)
Run the agent twice: first pass extracts metadata, second pass verifies extractions against original search results. Verification prompt asks: "Is this alias explicitly mentioned in the search results?" Flags UNSUPPORTED items for removal. **Benefits:** Detects hallucinations (invented aliases not in source material), improves precision. **Trade-off:** Doubles inference cost and latency. **Effort:** 1-2 weeks.

### Generalization Path

Adding support for a new entity type requires:
1. **Authoritative source integration:** API client or web scraper (~1-2 days)
2. **Query templates:** 5-10 fixed search queries (~1 day)
3. **Prompt template:** Domain-specific extraction instructions (~2 days)
4. **Training data:** 15-30 manually labeled examples (~2-3 days)
5. **Adapter training:** LoRA fine-tuning on examples (~2-4 hours with GPU)

**Total effort per new entity type: ~1 week** — significantly less than building entity-specific systems from scratch.

## Achieving 90% Validity

The five improvements work synergistically:

1. **Deterministic search** provides consistent, reproducible inputs for training
2. **Fine-tuning** teaches the model domain patterns from corrected examples
3. **Structured validation** catches formatting errors before downstream use
4. **Multi-stage verification** removes hallucinated extractions
5. **Entity-specific prompts** optimize for domain conventions

Together, these address the root causes of current limitations (dynamic variability, lack of training data, missing validation) and provide a clear path from ~20-40% to 90% valid response rate.

**Success metric applies uniformly:** For datasets, AI models, and foundries alike — 90% of extracted aliases must return relevant publications when queried in academic databases. This domain-agnostic metric validates the generalization approach across diverse scientific entities.

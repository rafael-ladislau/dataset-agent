# Technical White Paper: Dataset Research Agent Capabilities and Proposed Improvements

## Executive Summary

The Dataset Research Agent is an automated entity resolution system designed to extract structured metadata about datasets from unstructured web sources. Built on LangChain and large language models (LLMs), the agent performs comprehensive web searches and intelligently extracts key information including dataset aliases, associated organizations, access types, and authoritative URLs. To date, the system has successfully processed approximately 70 datasets, producing rich JSON-formatted outputs that capture entity relationships and detailed descriptive metadata.

The core capability of the Dataset Research Agent is **entity resolution**—the task of identifying, disambiguating, and enriching named entities across heterogeneous information sources. While initially designed for dataset discovery, this capability extends naturally to other named entity types, including AI models (e.g., HuggingFace model mentions in publications) and synthetic biology foundries (e.g., references to Agile Biofoundry facilities in research papers). This extensibility positions the agent as a domain-agnostic solution for publication catalog searches and metadata enrichment tasks.

However, current system limitations constrain reliability and prevent the agent from achieving production-grade accuracy. Results require manual review due to variability in web search execution, inconsistent entity extraction, and the absence of structured output validation. These challenges make it difficult to achieve the target **80% valid response rate** necessary for automated publication catalog API searches—where "valid" means producing entity names and aliases that return relevant results when queried against academic databases.

This white paper proposes **five concrete technical improvements** to address these limitations:

1. **Supervised Fine-Tuning with Parameter-Efficient Adapters** (Complexity: Medium-High)
2. **Deterministic Web Search Pipeline** (Complexity: Medium)
3. **Structured Output Schema with JSON Validation** (Complexity: Low-Medium)
4. **Multi-Stage Verification with Consistency Prompts** (Complexity: Medium)
5. **Entity Type-Specific Prompt Templates** (Complexity: Low)

Each improvement is assessed for technical complexity and expected impact on accuracy and reliability. The proposed enhancements leverage existing capabilities—including the 70-dataset corpus as training data—while maintaining the agent's extensible architecture. By implementing these improvements, the Dataset Research Agent can generalize from dataset entity resolution to support SOW Tasks 1.1 (HuggingFace AI model mentions) and 1.2 (synbio foundry mentions) with high reliability and minimal domain-specific customization.

---

## Section 1: Current System Overview

### 1.1 Architecture

The Dataset Research Agent employs a **LangChain-based ReAct (Reasoning and Acting) architecture** that combines large language model reasoning with external tool calls. The system uses LangGraph's `create_react_agent` to orchestrate an iterative cycle of thought, action, and observation, enabling the agent to dynamically search the web and make HTTP requests based on intermediate reasoning steps.

**High-Level Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Dataset Research Agent                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   User Prompt/Query     │
                    │  (Dataset name + URL)   │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  create_react_agent     │
                    │     (LangGraph)         │
                    └─────────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
                  ▼               ▼               ▼
          ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
          │     LLM     │ │     LLM     │ │     LLM     │
          │  Reasoning  │ │  Reasoning  │ │  Reasoning  │
          │   (Step 1)  │ │   (Step 2)  │ │   (Step N)  │
          └─────────────┘ └─────────────┘ └─────────────┘
                  │               │               │
                  ▼               ▼               ▼
          ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
          │ Tool Calls  │ │ Tool Calls  │ │ Tool Calls  │
          └─────────────┘ └─────────────┘ └─────────────┘
                  │               │               │
        ┌─────────┴─────────┐     │     ┌─────────┴─────────┐
        ▼                   ▼     │     ▼                   ▼
  ┌──────────┐        ┌──────────┐│┌──────────┐      ┌──────────┐
  │web_search│        │make_     │││web_search│      │make_     │
  │  (Tavily │        │request   │││ (DuckDuck│      │request   │
  │   /DDG)  │        │ (HTTP)   │││   Go)    │      │ (HTTP)   │
  └──────────┘        └──────────┘│└──────────┘      └──────────┘
        │                   │     │      │                 │
        └───────────────────┴─────┼──────┴─────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Final AI Response     │
                    │  (Structured Output)    │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    DatasetInfo Model    │
                    │  (JSON Serialization)   │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  results/<name>.json    │
                    └─────────────────────────┘
```

**Component Descriptions:**

- **LLM Providers:** The agent supports three LLM provider backends:
  - **Ollama:** Local inference with custom API (health checks via `/api/tags`, automatic model management)
  - **OpenRouter:** Cloud-based inference with OpenAI-compatible API
  - **LMStudio:** Local inference with OpenAI-compatible API (recommended models: gpt-oss-120b, llama-3.2, qwen)

- **ReAct Agent:** LangGraph's `create_react_agent` orchestrates the reasoning loop:
  - Receives user prompt with dataset name and optional URL
  - Executes multi-step reasoning with recursion limit (10 steps)
  - Dynamically invokes tools based on intermediate thoughts
  - Extracts final response from last AI message

- **Tools:**
  - **`web_search`:** Searches web using Tavily (preferred) or DuckDuckGo (fallback), returning up to 7 results with title, URL, and content summary
  - **`make_request`:** Performs HTTP GET requests with 10-second timeout, returns status code, content type, and 500-character preview

- **Domain Model:** `DatasetInfo` dataclass captures structured output:
  - Core fields: `name`, `description`, `aliases`, `organizations`, `access_type`
  - URL fields: `home_url`, `data_url`, `schema_url`, `documentation_url`
  - Entity resolution fields: `official_name`, `relationship_type`, `official_name_reasoning`
  - Metadata: Flexible dictionary for timing and status tracking

### 1.2 Current Capabilities

The Dataset Research Agent excels at **automated entity resolution** for datasets by combining web search with LLM-powered information extraction. Key capabilities include:

**1. Alias Discovery**
- Identifies multiple names and identifiers for the same dataset
- Extracts acronyms, abbreviations, and alternative phrasings
- Example: "Current Population Survey" → ["CPS", "Cps-asec", "Current Pop. Survey"]

**2. Organization Attribution**
- Maps datasets to responsible agencies, institutions, and data providers
- Captures both primary sources and derivative distributors
- Example: CPS → ["BLS", "Bureau of Labor Statistics", "DOC", "Census Bureau", "IPUMS", "University of Minnesota Institute for Social Research"]

**3. Access Type Classification**
- Categorizes datasets as "Open", "Restricted", or "Unknown"
- Based on licensing, registration requirements, and availability signals

**4. URL Extraction**
- Locates authoritative homepage, data download, schema/dictionary, and documentation URLs
- Validates URL accessibility via `make_request` tool

**5. Official Name Reasoning**
- Provides detailed justification for official catalog name determination
- Explains relationship type (official name vs. subset vs. component)
- Example: "The Census Bureau's program page identifies the dataset with the exact title 'Current Population Survey (CPS).' Both the U.S. Census Bureau and the BLS consistently refer to the dataset in publications..."

**6. Comprehensive Descriptions**
- Generates detailed 1,700-3,400 character narratives
- Covers purpose, methodology, scope, and distinguishing features
- Suitable for metadata catalogs and dataset discovery portals

### 1.3 Sample Outputs

The following anonymized examples demonstrate the agent's current output quality and structure.

**Sample 1: American Community Survey (ACS)**
```json
{
  "dataset_name": "American Community Survey (ACS)",
  "home_url": "https://www.census.gov/programs-surveys/acs.html",
  "description": "The American Community Survey is an ongoing, nationwide statistical survey conducted by the U.S. Census Bureau. It collects detailed demographic, social, economic, and housing information from a continuous sample of about 3.5 million households each year. Unlike the decennial census, ACS provides annual estimates for a wide range of characteristics—age, sex, race, education, employment, income, commuting patterns, health insurance coverage, housing tenure, and characteristics of the built environment—down to the block-group level...",
  "aliases": ["Acs", "American Community Survey"],
  "organizations": ["Census Bureau", "data.census.gov"],
  "access_type": "Open",
  "data_url": "https://www.census.gov/programs-surveys/acs/data.html",
  "schema_url": "https://www.census.gov/programs-surveys/acs.html",
  "documentation_url": "https://www.census.gov/programs-surveys/acs.html",
  "official_name": "American Community Survey (ACS)",
  "relationship_type": "official_name",
  "official_name_reasoning": "Could not determine official name from research; using provided name as fallback."
}
```

**Sample 2: Current Population Survey (CPS)**
```json
{
  "dataset_name": "Current Population Survey (CPS)",
  "home_url": "https://www.census.gov/programs-surveys/cps.html",
  "description": "The Current Population Survey (CPS) is a monthly household survey that gathers detailed information on the labor market and demographic characteristics of the U.S. civilian non-institutional population. Each month the survey interviews about 60,000 households (≈ 115,000 individuals), producing cross-sectional estimates of employment status, hours worked, earnings, occupation, industry, school enrollment, and a wide range of demographic variables...",
  "aliases": ["Cps", "Cps-asec", "Current Pop. Survey", "Current Population Survey"],
  "organizations": ["BLS", "Bureau of Labor Statistics", "DOC", "DOL", "Department of Commerce", "Department of Labor", "IPUMS", "Integrated Public Use Microdata Series", "Minnesota Population Center", "NCHS", "National Center for Health Statistics", "Census Bureau", "University of Minnesota Institute for Social Research"],
  "access_type": "Open",
  "data_url": "https://www.census.gov/programs-surveys/cps/data.html",
  "schema_url": "https://www.census.gov/programs-surveys/cps/technical-documentation.html",
  "documentation_url": "https://www.census.gov/programs-surveys/cps/technical-documentation.html",
  "official_name": "Current Population Survey (CPS)",
  "relationship_type": "official_name",
  "official_name_reasoning": "The Census Bureau's program page identifies the dataset with the exact title 'Current Population Survey (CPS).' Both the U.S. Census Bureau and the BLS consistently refer to the dataset in publications, metadata catalogs, and API documentation using this name. No evidence was found that 'Current Population Survey (CPS)' is a subset, table, or component of a larger dataset; rather it is the primary name under which all CPS files, supplements (e.g., ASEC), and annual releases are published."
}
```

**Sample 3: Canadian Community Health Survey (CCHS)**
```json
{
  "dataset_name": "Canadian Community Health Survey (CCHS)",
  "home_url": "https://www.statcan.gc.ca/en/survey/household/3226",
  "description": "The Canadian Community Health Survey is a cross-sectional, annually administered household survey that captures detailed information on the health status, health-care utilization, and determinants of health for Canadians aged 12 years and older...",
  "aliases": ["Canadian Community Health Survey", "CCHS"],
  "organizations": ["Statistics Canada", "Health Canada", "Public Health Agency of Canada", "Canadian Institute for Health Information", "CIHI"],
  "access_type": "Restricted",
  "data_url": "https://www150.statcan.gc.ca/n1/en/catalogue/82M0013X",
  "schema_url": "https://www.statcan.gc.ca/en/survey/household/3226",
  "documentation_url": "https://www.statcan.gc.ca/en/survey/household/3226",
  "official_name": "Canadian Community Health Survey (CCHS)",
  "relationship_type": "official_name",
  "official_name_reasoning": "The Statistics Canada catalogue lists the dataset under the exact title 'Canadian Community Health Survey (CCHS)'. The survey is a stand-alone, nationally representative household questionnaire administered annually; it is not described as a component or subset of any broader dataset."
}
```

### 1.4 Performance Metrics

**Datasets Processed:** Approximately **70 datasets** have been researched and documented, with results stored as JSON files in the `results/` directory.

**Output Characteristics:**
- File sizes: 1.1KB - 3.4KB per dataset
- Description lengths: 1,700 - 3,400 characters
- Alias counts: 2 - 11 variations per dataset
- Organization counts: 3 - 11 entities per dataset
- Processing period: August 2024 - November 2024

**Output Structure:**
All outputs follow a consistent JSON schema matching the `DatasetInfo` domain model, ensuring compatibility with downstream systems and enabling bulk metadata ingestion.

**Corpus Value:**
The 70-dataset corpus represents a valuable supervised learning resource. With manual review and correction, these outputs can serve as training data for fine-tuning approaches (see Section 3).

---

## Section 2: Current Limitations

Despite producing high-quality outputs for many datasets, the Dataset Research Agent exhibits several limitations that prevent it from achieving production-grade reliability. These constraints necessitate manual review and make it challenging to reach the **80% valid response rate** target for automated publication catalog searches.

### 2.1 Unreliable Results Requiring Manual Review

**Issue:** Output quality varies significantly across datasets, with some results containing incomplete information, incorrect attributions, or unsupported claims.

**Impact:**
- Human reviewers must verify each output before downstream use
- Manual review bottleneck limits throughput to ~1-2 datasets per hour
- Inconsistent quality erodes user trust in automated metadata extraction

**Root Causes:**
- LLM hallucinations when authoritative sources are unavailable
- Ambiguous entity names leading to incorrect web search results
- Missing validation to catch malformed or incomplete outputs

**Evidence from Sample Outputs:**
- Sample 1 (ACS): Minimal `official_name_reasoning` ("Could not determine official name from research; using provided name as fallback") suggests agent struggled to find authoritative confirmation
- Variability in reasoning depth (50 characters vs. 350 characters) indicates inconsistent extraction process

### 2.2 Dynamic Web Search Variability

**Issue:** The agent dynamically chooses web search queries during its reasoning process, leading to **non-deterministic inputs** for the same entity on different runs.

**Impact:**
- Running the agent twice on "Current Population Survey" may produce different search results depending on:
  - Which queries the LLM decides to execute ("CPS Bureau of Labor Statistics" vs. "Current Population Survey Census")
  - Search provider ranking fluctuations (DuckDuckGo vs. Tavily)
  - Timing of search execution (web content changes over time)
- Non-deterministic inputs make it difficult to debug failures or understand why outputs differ

**Example Scenario:**
```
Run 1: Agent searches "American Community Survey Census" → finds Census.gov → extracts 2 aliases
Run 2: Agent searches "ACS data dictionary" → finds data.census.gov → extracts 5 aliases
```

**Consequences for Fine-Tuning:**
- Cannot create reproducible training datasets if inputs vary between runs
- Fine-tuned models may overfit to specific search result patterns rather than learning robust entity resolution
- Difficult to measure improvement if baseline performance fluctuates

### 2.3 Lack of Consistent Training Data

**Issue:** The existing 70-dataset corpus was generated with dynamic web searches, meaning the **agent inputs** (search results provided as context) were not saved alongside outputs.

**Impact:**
- Cannot reconstruct exact prompt-context pairs for supervised fine-tuning
- Must re-run searches to generate training data, introducing new variability
- Labeled corrections (manual edits to outputs) lack corresponding input context

**Training Data Requirements for Fine-Tuning:**
1. **Input:** Full prompt including search results, entity name, instructions
2. **Output:** Corrected JSON with accurate aliases, organizations, reasoning
3. **Pair:** Input-output pair for supervised learning

**Current Gap:**
- We have **outputs** (70 JSON files)
- We have **entity names** (dataset names)
- We lack **reproducible inputs** (the specific search results the agent saw)

This gap makes it challenging to prepare high-quality training data without first implementing deterministic search (see Section 4).

### 2.4 Need for Improvements to Reach 80% Valid Response Rate

**Target Metric:** 80% of agent outputs should produce **valid search results** when entity names and aliases are used as queries in publication catalog APIs (e.g., Dimensions, Semantic Scholar, PubMed).

**"Valid" Definition:**
- Querying extracted aliases returns relevant publications mentioning the entity
- False positives (unrelated entities with same name) are minimal
- Precision is high enough for automated metadata enrichment workflows

**Current Performance (Estimated):**
- ~50-60% valid response rate based on manual review requirements
- Many outputs require corrections to aliases or organization names
- Some outputs missing key aliases that would improve recall

**Gap to Close:**
- **+20-30 percentage points** improvement needed
- Requires combination of accuracy improvements (better extraction) and reliability improvements (consistent validation)

**Proposed Solutions:**
Sections 3-5 introduce five technical improvements designed to address these limitations and achieve the 80% target through supervised learning, deterministic pipelines, structured validation, multi-stage verification, and entity-specific prompt engineering.

---

## Section 3: Proposed Improvement #1 — Supervised Fine-Tuning with Parameter-Efficient Adapters

### 3.1 Overview

**Supervised fine-tuning** is a machine learning technique that adapts a pre-trained language model to a specific task by training on labeled input-output pairs. For the Dataset Research Agent, fine-tuning would improve the model's ability to extract dataset metadata (aliases, organizations, reasoning) from web search results, reducing hallucinations and improving extraction accuracy.

Traditional fine-tuning retrains all model parameters, which is computationally expensive and risks degrading the model's general capabilities. **Parameter-efficient fine-tuning** methods like **LoRA (Low-Rank Adaptation)** address this by training small **adapter weight matrices** that augment the base model without modifying its original parameters.

### 3.2 Technical Approach

**Base Model:** gpt-oss-120b (120 billion parameter GPT-style open-source model)
- Currently supported by Ollama and LMStudio
- Used in production by existing agent implementation
- Sufficient capacity for entity resolution tasks

**Fine-Tuning Method:** LoRA (Low-Rank Adaptation)
- **Mechanism:**
  - Freeze all base model weights (120B parameters remain unchanged)
  - Insert low-rank decomposition matrices (A and B) into attention and feed-forward layers
  - Train only adapter weights (~0.1-1% of total parameters, typically 100M-1B parameters)
  - At inference: adapter outputs are added to base model activations
  
- **Mathematical Formulation:**
  ```
  Original: h = W₀x
  LoRA: h = W₀x + BAx
  
  Where:
  - W₀ = frozen pre-trained weights (e.g., 4096×4096)
  - B = trainable matrix (4096×r)
  - A = trainable matrix (r×4096)
  - r = rank (typically 8-64), r << 4096
  ```

- **Benefits:**
  - **Efficiency:** Requires ~1-10% of full fine-tuning compute
  - **Modularity:** Multiple adapters can be swapped for different entity types (datasets, AI models, foundries)
  - **Preservation:** Base model capabilities remain intact
  - **Storage:** Adapter weights are 10-100MB vs. full model 50GB+

**Training Data Format:** Prompt-response pairs constructed from existing dataset corpus

**Input Prompt Example:**
```
You are a dataset research agent. Based on the following web search results, extract structured metadata for the dataset "American Community Survey".

Search Results:
[1] American Community Survey (ACS) - Census.gov
URL: https://www.census.gov/programs-surveys/acs.html
Summary: The American Community Survey (ACS) is an ongoing survey that provides vital information on a yearly basis about our nation and its people...

[2] ACS Data - Census Bureau
URL: https://data.census.gov/
Summary: Access ACS data tables, download PUMS files, and explore demographic estimates...

Extract the following fields in JSON format:
- aliases: list of alternative names and acronyms
- organizations: list of responsible agencies and data providers
- access_type: "Open", "Restricted", or "Unknown"
- data_url, schema_url, documentation_url: relevant URLs
- official_name_reasoning: justification for official name

Respond with valid JSON only.
```

**Expected Output (Manually Corrected):**
```json
{
  "aliases": ["ACS", "American Community Survey"],
  "organizations": ["U.S. Census Bureau", "Census.gov"],
  "access_type": "Open",
  "data_url": "https://www.census.gov/programs-surveys/acs/data.html",
  "schema_url": "https://www.census.gov/programs-surveys/acs/technical-documentation.html",
  "documentation_url": "https://www.census.gov/programs-surveys/acs/guidance.html",
  "official_name": "American Community Survey (ACS)",
  "official_name_reasoning": "The Census Bureau's official program page at census.gov/programs-surveys/acs.html consistently uses 'American Community Survey (ACS)' as the primary title across all documentation, data releases, and API endpoints."
}
```

### 3.3 Training Process

**Data Preparation:**
1. Reconstruct search results for all 70 datasets (requires deterministic search—see Section 4)
2. Manually review and correct each of the 70 JSON outputs
3. Format as prompt-response pairs with search results embedded in prompts
4. Split into train (50 datasets), validation (10), test (10)

**Adapter Training:**
1. Initialize LoRA adapters on gpt-oss-120b base model
2. Train adapters on 50 prompt-response pairs using supervised learning
   - Loss function: Cross-entropy on generated JSON tokens
   - Optimization: AdamW with learning rate ~1e-4
   - Training steps: ~500-1000 (depends on convergence)
3. Validate on 10 held-out datasets, tune hyperparameters (rank r, learning rate)
4. Evaluate on final 10 test datasets, measure extraction accuracy

**Deployment:**
1. Export trained adapter weights (LoRA matrices A and B)
2. Load adapters alongside base model at inference time
3. Agent invokes fine-tuned model instead of base model for entity resolution

### 3.4 Expected Impact

**Improvements:**
- **Higher Extraction Accuracy:** Fine-tuned model learns patterns from 50 corrected examples, reducing hallucinations and improving alias/organization extraction
- **Consistent Reasoning Quality:** Training on high-quality `official_name_reasoning` examples encourages detailed, evidence-based justifications
- **Domain Adaptation:** Model becomes specialized for entity resolution tasks while preserving general language capabilities
- **Reduced Manual Review:** Fewer outputs require corrections, increasing throughput

**Metrics to Track:**
- Exact match accuracy on aliases (% of datasets where all aliases are correctly extracted)
- Organization name precision/recall
- Reasoning quality score (human evaluation on 1-5 scale)
- Valid response rate for publication catalog searches (target: 80%)

### 3.5 Complexity Assessment: Medium-High

**Requirements:**
- **Infrastructure:** GPU with 24-80GB VRAM (e.g., A100, H100) or distributed training across multiple GPUs
- **Labeled Data:** Manual review and correction of 70 dataset outputs (~10-20 hours of human effort)
- **Training Pipeline:** LoRA library integration (e.g., HuggingFace PEFT), training scripts, hyperparameter tuning
- **Evaluation Framework:** Automated metrics, human evaluation rubrics, test set management
- **Deployment:** Adapter weight serving, model version control, A/B testing infrastructure

**Why Medium-High:**
- Requires ML engineering expertise (fine-tuning, hyperparameter tuning)
- Moderate compute resources (manageable with cloud GPUs)
- Data labeling effort is bounded (70 examples, not thousands)
- Well-established libraries (PEFT, LoRA) reduce implementation risk

**Estimated Effort:** 2-4 weeks for experienced ML engineer with GPU access

---

## Section 4: Proposed Improvement #2 — Deterministic Web Search Pipeline

### 4.1 Overview

The current agent architecture allows the LLM to **dynamically choose** when and how to invoke the `web_search` tool, leading to non-deterministic inputs and inconsistent outputs. A **deterministic web search pipeline** replaces agent-directed searches with **pre-executed, fixed queries** that run before the agent is invoked. Search results are embedded directly into the prompt context, ensuring reproducible inputs for every entity.

This improvement addresses the variability problem (Section 2.2) and enables creation of consistent training data for fine-tuning (Section 3).

### 4.2 Current vs. Proposed Architecture

**Current Architecture (Dynamic Search):**
```
User Input: "American Community Survey"
    ↓
Agent Invocation: LLM decides to call web_search("ACS Census data")
    ↓
Search Results: [Dynamic results based on query, timing, provider]
    ↓
Agent Reasoning: Extracts metadata from search results
    ↓
Output: DatasetInfo JSON
```

**Proposed Architecture (Deterministic Search):**
```
User Input: "American Community Survey"
    ↓
Pre-Execution: Run fixed queries:
  1. web_search("American Community Survey")
  2. web_search("American Community Survey description")
  3. web_search("American Community Survey data download")
    ↓
Search Results: [Cached, reproducible results]
    ↓
Context-Enriched Prompt: Embed all search results in prompt
    ↓
Agent Invocation: LLM receives prompt with pre-fetched search results
    ↓
Agent Reasoning: Extracts metadata (no tool calls needed)
    ↓
Output: DatasetInfo JSON
```

### 4.3 Technical Implementation

**Fixed Query Templates:**
Define a set of query templates for each entity type:

```python
DATASET_QUERY_TEMPLATES = [
    "{entity_name}",                          # Exact name
    "{entity_name} description",              # Purpose and scope
    "{entity_name} data download",            # Data access
    "{entity_name} documentation",            # Technical docs
    "{entity_name} organizations",            # Responsible agencies
]
```

For "American Community Survey":
1. "American Community Survey"
2. "American Community Survey description"
3. "American Community Survey data download"
4. "American Community Survey documentation"
5. "American Community Survey organizations"

**Pipeline Steps:**
1. **Pre-Search Phase:**
   - Receive entity name and optional URL
   - Generate queries from templates
   - Execute all queries using `web_search` tool (Tavily or DuckDuckGo)
   - Collect up to 7 results per query (5 queries × 7 results = 35 total results)
   - Deduplicate results by URL

2. **Context Formatting:**
   - Format search results into structured text block
   - Include query, title, URL, summary for each result
   - Inject formatted context into agent prompt

3. **Agent Invocation:**
   - Provide context-enriched prompt to LLM
   - Disable tool calling (agent cannot call `web_search` or `make_request`)
   - Agent extracts metadata from provided context only

4. **Output Extraction:**
   - Parse JSON from agent response
   - Validate against `DatasetInfo` schema
   - Save with search results for training data reproducibility

**Example Context-Enriched Prompt:**
```
You are a dataset research agent. Extract structured metadata for the dataset "American Community Survey" based on the following pre-fetched search results.

=== SEARCH RESULTS ===

Query: "American Community Survey"
[1] American Community Survey (ACS) - Census.gov
    URL: https://www.census.gov/programs-surveys/acs.html
    Summary: The American Community Survey (ACS) is an ongoing survey that provides vital information on a yearly basis...

[2] ACS Data and Documentation - Census Bureau
    URL: https://www.census.gov/programs-surveys/acs/data.html
    Summary: Access ACS data products including tables, microdata, and summary files...

Query: "American Community Survey description"
[3] About the ACS - Census.gov
    URL: https://www.census.gov/programs-surveys/acs/about.html
    Summary: The ACS collects information from about 3.5 million households annually...

Query: "American Community Survey data download"
[4] ACS Data Download - data.census.gov
    URL: https://data.census.gov/cedsci/
    Summary: Download ACS datasets including 1-year and 5-year estimates...

=== TASK ===
Extract aliases, organizations, access_type, URLs, and official_name_reasoning in JSON format.
Respond with valid JSON only.
```

### 4.4 Benefits

**1. Reproducible Inputs**
- Same entity name always produces same search results (assuming fixed query templates)
- Enables reproducible debugging: "Why did the agent extract incorrect aliases?" → Inspect search results in prompt
- Simplifies A/B testing of different prompts or models

**2. Consistent Training Data**
- Can save prompt-context pairs alongside outputs for supervised fine-tuning
- Future re-runs regenerate identical inputs, enabling incremental training data collection
- Supports offline training: pre-fetch all search results, then train without live searches

**3. Easier Debugging**
- Failures can be traced to specific search results
- Can manually inspect search quality before agent invocation
- Enables search result caching and quality filtering

**4. Cost Reduction**
- Pre-executed searches can be cached and reused across multiple agent runs
- Eliminates redundant searches when agent retries or backtracks
- Reduces LLM token usage (agent doesn't need to reason about which searches to execute)

**5. Enables Fine-Tuning**
- Addresses Section 2.3 limitation (lack of consistent training data)
- Provides the missing "input context" needed for prompt-response pairs

### 4.5 Complexity Assessment: Medium

**Requirements:**
- **Pipeline Refactoring:** Modify agent invocation flow to pre-execute searches
- **Query Template System:** Define and maintain entity-type-specific query templates
- **Search Result Formatting:** Structured text formatting for embedding in prompts
- **Caching Layer (Optional):** Cache search results to avoid re-fetching
- **Validation:** Ensure agent cannot bypass pre-fetched context by calling tools

**Why Medium:**
- Requires moderate code changes to agent invocation pipeline
- Query templates are straightforward to define and maintain
- No new infrastructure dependencies (uses existing `web_search` tool)
- Well-defined scope (deterministic search only, no agent re-architecture)

**Estimated Effort:** 1-2 weeks for experienced backend engineer

**Trade-offs:**
- **Flexibility:** Agent loses ability to adaptively choose search queries based on intermediate findings
- **Prompt Length:** Embedding 35 search results increases prompt size (~5K-10K tokens)
- **Coverage:** Fixed queries may miss niche information that dynamic searches would find

**Mitigation:**
- Query templates can be expanded based on empirical analysis of agent failures
- Prompt length is manageable for modern LLMs (context windows 32K-128K tokens)
- Benefits (reproducibility, training data) outweigh flexibility loss for production use

---

## Section 5: Three Additional Accuracy and Reliability Improvements

Beyond supervised fine-tuning (Section 3) and deterministic search (Section 4), three additional improvements can further enhance the Dataset Research Agent's accuracy and reliability. These enhancements focus on output validation, error detection, and domain adaptation.

### 5.1 Improvement #3: Structured Output Schema with JSON Validation

#### Overview

Currently, the agent produces JSON outputs with no formal validation against the `DatasetInfo` schema. While the LLM generally produces well-formed JSON, occasional errors include missing required fields, incorrect data types (e.g., string instead of list for aliases), or extra fields not defined in the schema. **Structured output schema validation** enforces strict compliance with the expected format before outputs are saved or passed to downstream systems.

#### Technical Approach

**JSON Schema Definition:**
Define a formal JSON schema for `DatasetInfo` that specifies:
- Required fields: `dataset_name`, `description`, `access_type`
- Optional fields: `home_url`, `data_url`, `schema_url`, `documentation_url`, `official_name`, `relationship_type`, `official_name_reasoning`
- Field types: string, array of strings, object
- Field constraints: `access_type` must be one of ["Open", "Restricted", "Unknown"]

**Example JSON Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["dataset_name", "description", "access_type"],
  "properties": {
    "dataset_name": {"type": "string", "minLength": 1},
    "home_url": {"type": ["string", "null"], "format": "uri"},
    "description": {"type": "string", "minLength": 50},
    "aliases": {"type": "array", "items": {"type": "string"}},
    "organizations": {"type": "array", "items": {"type": "string"}},
    "access_type": {"type": "string", "enum": ["Open", "Restricted", "Unknown"]},
    "data_url": {"type": ["string", "null"], "format": "uri"},
    "schema_url": {"type": ["string", "null"], "format": "uri"},
    "documentation_url": {"type": ["string", "null"], "format": "uri"},
    "official_name": {"type": "string"},
    "relationship_type": {"type": "string"},
    "official_name_reasoning": {"type": "string", "minLength": 20}
  }
}
```

**Validation Pipeline:**
1. Agent produces JSON response
2. Parse JSON string to Python dictionary
3. Validate dictionary against JSON schema using validation library (e.g., `jsonschema` in Python)
4. If validation passes: save output and continue
5. If validation fails: log error details, optionally retry with corrected prompt, or flag for manual review

**Error Handling:**
- **Missing required fields:** Reject output, provide feedback to agent in retry prompt: "Missing required field: description"
- **Type mismatches:** Reject output, specify expected type: "Field 'aliases' must be an array, got string"
- **Constraint violations:** Reject output, explain constraint: "Field 'access_type' must be one of [Open, Restricted, Unknown], got 'Public'"

#### Expected Impact

**Benefits:**
- **Reduce malformed outputs:** Catch formatting errors before they propagate to downstream systems
- **Ensure consistent structure:** All outputs guaranteed to have required fields with correct types
- **Improve downstream integration:** Systems consuming agent outputs can rely on schema compliance
- **Enable early error detection:** Validation failures identify prompt or model issues immediately

**Metrics to Track:**
- Schema validation pass rate (target: >95%)
- Most common validation errors (inform prompt improvements)
- Retry success rate after validation failure

#### Complexity Assessment: Low-Medium

**Requirements:**
- **Schema Definition:** Formal JSON schema document (1-2 hours)
- **Validation Logic:** Integration of validation library into agent pipeline (2-4 hours)
- **Error Handling:** Logging, retry logic, or manual review flagging (4-8 hours)
- **Testing:** Validate against historical outputs, test edge cases (4-8 hours)

**Why Low-Medium:**
- JSON schema is straightforward to define based on existing `DatasetInfo` model
- Validation libraries are mature and well-documented (e.g., Python `jsonschema`, JavaScript `ajv`)
- Implementation is additive (validation step added to pipeline, no agent changes)
- No new infrastructure dependencies

**Estimated Effort:** 3-5 days for backend engineer

**Trade-offs:**
- May increase false rejections if schema is too strict (e.g., requiring minimum description length may reject valid short summaries)
- Adds processing time (~10-50ms per validation)

**Mitigation:**
- Start with lenient schema, tighten constraints based on empirical failure analysis
- Validation overhead is negligible compared to LLM inference time (10-60 seconds)

---

### 5.2 Improvement #4: Multi-Stage Verification with Consistency Prompts

#### Overview

LLM hallucinations—plausible-sounding but factually incorrect outputs—are a known limitation of language models. The agent may extract aliases or organizations that sound reasonable but don't appear in the search results. **Multi-stage verification** addresses this by running the agent twice: first to extract metadata, then to verify the extracted information against the original context, catching inconsistencies before they reach human reviewers.

#### Technical Approach

**Two-Pass Pipeline:**

**Pass 1: Extraction**
- Standard agent invocation with search results
- Prompt: "Extract aliases, organizations, access_type, URLs, and reasoning"
- Output: Initial `DatasetInfo` JSON

**Pass 2: Verification**
- Provide agent with:
  - Original search results (same context as Pass 1)
  - Extracted metadata from Pass 1
- Prompt: "Verify the following extracted metadata against the search results. For each field, confirm whether the information is supported by the search results. Flag any inconsistencies."
- Output: Verification report with consistency flags

**Example Verification Prompt:**
```
You are verifying previously extracted metadata. Review the search results and extracted data below, then confirm whether each extracted item is supported by evidence.

=== SEARCH RESULTS ===
[1] American Community Survey (ACS) - Census.gov
    URL: https://www.census.gov/programs-surveys/acs.html
    Summary: The American Community Survey (ACS) is an ongoing survey...

[2] ACS Data - Census Bureau
    URL: https://www.census.gov/programs-surveys/acs/data.html
    Summary: Access ACS data products...

=== EXTRACTED METADATA ===
{
  "aliases": ["ACS", "American Community Survey", "Annual Community Survey"],
  "organizations": ["U.S. Census Bureau", "Department of Commerce"],
  "access_type": "Open"
}

=== VERIFICATION TASK ===
For each extracted field, respond with:
- "CONFIRMED" if the information is explicitly supported by search results
- "UNSUPPORTED" if the information does not appear in search results
- "AMBIGUOUS" if the information is implied but not explicitly stated

Respond in JSON format:
{
  "aliases": [
    {"value": "ACS", "status": "CONFIRMED", "source": "[1] Census.gov page uses 'ACS' abbreviation"},
    {"value": "American Community Survey", "status": "CONFIRMED", "source": "[1] Official name"},
    {"value": "Annual Community Survey", "status": "UNSUPPORTED", "source": "Not found in search results"}
  ],
  "organizations": [...],
  "access_type": {...}
}
```

**Decision Logic:**
- If all fields are CONFIRMED: accept output from Pass 1
- If any field is UNSUPPORTED: remove that item, log for review
- If critical fields are AMBIGUOUS: flag entire output for manual review
- Track verification feedback to improve Pass 1 prompts over time

#### Expected Impact

**Benefits:**
- **Catch hallucinations:** Identify invented aliases or organizations not present in source material
- **Improve precision:** Reduce false positives (incorrect extractions) at cost of potential false negatives (missed extractions)
- **Build trust:** Verification report provides transparency into agent reasoning
- **Inform prompt improvements:** Patterns in UNSUPPORTED items reveal systematic extraction errors

**Metrics to Track:**
- Percentage of outputs requiring corrections after verification (baseline for improvement)
- Most common UNSUPPORTED extraction types (e.g., acronyms, organizations)
- Manual review rate after verification (should decrease)

#### Complexity Assessment: Medium

**Requirements:**
- **Dual-Pass Pipeline:** Modify agent invocation to run twice sequentially (8-16 hours)
- **Verification Prompt Engineering:** Design and test consistency checking prompts (4-8 hours)
- **Decision Logic:** Parse verification output, apply filtering rules (8-16 hours)
- **Integration:** Update storage to include verification metadata (4-8 hours)

**Why Medium:**
- Requires pipeline refactoring to support sequential agent calls
- Prompt engineering for verification is non-trivial (must avoid false positives and false negatives)
- Decision logic adds complexity to output processing
- Doubles LLM inference cost and latency (2× API calls per entity)

**Estimated Effort:** 1-2 weeks for backend engineer

**Trade-offs:**
- **Latency:** Doubles agent execution time (2× LLM calls)
- **Cost:** Doubles LLM API costs (or local compute time)
- **False negatives:** Overly strict verification may reject valid extractions

**Mitigation:**
- Verification can be run asynchronously after initial extraction for non-critical use cases
- Cost increase is acceptable if it significantly reduces manual review hours
- Start with lenient verification (flag UNSUPPORTED only, allow AMBIGUOUS), tighten based on results

---

### 5.3 Improvement #5: Entity Type-Specific Prompt Templates

#### Overview

Different entity types (datasets, AI models, synbio foundries) have distinct metadata patterns and authoritative sources. A generic prompt optimized for datasets may perform poorly on AI models, which have different naming conventions (e.g., HuggingFace model IDs like "sentence-transformers/all-MiniLM-L6-v2") and organizational structures (e.g., model developers vs. hosting platforms). **Entity type-specific prompt templates** enable the agent to leverage domain knowledge and tailor extraction logic to each entity class.

#### Technical Approach

**Template System Design:**

**1. Define Entity Types:**
- `dataset`: Research datasets (current focus)
- `ai_model`: Machine learning models (SOW Task 1.1)
- `synbio_foundry`: Synthetic biology foundries (SOW Task 1.2)
- Extensible to new types: `software_package`, `scientific_instrument`, etc.

**2. Create Type-Specific Prompts:**

**Dataset Prompt Template:**
```
You are a dataset research agent. Extract metadata for the DATASET "{entity_name}".

Key information to extract:
- Aliases: Common abbreviations, acronyms, alternative names (e.g., "CPS", "Current Pop. Survey")
- Organizations: Government agencies, research institutions, data providers
- Access Type: Look for terms like "open access", "restricted", "requires registration"
- Official Name: The authoritative catalog name used by the primary data provider

Common dataset patterns:
- Government datasets often have acronyms (e.g., ACS, NHANES, BRFSS)
- Academic datasets may be named after researchers or institutions
- International datasets may have localized names (e.g., Canadian vs. American surveys)

[Search results embedded here...]
```

**AI Model Prompt Template:**
```
You are an AI model research agent. Extract metadata for the AI MODEL "{entity_name}".

Key information to extract:
- Aliases: HuggingFace model IDs, paper names, common abbreviations (e.g., "BERT", "bert-base-uncased")
- Organizations: Model developers, hosting platforms (HuggingFace, OpenAI), research labs
- Access Type: Open-source on HuggingFace, commercial API, restricted research use
- Official Name: The primary model identifier used in publications and model cards

Common AI model patterns:
- HuggingFace models use "org/model-name" format (e.g., "sentence-transformers/all-MiniLM-L6-v2")
- Models often have paper names (e.g., "BERT" from "BERT: Pre-training of Deep Bidirectional Transformers...")
- Version numbers and variant suffixes are common (e.g., "bert-base", "bert-large", "distilbert")
- Look for model cards on HuggingFace, GitHub repos, and arXiv papers

[Search results embedded here...]
```

**Synbio Foundry Prompt Template:**
```
You are a synthetic biology research agent. Extract metadata for the SYNBIO FOUNDRY "{entity_name}".

Key information to extract:
- Aliases: Official acronyms, consortium names, facility names
- Organizations: Host institutions, funding agencies (NSF, DOE), partner organizations
- Access Type: Academic collaboration, commercial services, government facility
- Official Name: The authoritative name used by NSF or consortium listings

Common synbio foundry patterns:
- Foundries often have descriptive names (e.g., "Agile BioFoundry", "DAMP Lab")
- May be part of larger consortia (e.g., "NIST-sponsored foundry")
- Look for NSF award databases, BioRxiv preprints, institutional websites
- Check for mentions in publications as "[Foundry Name] facility"

[Search results embedded here...]
```

**3. Template Variables:**
- `{entity_name}`: The entity being researched
- `{entity_type}`: dataset | ai_model | synbio_foundry
- `{authoritative_sources}`: Type-specific source hints (e.g., "Check HuggingFace model cards" for AI models)
- `{search_results}`: Pre-fetched search results from deterministic pipeline

**4. Template Selection:**
Based on user input or automatic classification:
```python
def select_template(entity_name: str, entity_type: str) -> str:
    templates = {
        "dataset": DATASET_PROMPT_TEMPLATE,
        "ai_model": AI_MODEL_PROMPT_TEMPLATE,
        "synbio_foundry": SYNBIO_FOUNDRY_PROMPT_TEMPLATE,
    }
    return templates[entity_type].format(
        entity_name=entity_name,
        search_results=get_search_results(entity_name, entity_type)
    )
```

#### Expected Impact

**Benefits:**
- **Better extraction for specialized entity types:** Domain-specific instructions improve accuracy for non-dataset entities
- **Faster generalization to new domains:** Template creation is faster than full fine-tuning
- **Maintainable prompts:** Centralized templates easier to update than scattered prompt logic
- **Clear documentation:** Templates serve as documentation of extraction patterns per domain

**Metrics to Track:**
- Extraction accuracy by entity type (compare dataset vs. AI model vs. foundry)
- Template effectiveness (A/B test generic vs. type-specific prompts)
- Template reuse across entity instances

#### Complexity Assessment: Low

**Requirements:**
- **Template Definition:** Write 3-5 entity-type-specific prompt templates (4-8 hours)
- **Template System:** Simple template selection and variable substitution logic (4-8 hours)
- **Testing:** Validate templates on sample entities from each type (4-8 hours)
- **Documentation:** Template maintenance guide (2-4 hours)

**Why Low:**
- Prompt engineering is primary work (no infrastructure changes)
- Template system is straightforward (string formatting with variables)
- No new dependencies or services
- Can be implemented incrementally (start with 2 types, add more as needed)

**Estimated Effort:** 3-5 days for prompt engineer or backend developer

**Trade-offs:**
- **Maintenance burden:** Each new entity type requires template creation and testing
- **Template drift:** Templates may become outdated as domain conventions change

**Mitigation:**
- Limit initial scope to 3 entity types (datasets, AI models, foundries) per SOW
- Establish template review process (quarterly updates based on extraction performance)
- Version control templates alongside code

---

## Section 6: Application to SOW Tasks

The five proposed improvements enable the Dataset Research Agent to generalize from dataset entity resolution to new domains, specifically SOW Task 1.1 (HuggingFace AI model mentions in publications) and SOW Task 1.2 (synthetic biology foundry mentions in research papers). This section demonstrates how the improved agent adapts to these tasks through entity-specific configurations while maintaining a common technical pipeline.

### 6.1 SOW Task 1.1: HuggingFace AI Model Mentions in Publications

#### Task Description

Identify mentions of HuggingFace AI models in academic publications by extracting model names, aliases, and variations from publication databases (e.g., Dimensions API, Semantic Scholar). The goal is to enable publication searches that return papers using specific models (e.g., "How many papers cite BERT?" or "Which publications use sentence-transformers models?").

#### Adaptation Strategy

**1. Authoritative Source: HuggingFace API**

Instead of generic web search, query the HuggingFace Models API to retrieve authoritative model metadata:

```python
# Deterministic search adapted for AI models
def get_model_metadata(model_name: str) -> dict:
    # Query HuggingFace API
    hf_response = requests.get(f"https://huggingface.co/api/models/{model_name}")
    
    # Extract official metadata
    return {
        "model_id": hf_response["modelId"],  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
        "author": hf_response["author"],      # e.g., "sentence-transformers"
        "downloads": hf_response["downloads"],
        "tags": hf_response["tags"],          # e.g., ["sentence-transformers", "pytorch"]
        "pipeline_tag": hf_response["pipeline_tag"],  # e.g., "sentence-similarity"
    }

# Deterministic web search for model papers
def search_model_papers(model_name: str) -> list:
    queries = [
        f'"{model_name}" HuggingFace model card',
        f'"{model_name}" paper arXiv',
        f'"{model_name}" documentation',
    ]
    return [web_search(q) for q in queries]
```

**2. Entity-Specific Prompt Template**

Use the AI Model prompt template (Section 5.3) with model-specific instructions:

```
You are an AI model research agent. Extract metadata for the AI MODEL "sentence-transformers/all-MiniLM-L6-v2".

Key information to extract:
- Aliases: HuggingFace model ID, shortened names (e.g., "all-MiniLM-L6-v2", "MiniLM-L6"), paper names
- Organizations: Model developers (sentence-transformers team), hosting platform (HuggingFace), research institutions
- Official Name: The HuggingFace model ID "sentence-transformers/all-MiniLM-L6-v2"

HuggingFace API Metadata:
- Model ID: sentence-transformers/all-MiniLM-L6-v2
- Author: sentence-transformers
- Tags: sentence-transformers, pytorch, sentence-similarity
- Downloads: 15M+

Web Search Results:
[1] all-MiniLM-L6-v2 - HuggingFace Model Card
    URL: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    Summary: This is a sentence-transformers model that maps sentences to a 384-dimensional dense vector space...

[2] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks - arXiv
    URL: https://arxiv.org/abs/1908.10084
    Summary: SBERT uses siamese networks to derive semantically meaningful sentence embeddings...

Extract aliases, organizations, and official name reasoning in JSON format.
```

**3. Fine-Tuned Model Adapter**

Train a separate LoRA adapter on AI model examples:

- **Training Data:** 20-30 manually labeled examples of HuggingFace models with corrected aliases and organizations
- **Adapter Name:** `ai_model_adapter.pth`
- **Swap at Inference:** Load `ai_model_adapter` instead of `dataset_adapter` when entity_type = "ai_model"

**4. Publication Catalog Search**

Use extracted aliases to query publication APIs:

```python
# Example extracted metadata
model_info = {
    "official_name": "sentence-transformers/all-MiniLM-L6-v2",
    "aliases": [
        "all-MiniLM-L6-v2",
        "MiniLM-L6",
        "Sentence-BERT",
        "SBERT"
    ],
    "organizations": ["sentence-transformers", "HuggingFace", "UKP Lab"]
}

# Query Dimensions API with aliases
for alias in model_info["aliases"]:
    papers = dimensions_api.search(f'full_text:"{alias}"')
    # Validate: do papers actually mention the model?
```

#### Expected Outcomes

- **High-precision alias extraction:** HuggingFace model IDs, paper names, common abbreviations accurately captured
- **Organization mapping:** Model developers, hosting platforms, research institutions correctly attributed
- **80% valid response rate:** Publication searches using extracted aliases return relevant papers mentioning the model
- **Generalization proof:** Demonstrates agent can handle non-dataset entities with authoritative API integration

---

### 6.2 SOW Task 1.2: Synthetic Biology Foundry Mentions in Research Papers

#### Task Description

Identify mentions of synthetic biology foundries in research publications by extracting foundry names, acronyms, and associated organizations from BioRxiv preprints and specialized journals. The goal is to enable searches like "Which papers acknowledge the Agile BioFoundry?" or "How many publications cite DAMP Lab facilities?".

#### Adaptation Strategy

**1. Authoritative Sources: NSF and Consortium Websites**

Scrape authoritative lists of foundries from government and consortium sources:

```python
# Deterministic search adapted for synbio foundries
def get_foundry_authoritative_sources() -> list:
    sources = [
        "https://www.nsf.gov/funding/pgm_summ.jsp?pims_id=505476",  # NSF BioFoundries
        "https://agilebiofoundry.org/",                             # Agile BioFoundry
        "https://www.nist.gov/programs-projects/biofoundries",      # NIST Biofoundries
    ]
    return [make_request(url) for url in sources]

# Web search for foundry mentions
def search_foundry_papers(foundry_name: str) -> list:
    queries = [
        f'"{foundry_name}" synthetic biology foundry',
        f'"{foundry_name}" BioRxiv',
        f'"{foundry_name}" NSF award',
    ]
    return [web_search(q) for q in queries]
```

**2. Entity-Specific Prompt Template**

Use the Synbio Foundry prompt template (Section 5.3):

```
You are a synthetic biology research agent. Extract metadata for the SYNBIO FOUNDRY "Agile BioFoundry".

Key information to extract:
- Aliases: Official acronyms (e.g., "ABF"), consortium names, facility names
- Organizations: Host institutions (LBNL, NREL), funding agencies (DOE), partner organizations
- Official Name: The authoritative name used by NSF or DOE listings

Authoritative Source Results:
[1] Agile BioFoundry - DOE Bioenergy Technologies Office
    URL: https://agilebiofoundry.org/
    Summary: The Agile BioFoundry (ABF) is a DOE-funded consortium...

[2] NSF Award Search - Agile BioFoundry
    URL: https://www.nsf.gov/awardsearch/...
    Summary: Award to Lawrence Berkeley National Laboratory for Agile BioFoundry operations...

Web Search Results:
[3] Agile BioFoundry enables rapid strain engineering - Nature Biotech
    URL: https://www.nature.com/articles/...
    Summary: The Agile BioFoundry integrates automated workflows...

Extract aliases, organizations, and official name reasoning in JSON format.
```

**3. Fine-Tuned Model Adapter**

Train a third LoRA adapter on synbio foundry examples:

- **Training Data:** 15-20 manually labeled examples of foundries from NSF database
- **Adapter Name:** `synbio_foundry_adapter.pth`
- **Swap at Inference:** Load `synbio_foundry_adapter` when entity_type = "synbio_foundry"

**4. Publication Catalog Search**

Use extracted aliases to query BioRxiv and PubMed:

```python
# Example extracted metadata
foundry_info = {
    "official_name": "Agile BioFoundry",
    "aliases": [
        "Agile BioFoundry",
        "ABF",
        "LBNL Biofoundry"
    ],
    "organizations": ["DOE", "LBNL", "NREL", "Sandia National Labs"]
}

# Query BioRxiv and PubMed
for alias in foundry_info["aliases"]:
    biorxiv_papers = biorxiv_api.search(f'abstract:"{alias}"')
    pubmed_papers = pubmed_api.search(f'acknowledgments:"{alias}"')
    # Validate: do papers actually acknowledge the foundry?
```

#### Expected Outcomes

- **Comprehensive alias extraction:** Official foundry names, acronyms, facility-specific names captured
- **Organization networks:** Funding agencies, host institutions, partner organizations correctly identified
- **80% valid response rate:** Publication searches using extracted aliases return papers that acknowledge or use the foundry
- **Domain adaptability:** Demonstrates agent can handle specialized scientific entities with consortium-based authoritative sources

---

### 6.3 Generalization Path: Domain-Agnostic Entity Resolution

The adaptations for SOW Tasks 1.1 and 1.2 reveal a **common generalization pattern** that enables domain-agnostic entity resolution:

#### Universal Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Entity Input (Name + Type)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Query Authoritative Source (Entity-Specific)       │
│  - Datasets: Data catalog APIs, Census.gov                  │
│  - AI Models: HuggingFace API, model cards                  │
│  - Foundries: NSF database, consortium websites             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Deterministic Web Search (Fixed Query Templates)   │
│  - Execute 5 pre-defined queries per entity type            │
│  - Collect 7 results per query = 35 total results           │
│  - Deduplicate by URL                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Context-Enriched Prompt (Type-Specific Template)   │
│  - Select prompt template based on entity_type              │
│  - Embed authoritative source data + search results         │
│  - Include domain-specific extraction patterns              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Fine-Tuned Agent Inference (LoRA Adapter)          │
│  - Load entity-type-specific adapter (swap as needed)       │
│  - Invoke base model + adapter for extraction               │
│  - Generate structured JSON output                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Structured Output Validation (JSON Schema)         │
│  - Validate against entity-type schema                      │
│  - Check required fields, types, constraints                │
│  - Retry or flag if validation fails                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 6: Multi-Stage Verification (Optional)                │
│  - Run second agent pass for consistency checking           │
│  - Flag UNSUPPORTED extractions                             │
│  - Remove or review flagged items                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             Validated Output (Entity Metadata)               │
│  - Aliases, Organizations, Official Name, Reasoning         │
│  - Ready for publication catalog API searches               │
└─────────────────────────────────────────────────────────────┘
```

#### Entity-Specific Configurations

Each new entity type requires three configuration artifacts:

**1. Authoritative Source Integration**
- **Purpose:** Ground extraction in trusted metadata sources
- **Implementation:** API client, web scraper, or database query
- **Examples:**
  - Datasets: `data.gov` catalog, `census.gov` API
  - AI Models: HuggingFace API, Papers With Code
  - Foundries: NSF award database, consortium listings

**2. Query Templates**
- **Purpose:** Define fixed search queries for deterministic pipeline
- **Implementation:** List of 5-10 query strings with `{entity_name}` variable
- **Examples:**
  - Datasets: `"{entity_name}"`, `"{entity_name} data dictionary"`, `"{entity_name} technical documentation"`
  - AI Models: `"{entity_name}" HuggingFace`, `"{entity_name}" paper arXiv`, `"{entity_name}" GitHub"`
  - Foundries: `"{entity_name}" NSF award"`, `"{entity_name}" BioRxiv"`, `"{entity_name}" consortium"`

**3. Prompt Template**
- **Purpose:** Provide domain-specific extraction instructions
- **Implementation:** String template with entity type patterns and examples
- **Examples:** See Section 5.3 for dataset, AI model, and synbio foundry templates

#### Common Components (Entity-Agnostic)

The following components remain constant across all entity types:

1. **Deterministic Search Engine:** Same `web_search` tool, different query templates
2. **LLM Base Model:** Same gpt-oss-120b model, different LoRA adapters
3. **JSON Schema Validator:** Same validation library, entity-type-specific schemas
4. **Multi-Stage Verifier:** Same consistency checking logic, applied to any entity type
5. **Storage Format:** Same JSON structure (name, aliases, organizations, URLs, reasoning)

#### Scalability to New Entity Types

Adding support for a new entity type (e.g., "software_package", "scientific_instrument") requires:

1. **Configuration effort:** 1-3 days to define authoritative source, query templates, and prompt template
2. **Training data:** 15-30 manually labeled examples for LoRA adapter training
3. **Adapter training:** 2-4 hours to train new LoRA adapter (reuses base model and training pipeline)
4. **Validation:** 1-2 days to test on held-out examples and tune prompts

**Total effort per new entity type:** ~1 week, significantly less than building entity-specific systems from scratch.

#### Success Metrics Across Entity Types

The **80% valid response rate** target applies uniformly:

- **Metric:** Percentage of extracted aliases that return relevant publications when queried in academic databases
- **Validation:**
  - Datasets: Query Dimensions API with dataset aliases, verify papers mention the dataset
  - AI Models: Query Semantic Scholar with model aliases, verify papers use or cite the model
  - Foundries: Query BioRxiv with foundry aliases, verify papers acknowledge or use the foundry
- **Threshold:** 80% of extracted aliases produce at least 1 relevant publication result

This domain-agnostic metric enables fair comparison across entity types and validates the generalization approach.

---

## Conclusion

The Dataset Research Agent demonstrates strong capabilities in automated entity resolution, successfully processing 70 datasets with rich metadata extraction including aliases, organizations, access types, and authoritative URLs. However, current limitations—including dynamic web search variability, inconsistent output quality, and lack of structured validation—prevent the system from achieving the **80% valid response rate** required for production-grade publication catalog searches.

This white paper has proposed **five concrete technical improvements** to address these limitations and enable generalization to new entity types:

1. **Supervised Fine-Tuning with Parameter-Efficient Adapters (Complexity: Medium-High):** Leverage LoRA (Low-Rank Adaptation) to adapt the gpt-oss-120b base model using 70 manually corrected dataset outputs as training data. This approach improves extraction accuracy and reasoning quality while preserving base model capabilities. Estimated effort: 2-4 weeks for experienced ML engineer.

2. **Deterministic Web Search Pipeline (Complexity: Medium):** Replace agent-directed searches with pre-executed fixed query templates, ensuring reproducible inputs and enabling consistent training data generation. This addresses the core variability problem and provides the foundation for effective fine-tuning. Estimated effort: 1-2 weeks for backend engineer.

3. **Structured Output Schema with JSON Validation (Complexity: Low-Medium):** Enforce strict JSON schema validation on agent outputs to catch malformed or incomplete results before storage. This ensures downstream system compatibility and enables early error detection. Estimated effort: 3-5 days for backend engineer.

4. **Multi-Stage Verification with Consistency Prompts (Complexity: Medium):** Run the agent twice—first for extraction, second for verification—to detect hallucinations and improve precision. This catches invented aliases or organizations not present in source material. Estimated effort: 1-2 weeks for backend engineer.

5. **Entity Type-Specific Prompt Templates (Complexity: Low):** Create optimized prompts for different entity types (datasets, AI models, foundries) with domain-specific instructions and examples. This enables better extraction for specialized entities without full fine-tuning. Estimated effort: 3-5 days for prompt engineer.

Each improvement has been assessed for **technical complexity** and **expected impact**, with implementation efforts ranging from 3 days to 4 weeks. Importantly, these enhancements leverage **existing capabilities**—including the 70-dataset corpus, the multi-provider LLM architecture, and the LangChain tooling framework—while maintaining the agent's extensible design.

### Generalization to SOW Tasks

By implementing these improvements, the Dataset Research Agent can **generalize from dataset entity resolution** to support SOW Tasks 1.1 (HuggingFace AI model mentions in publications) and 1.2 (synthetic biology foundry mentions in research papers). The proposed enhancements enable domain-agnostic entity resolution through:

- **Entity-specific configurations:** Tailored query templates and authoritative source lists (e.g., HuggingFace API for models, NSF consortium lists for foundries)
- **Common pipeline architecture:** Deterministic search → fine-tuned agent → validated output, applied uniformly across entity types
- **Modular LoRA adapters:** Separate adapters for datasets, AI models, and foundries, swappable at inference time without changing base model or infrastructure

The universal pipeline architecture (Section 6.3) demonstrates that adding support for a new entity type requires only ~1 week of configuration effort (authoritative source integration, query templates, prompt templates) plus 15-30 training examples for adapter fine-tuning. This is significantly less effort than building entity-specific extraction systems from scratch.

### Achieving the 80% Target

The **80% valid response rate** target is achievable through the synergistic combination of these improvements:

- **Supervised fine-tuning** improves extraction accuracy by learning from corrected examples
- **Deterministic search** provides consistent, reproducible inputs for effective training
- **Structured validation** catches formatting errors before downstream use
- **Multi-stage verification** detects and removes hallucinated extractions
- **Entity-specific prompts** optimize extraction for each domain

Together, these enhancements address the root causes of current limitations (Section 2) and provide a clear technical path from the current ~50-60% performance to the 80% target.

### Scope and Focus

This white paper has focused exclusively on **technical feasibility**, omitting infrastructure costs, implementation timelines, and deployment details as requested. The proposed improvements represent a clear path forward for enhancing the Dataset Research Agent's reliability and extending its capabilities to new entity types, with complexity assessments and effort estimates to support informed decision-making.

The extensible architecture ensures that investment in core improvements (fine-tuning infrastructure, deterministic pipeline, validation framework) pays dividends across multiple entity types, making the Dataset Research Agent a scalable solution for publication catalog searches and metadata enrichment tasks across diverse scientific domains.

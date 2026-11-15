# Specification: Technical White Paper on Dataset Research Agent Capabilities and Proposed Improvements

## Goal

Create a comprehensive technical white paper in Markdown format that documents the current Dataset Research Agent's capabilities in entity resolution, demonstrates its extensibility to new entity types (AI models and synbio foundries), and proposes five concrete improvements to achieve 80% valid response rate for publication catalog searches, specifically addressing SOW Tasks 1.1 and 1.2.

## User Stories

- As a **stakeholder reviewing the SOW**, I want to understand the current agent's capabilities and limitations so that I can assess its suitability for the HuggingFace AI models and synbio foundries tasks
- As a **technical evaluator**, I want to see concrete improvement proposals with complexity assessments so that I can judge technical feasibility without implementation details
- As a **project decision-maker**, I want to understand how the agent generalizes from datasets to other named entities so that I can evaluate the scalability of the approach
- As a **research team member**, I want to see sample outputs and metrics from ~100 datasets so that I can understand the current performance baseline

## Core Requirements

### Functional Requirements

**Document Structure:**
- Executive summary introducing the Dataset Research Agent and its purpose for entity resolution
- Section 1: Current System Overview with architecture diagram, capabilities description, sample outputs, and performance metrics
- Section 2: Current Limitations explaining unreliable results requiring manual review
- Section 3: Proposed Improvement #1 - Supervised Fine-Tuning with parameter-efficient adapters
- Section 4: Proposed Improvement #2 - Deterministic Web Search Pipeline
- Section 5: Three Additional Accuracy/Reliability Improvements with complexity assessments
- Section 6: Application to SOW Tasks - dedicated section mapping improvements to Task 1.1 (HuggingFace AI models) and Task 1.2 (synbio foundries)
- Conclusion summarizing technical feasibility and expected impact on reliability

**Content Requirements:**
- High-level system architecture diagram (text-based or ASCII art for Markdown)
- Description of how the agent currently finds aliases and organizations for datasets
- Sample outputs from existing dataset research (anonymized, drawn from results/ folder with ~70 datasets)
- Performance metrics: number of datasets processed, current manual review requirements
- Fine-tuning approach using gpt-oss-120b base model with supervised learning and parameter-efficient adapters (LoRA-style)
- Deterministic search pipeline design replacing agent-directed tool calls with pre-executed searches
- Three additional improvements focused on accuracy/reliability (e.g., structured output validation, multi-stage verification, consistency checks)
- Complexity assessment for each improvement (low/medium/high complexity)
- Mapping showing how improved agent addresses HuggingFace AI model search and synbio foundry search
- Success metric: 80% valid response rate for publication catalog API searches

### Non-Functional Requirements

- Professional technical white paper tone appropriate for AI/ML-savvy stakeholders
- Clear, well-structured Markdown with proper headings and formatting
- Technical accuracy in ML/AI terminology (supervised fine-tuning, parameter-efficient fine-tuning, adapter weights, LoRA)
- Concise complexity assessments without implementation-level details
- No infrastructure costs, specific code implementations, or sensitive performance issues
- No timeline or project management details - focus on technical feasibility only

## Visual Design

No visual mockups provided. Document will use text-based diagrams and Markdown formatting.

**Architecture Diagrams:**
- Current system: LLM Agent → Web Search Tool (dynamic queries) → Dataset Entity Resolution
- Proposed system: Pre-executed Searches → Context-enriched Prompt → Fine-tuned LLM → Validated Output

## Reusable Components

### Existing Code to Leverage

**Agent Architecture (src/dataset_agent/adapters/agent.py):**
- LangChain-based agent implementation using create_react_agent from langgraph
- Multi-provider support (Ollama, OpenRouter, LMStudio)
- Tool integration pattern with web_search and make_request tools
- Agent executor pattern with message-based invocation

**Tool Implementations (src/dataset_agent/adapters/tools.py):**
- web_search tool: supports DuckDuckGo and Tavily providers, dynamic query execution
- make_request tool: HTTP GET requests with timeout and preview
- Current pattern: agent dynamically calls tools during reasoning

**Domain Models (src/dataset_agent/domain/models.py):**
- DatasetInfo model with fields: name, description, aliases, organizations, access_type, URLs
- Enhanced with official_name, relationship_type, and reasoning fields
- Structured output format with metadata tracking

**Existing Results (results/ and output/ folders):**
- ~70 dataset research results in JSON format
- Can be used as training data examples for fine-tuning proposal
- Demonstrates current capabilities and output structure

### New Components Required

**White Paper Document:**
- No existing technical white paper template in the codebase
- Will create from scratch following standard technical documentation practices
- Must synthesize information from codebase, README, and requirements

**Architecture Diagrams:**
- Text-based diagrams for Markdown compatibility
- Current workflow vs. proposed improvements visualization

**Improvement Proposals:**
- Three additional accuracy improvements beyond fine-tuning and deterministic search
- Complexity assessments for each proposal
- Technical feasibility analysis without implementation details

## Technical Approach

**Document Creation:**
- Markdown file created in project root or docs/ folder for easy stakeholder access
- Sections structured with clear headings (H2 for major sections, H3 for subsections)
- Code blocks and formatting for technical concepts

**Content Sourcing:**
- Current capabilities: extracted from README.md and domain models
- Sample outputs: reference 1-2 anonymized examples from results/ folder
- Metrics: count of JSON files in results/ folder (~70 datasets processed)
- Architecture: describe current LangChain agent → tools workflow
- Limitations: based on requirements discussion about unreliable results requiring manual review

**Improvement Proposals:**

1. **Supervised Fine-Tuning with Parameter-Efficient Adapters**
   - Base model: gpt-oss-120b
   - Method: Supervised fine-tuning using LoRA (Low-Rank Adaptation) or similar parameter-efficient approach
   - Training data: Input prompts from agent execution paired with manually reviewed/corrected outputs from ~70-100 datasets
   - Creates additional adapter weight matrices that augment base model predictions
   - Complexity: Medium-High (requires training infrastructure, labeled data preparation, adapter training pipeline)

2. **Deterministic Web Search Pipeline**
   - Replace agent-directed tool calls with pre-executed search queries
   - Example: Run fixed query "ENTITY-NAME description" before agent invocation
   - Embed search results directly into prompt context
   - Benefits: reproducible training data, consistent inputs, easier debugging
   - Complexity: Medium (requires refactoring agent invocation pipeline, search result formatting)

3. **Additional Improvements (propose 3 specific ones in document):**
   - Structured Output Schema with JSON validation
   - Multi-stage Verification with consistency prompts
   - Entity Type-Specific Prompt Templates
   - Output Confidence Scoring
   - Ensemble Validation across multiple model calls

**SOW Task Mapping:**
- Task 1.1 (HuggingFace AI Models): Adapt agent to search for model names using HuggingFace API as authoritative source
- Task 1.2 (Synbio Foundries): Adapt agent to search for foundry mentions using consortium websites and NSF/Agile Biofoundry lists
- Demonstrate how deterministic search + fine-tuning enables domain-agnostic entity resolution

**Success Metrics:**
- Target: 80% of agent outputs judged as valid for publication catalog API searches
- Valid = produces relevant results when used as search terms in publication databases
- Implies high precision in entity name extraction and alias identification

## Out of Scope

- Actual implementation of proposed improvements
- Code changes or refactoring
- Infrastructure cost analysis or resource budgeting
- Detailed project timeline or sprint planning
- Performance benchmarking or quantitative validation
- Sensitive performance issues or failure case analysis
- Specific library versions or dependency management
- Database schema changes
- API endpoint modifications

## Success Criteria

- White paper document is comprehensive, covering all 6 required sections
- Architecture diagrams clearly illustrate current vs. proposed systems
- Sample outputs and metrics accurately represent ~70 datasets processed
- Five improvement proposals are technically sound with appropriate ML/AI terminology
- Complexity assessments provide actionable guidance on feasibility
- SOW Task 1.1 and 1.2 mapping demonstrates clear generalization path from datasets to AI models and synbio foundries
- Document tone is professional and appropriate for technical stakeholders
- All exclusions (costs, implementation details, timelines) are respected
- 80% valid response rate target is clearly articulated as success metric

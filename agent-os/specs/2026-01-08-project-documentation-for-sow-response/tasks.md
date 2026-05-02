# Task Breakdown: Technical White Paper on Dataset Research Agent

## Overview
Total Tasks: 4 task groups
Assigned roles: api-engineer (for technical content), testing-engineer (for validation)
Project Type: Technical documentation (white paper creation)

## Task List

### Content Research & Data Collection

#### Task Group 1: Gather System Information and Metrics
**Assigned implementer:** api-engineer
**Dependencies:** None

- [x] 1.0 Complete system analysis and data collection
  - [x] 1.1 Analyze current agent architecture
    - Review `src/dataset_agent/adapters/agent.py` for LLM provider implementation
    - Review `src/dataset_agent/adapters/tools.py` for web_search and make_request patterns
    - Review `src/dataset_agent/domain/models.py` for DatasetInfo structure
    - Document current agent workflow: Prompt → Agent → Tool Calls → Response
  - [x] 1.2 Collect performance metrics from existing results
    - Count total datasets processed in `results/` folder (~70 files)
    - Select 2-3 representative sample outputs for inclusion in white paper
    - Document current output structure (aliases, organizations, access_type, URLs)
    - Note metadata fields (timing, status, completion) from sample files
  - [x] 1.3 Document current limitations
    - Identify unreliability patterns requiring manual review
    - Note dynamic web search variability (agent-chosen queries)
    - Document training data opportunity (~70-100 datasets with reviewed outputs)
  - [x] 1.4 Research fine-tuning technical details
    - Document gpt-oss-120b model characteristics
    - Research LoRA (Low-Rank Adaptation) parameter-efficient fine-tuning approach
    - Outline supervised fine-tuning with adapter weights concept
    - Define training data format: prompt-response pairs

**Acceptance Criteria:**
- Current system architecture documented with clear workflow
- 2-3 sample dataset outputs identified for white paper
- Performance metrics collected (~70 datasets processed)
- Fine-tuning approach technically accurate (LoRA/adapter weights terminology)

### White Paper Structure & Core Sections

#### Task Group 2: Write White Paper Core Content
**Assigned implementer:** api-engineer
**Dependencies:** Task Group 1

- [x] 2.0 Complete white paper document creation
  - [x] 2.1 Create executive summary
    - Introduce Dataset Research Agent purpose
    - Summarize current capabilities (dataset entity resolution)
    - Preview extensibility to AI models and synbio foundries
    - State goal: 80% valid response rate for publication searches
  - [x] 2.2 Write Section 1 - Current System Overview
    - Create text-based architecture diagram (ASCII art or Mermaid)
    - Describe agent workflow: LLM + LangChain + tools (web_search, make_request)
    - Explain current capabilities: aliases, organizations, descriptions, URLs
    - Include 2-3 sample outputs (anonymized from results/)
    - Present metrics: ~70 datasets processed, JSON output format
  - [x] 2.3 Write Section 2 - Current Limitations
    - Explain unreliable results requiring manual review
    - Describe dynamic web search variability (unpredictable agent queries)
    - Note lack of consistent training data due to non-deterministic searches
    - State need for improvements to reach 80% valid response rate
  - [x] 2.4 Write Section 3 - Improvement #1: Supervised Fine-Tuning
    - Base model: gpt-oss-120b
    - Method: Parameter-efficient fine-tuning with LoRA adapters
    - Training data: Prompt-response pairs from ~70-100 datasets with manual corrections
    - Adapter weights augment base model without full retraining
    - Complexity assessment: Medium-High (requires training infrastructure, labeled data prep, adapter training pipeline)
    - Expected impact: Improved entity extraction accuracy, reduced manual review
  - [x] 2.5 Write Section 4 - Improvement #2: Deterministic Web Search Pipeline
    - Current: Agent dynamically chooses web_search queries
    - Proposed: Pre-execute fixed queries (e.g., "ENTITY-NAME description")
    - Embed search results in prompt context before agent invocation
    - Benefits: Reproducible inputs, consistent training data, easier debugging
    - Complexity assessment: Medium (requires pipeline refactoring, search result formatting)
    - Expected impact: Predictable agent inputs, enabling effective fine-tuning

**Acceptance Criteria:**
- Executive summary clearly states purpose and goals
- Section 1 includes architecture diagram and sample outputs
- Section 2 articulates current limitations clearly
- Sections 3-4 use correct ML terminology (LoRA, adapter weights, supervised learning)
- Complexity assessments provided for both improvements

### Additional Improvements & SOW Mapping

#### Task Group 3: Propose Additional Improvements and SOW Application
**Assigned implementer:** api-engineer
**Dependencies:** Task Group 2

- [x] 3.0 Complete additional improvements and SOW mapping
  - [x] 3.1 Write Section 5 - Three Additional Accuracy/Reliability Improvements
    - Improvement #3: Structured Output Schema with JSON Validation
      - Enforce strict JSON schema for agent outputs (aliases, organizations fields)
      - Validate outputs against schema before acceptance
      - Complexity: Low-Medium (schema definition, validation logic)
      - Expected impact: Reduce malformed outputs, ensure consistent structure
    - Improvement #4: Multi-Stage Verification with Consistency Prompts
      - Run agent twice: first for extraction, second for verification
      - Ask agent to verify extracted entities against original context
      - Flag inconsistencies for manual review
      - Complexity: Medium (requires dual-pass pipeline, consistency logic)
      - Expected impact: Catch hallucinations, improve precision
    - Improvement #5: Entity Type-Specific Prompt Templates
      - Create optimized prompts for different entity types (datasets, AI models, foundries)
      - Include domain-specific instructions and examples
      - Template variables for entity name, authoritative sources
      - Complexity: Low (prompt engineering, template system)
      - Expected impact: Better extraction for specialized entity types
  - [x] 3.2 Write Section 6 - Application to SOW Tasks
    - Subsection 6.1: Task 1.1 - HuggingFace AI Model Mentions
      - Adapt deterministic search to query HuggingFace API for model metadata
      - Use model names (e.g., "sentence-transformers/all-MiniLM-L6-v2")
      - Search Dimensions API for publications mentioning these models
      - Apply fine-tuned agent to extract model aliases and variations
      - Demonstrate generalization from datasets to AI models
    - Subsection 6.2: Task 1.2 - Synbio Foundry Mentions
      - Adapt deterministic search to scrape NSF/Agile Biofoundry consortium lists
      - Extract foundry names, acronyms from authoritative sources
      - Search BioRxiv and specialized journals for foundry mentions
      - Apply fine-tuned agent to identify foundry-associated authors
      - Demonstrate generalization from datasets to synbio entities
    - Subsection 6.3: Generalization Path
      - Show how improvements enable domain-agnostic entity resolution
      - Entity-specific configurations (authoritative sources, search queries)
      - Common pipeline: deterministic search → fine-tuned agent → validated output
  - [x] 3.3 Write conclusion
    - Summarize five proposed improvements
    - Restate technical feasibility focus (no costs, timelines, implementation)
    - Emphasize 80% valid response rate as success metric
    - Highlight extensibility from datasets to other named entities

**Acceptance Criteria:**
- Three additional improvements are technically sound and accuracy-focused
- Each improvement has complexity assessment (Low, Medium, High)
- SOW Tasks 1.1 and 1.2 have dedicated subsections with clear mapping
- Generalization path demonstrates domain-agnostic approach
- Conclusion summarizes all five improvements and success metric

### Document Review & Validation

#### Task Group 4: Review and Finalize White Paper
**Assigned implementer:** testing-engineer
**Dependencies:** Task Groups 1-3

- [x] 4.0 Review white paper quality and technical accuracy
  - [x] 4.1 Validate technical content accuracy
    - Verify ML/AI terminology correctness (LoRA, adapter weights, supervised fine-tuning)
    - Check that gpt-oss-120b model reference is appropriate
    - Ensure architecture diagram matches actual codebase structure
    - Confirm sample outputs are representative and anonymized
  - [x] 4.2 Check document structure and completeness
    - Verify all 6 required sections are present (Executive Summary, Sections 1-6, Conclusion)
    - Ensure architecture diagrams are clear and informative
    - Confirm 2-3 sample outputs are included
    - Verify ~70 datasets metric is stated
  - [x] 4.3 Validate exclusions compliance
    - Ensure no infrastructure costs mentioned
    - Confirm no specific code implementations included
    - Verify no sensitive performance issues discussed
    - Check that no timelines or project management details present
  - [x] 4.4 Review Markdown formatting
    - Check proper heading hierarchy (H1 for title, H2 for sections, H3 for subsections)
    - Verify code blocks are formatted correctly
    - Ensure lists and bullet points are consistent
    - Confirm document is readable and professional
  - [x] 4.5 Validate success criteria alignment
    - Confirm 80% valid response rate target is clearly stated
    - Verify all five improvements are documented with complexity assessments
    - Check SOW Tasks 1.1 and 1.2 mapping is comprehensive
    - Ensure technical feasibility focus is maintained throughout

**Acceptance Criteria:**
- Technical terminology is accurate and appropriate for AI/ML stakeholders
- All required sections present and complete
- Document follows professional white paper structure
- Exclusions (costs, code, timelines) are respected
- Markdown formatting is clean and consistent
- Success metrics and SOW mapping are clear

## Execution Order

Recommended implementation sequence:
1. Content Research & Data Collection (Task Group 1)
2. White Paper Structure & Core Sections (Task Group 2)
3. Additional Improvements & SOW Mapping (Task Group 3)
4. Document Review & Validation (Task Group 4)

## Notes

This is a documentation project, not a traditional software development feature. The implementer assignments are adapted as follows:
- **api-engineer**: Assigned to content creation tasks as they handle technical documentation and understand the system architecture
- **testing-engineer**: Assigned to review/validation tasks to ensure quality and accuracy

No database migrations, API endpoints, or UI components are required for this specification.

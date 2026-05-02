# Spec Requirements: Project Documentation for SOW Response

## Initial Description

I want to create a document so I explain to the people involved in this conversation I already have an agent, developed in the context of the current project, to find aliases and organizations of datasets. That I also think this agent can be extended to find search terms of entities in publicatons full texts like for synbio foundries. I want to explain the current agent still have some problems, the results are still not realiable and it require manual review. SO the agent can be improved. One of the thnigs I think we could do to improve it is fining tunning it. Since we already have the aliases and organizations for almost a hundred datasets and we could use to train a model. For that I think we would need to do a few changes to the agent process. For example, today we give the agent the web search tool to search for things the agent thinks will help to produce the correct response and wget tool to visit any website brought from the web search to get more details that could help the agent to get the correct response as well. Maybe instead of letting the agent to the web search it thinks best suites its needs, we should make the searches and web page vistis statics so it gets easier to train the model because we will now what web searches will be executed on each execution. Give a brief overview of the complexity of this improvemtns, also list 3 other improvements you think we can do to improve the reliability of the responses of the agent.

## Requirements Discussion

### First Round Questions

**Q1:** Document Format & Audience - I assume this should be a technical overview document (like a white paper or technical memo) aimed at stakeholders who understand AI/ML concepts. Should it be a formal PDF-style document, a Markdown technical brief, or something else?
**Answer:** A white paper with a technical overview is what I want. Markdown format is good. The stakeholders understand AI/ML-savvy.

**Q2:** Document Scope - Current Capabilities - I'm thinking the document should include: (a) a high-level system architecture diagram showing how the agent works, (b) sample outputs from existing dataset research (anonymized if needed), and (c) metrics on the ~100 datasets processed. Should we include all three, or focus only on conceptual descriptions?
**Answer:** Include all three.

**Q3:** Fine-tuning Training Data Details - You mentioned using the ~100 datasets' aliases and organizations as training data. Should the document explain a specific fine-tuning approach (e.g., LoRA, full fine-tuning, prompt tuning) or keep it conceptual? Do you have preference for the base model to fine-tune (current LLMs you're using: qwen3:32b on Ollama)?
**Answer:** I would like to consider the model gpt-oss-120b as base model. I'd like to do the fine tunning where I send previous prompts and reviewed responsed and it will create extra weights to be considered by the model. use the correct terms and concepts to explain what i want.

**Q4:** Static vs Dynamic Web Search - Implementation Complexity - For the static search approach, I assume you'd want to: (a) define a fixed set of authoritative sources per entity type (e.g., HuggingFace API for AI models, specific consortium websites for synbio foundries), (b) pre-fetch and cache these pages, and (c) have the agent work only with this fixed corpus. Is that the right interpretation, or do you envision something different?
**Answer:** For example, currently, the agent needs to come up with the description of the dataset being processed. The agent has a web search tool and I tell the agent it should use the web search tool how many times it wants to find the description. Then the agent do web searchs with different queries like "What does the DATASET-X do?", "DATASET-X description", and etc. Instead of giving the web search tool to the agent, I want to run a deterministic web search with a query like "DATASET-X description", and then add the result to the prompt being passed to the agent instead of letting it call the web search tool with the query it comes up with.

**Q5:** Success Metrics - Should the document include proposed success metrics for the improvements (e.g., "reduce manual review time by X%", "achieve Y% accuracy on validation set")? Or keep it more qualitative?
**Answer:** We should aim the agent to return 80% of response we judge as valid, that will return good results when used in the Publication catalog api to find mentions to something.

**Q6:** Timeline & Resource Requirements - Should the document address estimated timeline and resources needed for the proposed improvements (fine-tuning, static search implementation, etc.), or focus purely on technical feasibility?
**Answer:** Focus on technical feasibility.

**Q7:** Integration with SOW Tasks 1.1 & 1.2 - Should the document explicitly map how the improved agent would address the HuggingFace AI models search (Task 1.1) and the synbio foundries search (Task 1.2), showing the generalization from datasets to other named entities?
**Answer:** Add a separation section showing how the improved agent can help with tasks 1.1 and 1.2.

**Q8:** Other Improvements - Focus Areas - For the 3 additional improvements beyond fine-tuning and static searches, should I focus on: (a) accuracy/reliability improvements, (b) scalability improvements, (c) usability improvements, or (d) a mix of all three?
**Answer:** Focus on accuracy/reliability improvements.

**Q9:** What should be excluded from this document? - Are there specific technical details you want to leave out (e.g., infrastructure costs, specific code implementations, sensitive performance issues)?
**Answer:** Leave out infrastructure costs, specific code implementations, and sensitive performance issues.

### Existing Code to Reference

No similar existing features identified for reference.

### Follow-up Questions

None needed - requirements are clear.

## Visual Assets

### Files Provided:
No visual assets provided.

### Visual Insights:
N/A - No visuals were provided.

## Requirements Summary

### Functional Requirements

**Primary Document Goal:**
- Create a technical white paper (Markdown format) explaining the current Dataset Research Agent capabilities
- Demonstrate how the agent can be extended from dataset entity resolution to other named entities (AI models, synbio foundries)
- Propose improvements to increase reliability and reduce manual review requirements
- Map the improved agent to SOW Tasks 1.1 (HuggingFace AI models) and 1.2 (synbio foundries)

**Content Sections Required:**

1. **Current System Overview**
   - High-level system architecture diagram showing agent workflow
   - Description of how the agent finds aliases and organizations for datasets
   - Sample outputs from existing ~100 datasets processed (anonymized if needed)
   - Metrics and statistics on the dataset corpus
   - Current limitations: unreliable results requiring manual review

2. **Proposed Improvement #1: Supervised Fine-Tuning**
   - Use gpt-oss-120b as base model
   - Supervised fine-tuning approach (parameter-efficient fine-tuning with adapter weights like LoRA)
   - Training data: prompt-response pairs from ~100 datasets with manually reviewed/corrected outputs
   - Complexity assessment for this approach
   - Expected impact on reliability

3. **Proposed Improvement #2: Deterministic Web Search Pipeline**
   - Replace agent-directed dynamic web searches with pre-determined search queries
   - Example: Instead of agent choosing queries, run fixed query like "DATASET-X description"
   - Embed search results directly into prompt context rather than tool calls
   - Benefits: reproducible training data, consistent agent inputs
   - Complexity assessment for this approach

4. **Three Additional Accuracy/Reliability Improvements**
   - Document needs to propose 3 additional improvements focused on accuracy/reliability
   - Each should include brief overview of complexity
   - All improvements should aim toward 80% valid response rate

5. **Application to SOW Tasks (Dedicated Section)**
   - Show how improved agent addresses Task 1.1: HuggingFace AI model mentions
   - Show how improved agent addresses Task 1.2: Synbio foundry mentions
   - Demonstrate generalization from dataset entity resolution to other named entities

**Success Metrics:**
- Target: 80% of agent responses judged as valid for use in Publication catalog API searches
- Valid = produces good results when searching for entity mentions in publications

**Audience Considerations:**
- Technical stakeholders with AI/ML knowledge
- Understands concepts like fine-tuning, LLM architectures, prompt engineering
- Interested in technical feasibility, not implementation details

### Non-Functional Requirements

**Document Quality:**
- Professional technical white paper format
- Clear, well-structured Markdown
- Technical accuracy in ML/AI terminology
- Appropriate for stakeholder communication

**Exclusions:**
- Infrastructure costs and resource estimates
- Specific code implementations
- Sensitive performance issues
- Timeline and project management details

### Reusability Opportunities

None identified - this is a net-new documentation piece with no existing templates or similar documents in the codebase.

### Scope Boundaries

**In Scope:**
- Technical white paper document creation
- Architecture diagrams showing current and proposed systems
- Analysis of improvement complexity
- Mapping to SOW tasks 1.1 and 1.2
- Three additional reliability improvement proposals
- Sample outputs and metrics from existing work

**Out of Scope:**
- Actual implementation of improvements
- Code changes to the agent
- Cost analysis or budgeting
- Detailed project timeline
- Infrastructure architecture details
- Performance benchmarking reports

### Technical Considerations

**Fine-Tuning Approach (Improvement #1):**
- Base model: gpt-oss-120b
- Method: Supervised fine-tuning with parameter-efficient adapters (e.g., LoRA - Low-Rank Adaptation)
- Training data format: Input prompts paired with manually reviewed/corrected outputs
- Creates additional weight matrices (adapters) that augment the base model without full retraining
- Correct terminology: "adapter weights," "parameter-efficient fine-tuning," "supervised learning"

**Deterministic Search Pipeline (Improvement #2):**
- Current: Agent has tool access, generates dynamic search queries based on reasoning
- Proposed: Pre-execution of fixed search queries, results embedded in prompt context
- Example transformation:
  - Before: Agent calls `web_search("What does DATASET-X do?")`, `web_search("DATASET-X description")`
  - After: System executes `web_search("DATASET-X description")` before agent invocation, adds results to prompt
- Benefits: reproducible inputs, consistent training data, easier to debug
- Trade-off: Less flexible but more predictable

**Additional Improvements (3 Required):**
- Must focus on accuracy and reliability
- Should complement fine-tuning and deterministic search
- Examples to consider (document should propose specific ones):
  - Structured output schemas with validation
  - Multi-stage verification prompts
  - Ensemble approaches with multiple model calls
  - Enhanced context retrieval strategies
  - Domain-specific prompt templates
  - Output consistency checks

**Generalization to New Entity Types:**
- Current: Optimized for dataset entity resolution
- Target: Extend to AI model names (Task 1.1) and synbio foundries (Task 1.2)
- Key challenge: Different entity characteristics require adapted search strategies
- Solution approach: Domain-agnostic architecture with entity-specific configurations

**Integration with Publication Catalog API:**
- Agent outputs used as search terms in publication catalogs
- Quality gate: 80% of outputs should produce relevant publication matches
- Implies need for precision over recall in entity extraction

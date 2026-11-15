# Product Mission

> **Documentation:** Detailed docs live in [`docs/`](../../docs/). See [`docs/agent-os-overview.md`](../../docs/agent-os-overview.md) for an overview of this `agent-os/` folder.

## Pitch

Dataset Research Agent is a Python-based API and CLI that helps data teams automatically research datasets (description, aliases, organizations, access type, key URLs) by orchestrating LLMs with web search and persisting results for cataloging workflows.

## Users

### Primary Customers

- Data Catalog/Metadata Teams: Automate metadata enrichment for internal/external datasets
- Research Analysts and Librarians: Accelerate literature/data discovery with consistent outputs

### User Personas

**Data Catalog Specialist** (28-50 years old)
- **Role:** Metadata engineer / data steward
- **Context:** Curating enterprise data catalogs across many sources
- **Pain Points:** Manual, inconsistent metadata; slow research turnaround
- **Goals:** Consistent, auditable dataset descriptions and aliases at scale

**Research Analyst** (24-45 years old)
- **Role:** Analyst in public policy/academia
- **Context:** Needs quick understanding of dataset scope, access, and provenance
- **Pain Points:** Time-consuming search; fragmented docs and access rules
- **Goals:** Rapid, trustworthy summaries with links to data, schema, and docs

## The Problem

### Manual dataset research slows delivery

Teams spend hours per dataset gathering descriptions, aliases, organizations, and access details. This delays analytics and catalog freshness and leads to inconsistent outputs.

**Our Solution:** Automate dataset research with an LLM + web-search agent producing standardized JSON results.

### Inconsistent naming and citations

Aliases and organization identifiers vary widely across sources, creating duplicates and poor searchability in catalogs.

**Our Solution:** Extract, normalize, and de-duplicate aliases and organizations with deterministic utilities.

### Access and documentation fragmentation

Data, schema, and documentation links are scattered, often outdated, and hard to validate.

**Our Solution:** Locate and validate best-available URLs and persist results alongside task status in SQLite.

## Differentiators

### Clean architecture with swappable providers

Unlike ad-hoc scripts, this project separates domain, adapters, and utilities, enabling easy swaps between Ollama and OpenRouter and between search providers (DuckDuckGo, Tavily).

### API-first with async tasking

Compared to CLI-only tools, this ships a FastAPI service with background jobs, pagination, and simple API-key auth for integration into pipelines.

### Traceable persistence

Results are stored as JSON files and tracked in SQLite with task lifecycle, aiding auditability and reprocessing.

## Key Features

### Core Features

- **Dataset research orchestration:** LLM + web search prompts tailored for description, organizations, aliases, access type, and URLs
- **Official name detection:** Automatically determines if a dataset name is the official catalog name or a subset/component of a larger dataset, with reasoning
- **Relationship classification:** Identifies relationship types (official_name, subset_of, table_within, component_of) between provided names and official dataset names
- **CSV batch processing:** Process multiple datasets from CSV files with comprehensive logging and error handling
- **URL validation utility:** Optional HTTP validation of candidate links
- **Alias normalization:** Deduplication and substring-aware filtering suited for catalog search
- **Result persistence:** JSON file output with enhanced metadata (official_name, relationship_type, reasoning) and SQLite task/result tracking
- **API endpoints:** Submit task, check status, fetch result, list tasks, health

### Collaboration Features

- **Containerized runtime:** Docker and Compose with Ollama service
- **Pluggable providers:** Ollama, OpenRouter, or LMStudio for LLM; DuckDuckGo or Tavily for web search
- **Logging:** Rotating logs and structured access/error logs


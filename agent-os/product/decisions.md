# Decisions

## LLM Provider
- Default to Ollama (local, containerized) for cost/control; alternative OpenRouter supported via API key; LMStudio supported for local inference with GUI-based model management and OpenAI-compatible API.

## Search Provider
- Default DuckDuckGo for zero-config; optional Tavily when API key available for higher-quality results.

## Persistence Strategy
- Operational state in SQLite with WAL and indices; artifacts saved as JSON files for portability and easy integration.

## API Design
- FastAPI chosen for async, ecosystem, and OpenAPI docs. Background tasks used for non-blocking research.

## Authentication
- Simple API key via header `X-API-Key`. Development mode tolerates missing keys; production should require explicit keys.

## Containerization
- Docker-based distribution with a compose stack including `ollama` service and optional GPU.

## Logging
- Rotating file handlers for general, error, and uvicorn access logs stored under `./logs`.

## Tooling Abstractions
- LangChain agent with swappable providers and tools to minimize vendor lock-in and ease experimentation.

# Roadmap

## Phase 0: Already Completed

- [x] FastAPI service with endpoints for task submission, status, result, listing, and health
- [x] Background task processing for research execution
- [x] SQLite persistence for tasks and results with indices and WAL
- [x] JSON repository for result artifacts
- [x] LLM agent using LangChain with three providers: Ollama (default), OpenRouter, and LMStudio
- [x] LMStudio provider integration with OpenAI-compatible API and health checks
- [x] Web search tools (DuckDuckGo default, Tavily optional) and HTTP validation tool
- [x] Dockerfile and docker-compose with Ollama service
- [x] Rotating logs for general, error, and access
- [x] CSV batch processing script for multiple datasets
- [x] Official dataset name detection and relationship classification (official_name, subset_of, table_within, component_of)
- [x] Enhanced DatasetInfo model with official_name, relationship_type, and reasoning fields

## Phase 1: Current Development

- [ ] Improve error handling and retries in agent/tool stack
- [ ] Add unit tests for API routes and database layer; basic CI

## Phase 2: Near-Term Enhancements

- [ ] Observability (structured logs, metrics endpoints)
- [ ] Expand URL discovery logic (mirror/archival links)



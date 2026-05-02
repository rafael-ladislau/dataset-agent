# Technical Stack

- application_framework: FastAPI (>=0.109) on Python 3.10.15
- database_system: SQLite (sqlite3) with WAL and indices
- javascript_framework: n/a
- import_strategy: n/a
- css_framework: n/a
- ui_component_library: n/a
- fonts_provider: n/a
- icon_library: n/a
- application_hosting: Docker + docker-compose; Uvicorn; API exposed on 8888 (container)
- database_hosting: SQLite file at `./data/research.db` (mounted to `/app/data` in compose)
- asset_hosting: n/a

## LLM & Tools
- Providers:
  - Ollama (default) via `langchain_ollama.ChatOllama` (compose service `ollama`, default model `qwen3:32b`)
  - OpenRouter via `langchain_openai.ChatOpenAI` (optional; requires `OPENROUTER_API_KEY`)
  - LMStudio via `langchain_openai.ChatOpenAI` with OpenAI-compatible API (local inference on port 1234; GUI-based model management)
- Tools:
  - Web Search: `duckduckgo-search` (default) or `tavily-python`
  - HTTP Validation: `requests` via `make_request` tool

## API & Runtime
- Endpoints: `/api/research` (POST, GET list), `/api/research/{task_id}`, `/api/research/{task_id}/result`, `/api/health`
- Auth: API key `X-API-Key`; keys from `API_KEYS` env; dev mode allows anonymous when unset
- Background: FastAPI `BackgroundTasks`
- Logging: Python logging with `RotatingFileHandler` (general, error, uvicorn access) under `./logs`

## Containers & Services
- Base image: `python:3.10.15-slim-bullseye`
- Services: `api` (this app) and `ollama` (LLM backend)
- GPU: NVIDIA device reservations for `ollama` if available

## Configuration (env)
- `API_KEYS`
- `SQLITE_DB_PATH` (default `./data/research.db`)
- `LOG_PATH` (default `./logs`)
- `LOG_LEVEL` (default `INFO`)
- `DATASET_AGENT_LLM_PROVIDER` (`ollama`, `openrouter`, or `lmstudio`)
- `DATASET_AGENT_LLM_MODEL` (model name for the selected provider)
- `DATASET_AGENT_TEMPERATURE` (sampling temperature, default `0.85`)
- `OLLAMA_HOST` (default `http://localhost:11434`)
- `OPENROUTER_API_KEY` (optional)
- `OPENROUTER_BASE_URL` (optional; default `https://openrouter.ai/api/v1`)
- `LMSTUDIO_BASE_URL` (default `http://localhost:1234/v1`)
- `LMSTUDIO_API_KEY` (placeholder, default `lm-studio`)
- `TAVILY_API_KEY` (optional)
- `WEB_SEARCH_PROVIDER` (`duckduckgo` or `tavily`)

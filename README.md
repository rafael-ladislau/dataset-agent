# dataset-agent

LLM-powered agent that researches dataset metadata (names, aliases, organizations, URLs, access type) and outputs structured JSON. Supports a **CLI** and an **async HTTP API**.

See `DOCUMENTACAO_PROJETO.md` for a full technical reference and `docs/white-paper-dataset-research-agent.md` for the project white paper.

## Requirements

- Python 3.11+
- An LLM provider — one of:
  - **Ollama** (default, local): [ollama.com](https://ollama.com)
  - **OpenRouter** (cloud): requires `DATASET_AGENT_OPENROUTER_API_KEY`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Optional: enable Dimensions literature gate
pip install -e ".[dev,dimensions]"
```

## Environment Setup

Copy the example file and fill in the values:

```bash
cp .env.example .env
```

### Environment Variables

All variables use the `DATASET_AGENT_` prefix.

#### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_AGENT_LLM_PROVIDER` | `ollama` | `ollama` or `openrouter` |
| `DATASET_AGENT_OLLAMA_MODEL` | `llama3` | Model name for Ollama |
| `DATASET_AGENT_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `DATASET_AGENT_OPENROUTER_API_KEY` | _(empty)_ | **Sensitive** — get at [openrouter.ai/keys](https://openrouter.ai/keys) |
| `DATASET_AGENT_OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Model slug for OpenRouter |

#### Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_AGENT_AGENT_MAX_ITERATIONS` | `5` | Max LLM tool-call iterations per step |
| `DATASET_AGENT_AGENT_TIMEOUT_SECONDS` | `120` | Per-step timeout |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

#### Persistence

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_AGENT_OUTPUT_DIR` | `./output` | Directory for generated JSON files |
| `DATASET_AGENT_TASKS_DB` | `./tasks.db` | SQLite file for async task state |

#### Literature Gate _(optional)_

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_AGENT_LITERATURE_GATE` | `noop` | `noop` (disabled) or `dimensions` |
| `DATASET_AGENT_LITERATURE_MAX_ATTEMPTS` | `3` | Retry attempts for gate validation |
| `DATASET_AGENT_LITERATURE_THRESHOLD` | `0.5` | Minimum score to pass validation |
| `DATASET_AGENT_DIMENSIONS_API_KEY` | _(empty)_ | **Sensitive** — get at [app.dimensions.ai](https://app.dimensions.ai) → Account → API |

## CLI

```bash
dataset-research run "NASS Census of Agriculture"

# with webhook notification on completion
dataset-research run "NASS Census of Agriculture" --webhook https://example.com/hook
```

Output JSON is printed to stdout; the saved file path is printed to stderr.

## API

Start the server:

```bash
dataset-research-api
# or: uvicorn dataset_agent.interfaces.api:app --reload
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tasks` | Create a research task (async) |
| `GET` | `/tasks/{id}` | Check task status |
| `GET` | `/tasks/{id}/result` | Retrieve completed `DatasetRecord` |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/validate` | Run literature gate on existing terms |
| `GET` | `/health` | Health check (tests Ollama connectivity if applicable) |

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"dataset_name":"Current Population Survey"}'
```

## Docker / Compose

```bash
docker compose up
```

The API is exposed on port `8000`. Data is persisted in `./data` (output JSON + SQLite). Set environment variables in a `.env` file at the project root before starting.

## Tests

```bash
pytest
```

Reference outputs from previous agent runs are stored in `results/` (70 datasets) and `workforce/` (11 datasets). These can be used to validate output format and compare quality against the refactored agent.

## Documentation

| File | Contents |
|------|----------|
| `DOCUMENTACAO_PROJETO.md` | Full technical reference (architecture, pipeline, adapters) |
| `docs/white-paper-dataset-research-agent.md` | Capabilities, limitations, and proposed improvements |
| `docs/lmstudio-integration-guide.md` | LMStudio integration guide (reference for future support) |
| `docs/lmstudio-provider-spec.md` | LMStudio provider specification |
| `docs/aliases-agent-summary.md` | Summary of alias extraction behavior |

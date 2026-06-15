# dataset-agent

LLM-powered agent that researches dataset metadata (names, aliases, organizations, URLs, access type) and outputs structured JSON. Supports a **CLI** and an **async HTTP API**.

See `DOCUMENTACAO_PROJETO.md` for a full technical reference and `docs/white-paper-dataset-research-agent.md` for the project white paper.

## Requirements

- Python 3.11+
- An LLM provider — one of:
  - **LM Studio** (default, local): [lmstudio.ai](https://lmstudio.ai)
  - **Ollama** (local): [ollama.com](https://ollama.com)
  - **OpenRouter** (cloud): requires `OPENROUTER_API_KEY`

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
| `DATASET_AGENT_LLM_PROVIDER` | `lmstudio` | `lmstudio`, `ollama`, or `openrouter` |
| `LMSTUDIO_BASE_URL` | `http://127.0.0.1:1234` | LM Studio server URL |
| `DATASET_AGENT_LMSTUDIO_MODEL` | `gemma-4-31b-it-mlx` | Model name as shown in LM Studio |
| `DATASET_AGENT_OLLAMA_MODEL` | `gemma4-31b` | Model name for Ollama |
| `DATASET_AGENT_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `OPENROUTER_API_KEY` | _(empty)_ | **Sensitive** — get at [openrouter.ai/keys](https://openrouter.ai/keys) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter Anthropic-compatible endpoint |
| `DATASET_AGENT_OPENROUTER_MODEL` | `anthropic/claude-3.5-sonnet` | Model slug for OpenRouter |

##### Using LM Studio

1. Download and install [LM Studio](https://lmstudio.ai).
2. Download a model (e.g., `gemma-4-31b-it-mlx`).
3. Load the model (click the **↔** icon).
4. Start the local server: **Developer → Local Server → Start Server** (default port `1234`).
5. Verify it is running: `curl http://localhost:1234/v1/models`.
6. Point the agent at it via `.env`:
   ```bash
   DATASET_AGENT_LLM_PROVIDER=lmstudio
   DATASET_AGENT_LMSTUDIO_MODEL=gemma-4-31b-it-mlx
   LMSTUDIO_BASE_URL=http://localhost:1234
   ```

##### Using OpenRouter

```bash
DATASET_AGENT_LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_api_key_here
DATASET_AGENT_OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

#### Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_AGENT_AGENT_MAX_ITERATIONS` | `5` | Max LLM tool-call iterations per step |
| `DATASET_AGENT_AGENT_TIMEOUT_SECONDS` | `120` | Per-step timeout |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DATASET_AGENT_LOG_DIR` | _(empty)_ | When set, writes rotating logs (`dataset_agent.log`, `dataset_agent_errors.log`, 10 MB × 5) |

#### API Security _(optional)_

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEYS` | _(empty)_ | Comma-separated keys; when set, write endpoints require an `x-api-key` header. Empty disables auth. |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |

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
| `POST` | `/optimize` | Optimize a Dimensions publications query (sync) |
| `POST` | `/optimize/tasks` | Optimize query (async) |
| `GET` | `/optimize/tasks/{id}/result` | Retrieve optimize result |
| `GET` | `/health` | Health check (tests Ollama connectivity if applicable) |

When `API_KEYS` is configured, write endpoints (`POST /tasks`, `/validate`, `/optimize`, `/optimize/tasks`) require an `x-api-key` header:

```bash
curl -s -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_key" \
  -d '{"dataset_name":"Current Population Survey"}'
```

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

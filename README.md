# dataset-agent

Pesquisa de metadados sobre datasets (LLM + ferramentas), com **CLI** e **API HTTP**. Ver `LOGICA_DO_PROJETO.md` para o fluxo conceptual.

## Requisitos

- Python 3.11+
- [Ollama](https://ollama.com) local (por defeito) ou OpenRouter (`DATASET_AGENT_LLM_PROVIDER=openrouter`)

## Configuração

```bash
cp .env.example .env
# editar .env
```

Variáveis principais:

| Variável | Descrição |
|----------|-----------|
| `DATASET_AGENT_LLM_PROVIDER` | `ollama` ou `openrouter` |
| `DATASET_AGENT_OLLAMA_MODEL` | Nome do modelo Ollama |
| `DATASET_AGENT_OPENROUTER_API_KEY` / `MODEL` | Se usar OpenRouter |
| `DATASET_AGENT_TASKS_DB` | Caminho SQLite para tarefas |
| `DATASET_AGENT_OUTPUT_DIR` | Pasta dos JSON gerados |
| `DATASET_AGENT_LITERATURE_GATE` | `noop` (por defeito) ou `dimensions` (requer extra `dimensions`) |

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# opcional: pip install -e ".[dev,dimensions]"
```

## CLI

```bash
dataset-research run "NASS Census of Agriculture"
# opcional: --webhook https://example.com/hook
```

## API

```bash
dataset-research-api
# ou: uvicorn dataset_agent.interfaces.api:app --reload
```

Criar tarefa:

```bash
curl -s -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"dataset_name":"Example Dataset"}'
```

Consultar estado e resultado: `GET /tasks/{id}`, `GET /tasks/{id}/result`.

## Testes

```bash
pytest
```

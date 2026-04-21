# API HTTP (dataset-research-api). Python 3.12 alinhado a requires-python >=3.11
FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATASET_AGENT_OUTPUT_DIR=/data/output \
    DATASET_AGENT_TASKS_DB=/data/tasks.db

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

# Dados persistentes (JSON + SQLite de tarefas)
VOLUME ["/data"]

EXPOSE 8000

CMD ["dataset-research-api"]

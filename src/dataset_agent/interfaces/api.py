"""FastAPI HTTP interface."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any
from urllib.parse import urljoin

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from dataset_agent.domain.models import (
    DatasetRecord,
    Group,
    LiteratureValidation,
    ResearchRequest,
    TermsEvaluation,
    ValidationRequest,
    YearsRange,
)
from dataset_agent.domain.ports import TaskStatus
from dataset_agent.adapters.tasks_sqlite import SqliteTaskRepository
from dataset_agent.logging_setup import configure_application_logging
from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)


def _merge_exclude_terms(body_terms: list[str], suggested: list[str]) -> list[str]:
    """Stable union with case-insensitive deduplication (first occurrence wins)."""
    seen: set[str] = set()
    out: list[str] = []
    for t in list(body_terms) + list(suggested):
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


@asynccontextmanager
async def _lifespan(app: FastAPI):
    configure_application_logging()
    logger.info("API starting (dataset_agent logging active)")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="dataset-agent",
    version="0.1.0",
    description="HTTP API: enqueue research by dataset name (`dataset_name` + optional `webhook_url`).",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    lifespan=_lifespan,
)


def build_use_case(settings: Settings):
    """Lazy wrapper so importing this module does not load LangChain."""
    from dataset_agent.bootstrap import build_use_case as _bootstrap_build

    return _bootstrap_build(settings)


def get_settings() -> Settings:
    return Settings()


class TaskCreateResponse(BaseModel):
    id: str
    status: str


def _run_background_task(task_id: str, settings: Settings) -> None:
    tasks = SqliteTaskRepository(settings.tasks_db)
    row = tasks.get_task(task_id)
    if not row:
        logger.error("Task %s not found in database", task_id)
        return
    logger.info("Task %s: status -> processing", task_id)
    tasks.update_task(task_id, status=TaskStatus.PROCESSING)
    try:
        payload = row["payload"]
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        request = ResearchRequest.model_validate(payload)
        logger.info(
            "Task %s: running pipeline dataset_name=%r webhook=%s",
            task_id,
            request.dataset_name,
            "yes" if request.webhook_url else "no",
        )
        use_case = build_use_case(settings)
        record, path = use_case.execute(request)
        logger.info("Task %s: pipeline finished; result at %s", task_id, path)
        tasks.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            result_path=str(path),
        )
        logger.info("Task %s: status -> completed", task_id)
        if record.webhook_url:
            logger.info("Task %s: posting to webhook", task_id)
            try:
                httpx.post(
                    record.webhook_url,
                    json=record.model_dump(mode="json"),
                    timeout=30.0,
                )
                logger.info("Task %s: webhook delivered successfully", task_id)
            except Exception as webhook_err:
                logger.warning("Task %s: webhook failed: %s", task_id, webhook_err)
    except Exception as e:
        logger.exception("Task %s: unexpected error", task_id)
        tasks.update_task(task_id, status=TaskStatus.FAILED, error=str(e))


@app.post("/tasks", response_model=TaskCreateResponse)
def create_task(
    body: ResearchRequest,
    background_tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings)],
) -> TaskCreateResponse:
    tasks = SqliteTaskRepository(settings.tasks_db)
    tid = tasks.create_task(body.model_dump(mode="json"))
    logger.info(
        "Task created id=%s dataset_name=%r (pending; background worker)",
        tid,
        body.dataset_name,
    )
    background_tasks.add_task(_run_background_task, tid, settings)
    return TaskCreateResponse(id=tid, status=TaskStatus.PENDING)


@app.get("/tasks/{task_id}")
def get_task(
    task_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    tasks = SqliteTaskRepository(settings.tasks_db)
    row = tasks.get_task(task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return row


@app.get("/tasks")
def list_tasks(
    settings: Annotated[Settings, Depends(get_settings)],
    limit: int = 50,
) -> list[dict[str, Any]]:
    tasks = SqliteTaskRepository(settings.tasks_db)
    return tasks.list_tasks(limit=limit)


@app.get("/tasks/{task_id}/result")
def get_task_result(
    task_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> DatasetRecord:
    tasks = SqliteTaskRepository(settings.tasks_db)
    row = tasks.get_task(task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    if row["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Task not completed: {row['status']}")
    path = row.get("result_path")
    if not path:
        raise HTTPException(status_code=500, detail="No result path")
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=500, detail="Result file missing")
    return DatasetRecord.model_validate_json(p.read_text(encoding="utf-8"))


def _probe_ollama(base_url: str, timeout_seconds: float = 3.0) -> dict[str, Any]:
    """GET /api/tags — verify the Ollama server responds."""
    base = base_url.rstrip("/") + "/"
    url = urljoin(base, "api/tags")
    started = time.perf_counter()
    try:
        r = httpx.get(url, timeout=timeout_seconds)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        if r.status_code != 200:
            return {
                "ok": False,
                "base_url": base_url,
                "error": f"HTTP {r.status_code}",
                "latency_ms": elapsed_ms,
            }
        data = r.json()
        models = data.get("models") if isinstance(data, dict) else None
        n = len(models) if isinstance(models, list) else None
        out: dict[str, Any] = {
            "ok": True,
            "base_url": base_url,
            "latency_ms": elapsed_ms,
        }
        if n is not None:
            out["models_loaded"] = n
        return out
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        err = str(e)
        out: dict[str, Any] = {
            "ok": False,
            "base_url": base_url,
            "error": err,
            "latency_ms": elapsed_ms,
            "probe_url": url,
        }
        if "127.0.0.1" in base_url or "localhost" in base_url.lower():
            out["hint"] = (
                "Inside a Docker container, 127.0.0.1/localhost refers to the container itself. "
                "Use DATASET_AGENT_OLLAMA_BASE_URL=http://host.docker.internal:11434 (or the host IP)."
            )
        elif "host.docker.internal" in base_url and "refused" in err.lower():
            out["hint"] = (
                "Ollama on the host may be bound only to 127.0.0.1. Start it with "
                "OLLAMA_HOST=0.0.0.0:11434 (or equivalent) so it accepts traffic from the Docker network."
            )
        return out


@app.post("/validate")
def validate_terms(
    body: ValidationRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> DatasetRecord:
    """
    Validate dataset terms against Dimensions publications.
    Returns same JSON format as /tasks endpoint result.
    """
    from dataset_agent.adapters.literature import literature_gate_from_settings, evaluate_terms_with_llm
    from dataset_agent.adapters.agent_langchain import LangChainAgent
    
    logger.info(
        "Validate request: main_dataset_name=%r dataset_names=%s flag_terms=%s "
        "exclude_terms=%s sample_size=%s llm_batch_size=%s",
        body.main_dataset_name,
        len(body.dataset_names),
        len(body.flag_terms),
        len(body.exclude_terms),
        body.sample_size,
        body.llm_batch_size,
    )
    
    # Build minimal DatasetRecord with provided terms
    record = DatasetRecord(
        engine=settings.default_engine,
        group=Group(name=settings.default_group_name),
        main_dataset_name=body.main_dataset_name,
        description=body.description or "",
        dataset_names=body.dataset_names,
        flag_terms=body.flag_terms,
        years_range=YearsRange(
            start_year=settings.default_years_start,
            end_year=settings.default_years_end,
        ),
        filter_us_affiliation=settings.default_filter_us_affiliation,
        publication_types=list(settings.default_publication_types),
        official_name=body.main_dataset_name,
        relationship_type="official_name",
        official_name_reasoning="Provided via /validate endpoint",
        exclude_terms=list(body.exclude_terms),
    )
    
    # Build agent for LLM validation
    agent = LangChainAgent(
        provider=settings.llm_provider,
        model_name=(
            settings.openrouter_model
            if settings.llm_provider == "openrouter"
            else settings.ollama_model
        ),
        api_key=settings.openrouter_api_key or None,
        base_url=(
            settings.openrouter_base_url
            if settings.llm_provider == "openrouter"
            else settings.ollama_base_url
        ),
        max_iterations=settings.agent_max_iterations,
        timeout_seconds=settings.agent_timeout_seconds,
    )
    
    # Run literature gate
    gate = literature_gate_from_settings(settings, agent=agent)
    result = gate.assess(
        record,
        sample_size=body.sample_size,
        llm_batch_size=body.llm_batch_size,
    )
    
    logger.info(
        "Validate result: passed=%s indicator=%.2f",
        result.passed,
        result.indicator,
    )
    
    # Save validation result
    record.literature_validation = LiteratureValidation(
        passed=result.passed,
        indicator=result.indicator,
        attempts=1,
        detail=result.detail,
    )
    record.publications_total = result.detail.get("publications_total")
    
    # Run terms evaluation
    validation_details = result.detail.get("validation_details", [])
    logger.info("Running terms evaluation...")
    eval_result = evaluate_terms_with_llm(
        dataset_name=record.main_dataset_name,
        description=record.description,
        dataset_names=record.dataset_names,
        flag_terms=record.flag_terms,
        validation_details=validation_details,
        agent=agent,
    )
    sug_names = list(eval_result.get("suggested_dataset_names") or [])
    sug_flags = list(eval_result.get("suggested_flag_terms") or [])
    retry_names = sug_names if sug_names else list(body.dataset_names)
    retry_flags = sug_flags if sug_flags else list(body.flag_terms)
    merged_exclude = _merge_exclude_terms(
        body.exclude_terms,
        list(eval_result.get("suggested_exclude_terms") or []),
    )

    record.terms_evaluation = TermsEvaluation(
        is_effective=eval_result["is_effective"],
        dataset_names_score=eval_result["dataset_names_score"],
        flag_terms_score=eval_result["flag_terms_score"],
        issues=eval_result["issues"],
        suggested_dataset_names=eval_result["suggested_dataset_names"],
        suggested_flag_terms=eval_result["suggested_flag_terms"],
        suggested_exclude_terms=list(eval_result.get("suggested_exclude_terms") or []),
        reasoning=eval_result["reasoning"],
    )

    base = str(request.base_url).rstrip("/")
    record.retry_validation = ValidationRequest(
        main_dataset_name=body.main_dataset_name,
        dataset_names=retry_names,
        flag_terms=retry_flags,
        description=body.description,
        sample_size=body.sample_size,
        llm_batch_size=body.llm_batch_size,
        exclude_terms=merged_exclude,
    )
    record.links = {
        "retry": {
            "href": f"{base}/validate",
            "method": "POST",
            "rel": "retry-validation",
        },
    }

    return record


@app.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    body: dict[str, Any] = {"status": "ok"}
    if settings.llm_provider == "ollama":
        body["ollama"] = _probe_ollama(settings.ollama_base_url)
    else:
        body["ollama"] = {
            "checked": False,
            "reason": "DATASET_AGENT_LLM_PROVIDER is not ollama",
        }
    return body

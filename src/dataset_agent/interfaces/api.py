"""FastAPI HTTP interface."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any
from urllib.parse import urljoin

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dataset_agent.domain.models import (
    DatasetRecord,
    Group,
    LiteratureValidation,
    OptimizeRequest,
    QueryOptimizationRecord,
    ResearchRequest,
    TermsEvaluation,
    ValidationRequest,
    YearsRange,
)
from dataset_agent.domain.ports import TaskStatus
from dataset_agent.adapters.auth import APIKeyAuth
from dataset_agent.adapters.literature import promote_dimensions_metrics_from_detail
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

_cors_origins = [o.strip() for o in Settings().cors_origins.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_use_case(settings: Settings):
    """Lazy wrapper so importing this module does not eagerly load the LLM client."""
    from dataset_agent.bootstrap import build_use_case as _bootstrap_build

    return _bootstrap_build(settings)


def build_optimize_use_case(settings: Settings):
    from dataset_agent.bootstrap import build_optimize_use_case as _bootstrap_opt

    return _bootstrap_opt(settings)


def _optimize_http_status(record: QueryOptimizationRecord) -> int:
    """503 para falhas de infraestrutura Dimensions; 200 para resultados de negócio (incl. success=false)."""
    if record.success:
        return 200
    phase = (record.failed_phase or "").strip()
    blob = " ".join(record.errors or []).lower()
    if phase == "precheck":
        return 503
    if phase == "phase1_alias_counting":
        if "all aliases returned zero" in blob:
            return 200
        return 503
    if phase == "phase2_variant_building":
        if "no query variants could be built" in blob:
            return 200
        if "zero publications" in blob:
            return 200
        if "all variant probes failed" in blob:
            return 503
        return 200
    return 200


def get_settings() -> Settings:
    return Settings()


def verify_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> str:
    """Dependency that enforces API key auth on write endpoints.

    Auth is skipped entirely when no keys are configured in settings.
    """
    auth = APIKeyAuth(settings.api_keys)
    if not auth.enabled:
        return "anonymous"
    if not auth.is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return auth.get_client_id(x_api_key)


class TaskCreateResponse(BaseModel):
    id: str
    status: str


_OPTIMIZE_TASK_TYPE = "optimize"


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
        if isinstance(payload, dict) and payload.get("task_type") == _OPTIMIZE_TASK_TYPE:
            opt_body = {k: v for k, v in payload.items() if k != "task_type"}
            opt_req = OptimizeRequest.model_validate(opt_body)
            logger.info(
                "Task %s: running optimize pipeline dataset_name=%r",
                task_id,
                opt_req.dataset_name,
            )
            from dataset_agent.bootstrap import build_optimize_use_case as _bootstrap_optimize

            uc = _bootstrap_optimize(settings)
            opt_record = asyncio.run(uc.execute(opt_req))
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = settings.output_dir / f"optimize-{task_id}.json"
            out_path.write_text(opt_record.model_dump_json(), encoding="utf-8")
            tasks.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                result_path=out_path.name,
            )
            logger.info("Task %s: optimize finished; result at %s", task_id, out_path)
            return

        request = ResearchRequest.model_validate(payload)
        logger.info(
            "Task %s: running pipeline dataset_name=%r webhook=%s",
            task_id,
            request.dataset_name,
            "yes" if request.webhook_url else "no",
        )
        from dataset_agent.bootstrap import run_full_pipeline

        record, path = asyncio.run(run_full_pipeline(request, settings))
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
    _client: Annotated[str, Depends(verify_api_key)],
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

    allowed_root = settings.output_dir.resolve()
    p = allowed_root / Path(path).name
    try:
        p.relative_to(allowed_root)
    except ValueError:
        raise HTTPException(status_code=500, detail="Result path outside allowed directory")
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
    _client: Annotated[str, Depends(verify_api_key)],
) -> DatasetRecord:
    """
    Validate dataset terms against Dimensions publications.
    Returns same JSON format as /tasks endpoint result.
    """
    from dataset_agent.adapters.literature import literature_gate_from_settings, evaluate_terms_with_llm
    from dataset_agent.bootstrap import _build_agent
    
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
    agent = _build_agent(settings)
    
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
    promote_dimensions_metrics_from_detail(record, result.detail)

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


@app.post(
    "/optimize",
    response_model=QueryOptimizationRecord,
    summary="Otimizar query Dimensions",
    description=(
        "Executa o pipeline de otimização em modo síncrono e devolve a melhor variante de query "
        "Dimensions, com métricas de confiança e `exclusion_terms` sugeridos.\n\n"
        "Integração recomendada com `/validate`: copie `exclusion_terms` da resposta para "
        "`exclude_terms` no corpo do `POST /validate` para validar a query filtrada. "
        "A API não persiste estes termos automaticamente em `DatasetRecord`.\n\n"
        "Para execução em background (evitar timeout HTTP), use `POST /optimize/tasks` e "
        "`GET /optimize/tasks/{task_id}/result`."
    ),
    responses={
        422: {"description": "Corpo inválido (validação Pydantic)."},
        503: {
            "description": "Dimensions não configurado ou erro ao contactar a API (ver `errors` / `failed_phase`).",
            "model": QueryOptimizationRecord,
        },
    },
)
async def optimize_endpoint(
    body: OptimizeRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    _client: Annotated[str, Depends(verify_api_key)],
) -> QueryOptimizationRecord | JSONResponse:
    """Pipeline síncrono em quatro fases. Configure timeout do cliente e do reverse proxy (ex.: 10–30 min)."""
    uc = build_optimize_use_case(settings)
    record = await uc.execute(body)
    status = _optimize_http_status(record)
    if status != 200:
        return JSONResponse(
            status_code=status,
            content=record.model_dump(mode="json"),
        )
    return record


@app.post(
    "/optimize/tasks",
    response_model=TaskCreateResponse,
    summary="Enfileirar otimização Dimensions (async)",
    description=(
        "Aceita o mesmo corpo que `POST /optimize`. O worker grava `optimize-{task_id}.json` "
        "em `output_dir` e expõe o resultado em `GET /optimize/tasks/{task_id}/result`."
    ),
)
def create_optimize_task(
    body: OptimizeRequest,
    background_tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings)],
    _client: Annotated[str, Depends(verify_api_key)],
) -> TaskCreateResponse:
    tasks = SqliteTaskRepository(settings.tasks_db)
    payload = {"task_type": _OPTIMIZE_TASK_TYPE, **body.model_dump(mode="json")}
    tid = tasks.create_task(payload)
    logger.info(
        "Optimize task created id=%s dataset_name=%r (pending; background worker)",
        tid,
        body.dataset_name,
    )
    background_tasks.add_task(_run_background_task, tid, settings)
    return TaskCreateResponse(id=tid, status=TaskStatus.PENDING)


@app.get("/optimize/tasks/{task_id}/result", response_model=QueryOptimizationRecord)
def get_optimize_task_result(
    task_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryOptimizationRecord:
    """Lê o ficheiro JSON produzido pelo worker para tarefas `POST /optimize/tasks`."""
    from pathlib import Path

    tasks = SqliteTaskRepository(settings.tasks_db)
    row = tasks.get_task(task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    payload = row.get("payload")
    if not isinstance(payload, dict) or payload.get("task_type") != _OPTIMIZE_TASK_TYPE:
        raise HTTPException(status_code=400, detail="Not an optimize task")
    if row["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Task not completed: {row['status']}")
    path = row.get("result_path")
    if not path:
        raise HTTPException(status_code=500, detail="No result path")
    allowed_root = settings.output_dir.resolve()
    p = allowed_root / Path(path).name
    try:
        p.relative_to(allowed_root)
    except ValueError:
        raise HTTPException(status_code=500, detail="Result path outside allowed directory")
    if not p.is_file():
        raise HTTPException(status_code=500, detail="Result file missing")
    return QueryOptimizationRecord.model_validate_json(p.read_text(encoding="utf-8"))


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

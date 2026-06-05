"""Wire concrete adapters from Settings."""

from __future__ import annotations

from dataset_agent.adapters.agent_anthropic import AnthropicAgent
from dataset_agent.adapters.dimensions_dsl import AsyncThrottledDimensionsDsl
from dataset_agent.adapters.extractor import HeuristicTextExtractor
from dataset_agent.adapters.literature import literature_gate_from_settings
from dataset_agent.adapters.storage_json import JsonDatasetRepository
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.application.research import DatasetResearchUseCase
from dataset_agent.domain.models import OptimizeRequest, ResearchRequest
from dataset_agent.domain.ports import AgentPort
from dataset_agent.settings import Settings


def _build_agent(settings: Settings) -> AnthropicAgent:
    """Create an AnthropicAgent from the current settings."""
    if settings.llm_provider == "lmstudio":
        base_url = settings.lmstudio_base_url
        model_name = settings.lmstudio_model
    else:
        base_url = settings.ollama_base_url
        model_name = settings.ollama_model
    return AnthropicAgent(
        model_name=model_name,
        base_url=base_url,
        api_key=settings.llm_api_key,
        max_tokens=settings.llm_max_tokens,
        max_iterations=settings.agent_max_iterations,
        timeout_seconds=settings.agent_timeout_seconds,
    )


def _shared_dsl(settings: Settings) -> AsyncThrottledDimensionsDsl:
    return AsyncThrottledDimensionsDsl(settings)


def build_use_case(
    settings: Settings,
    agent: AgentPort | None = None,
    dsl_port: AsyncThrottledDimensionsDsl | None = None,
) -> DatasetResearchUseCase:
    resolved = agent or _build_agent(settings)
    extractor = HeuristicTextExtractor()
    repo = JsonDatasetRepository(settings.output_dir)
    dsl = dsl_port or _shared_dsl(settings)
    gate = literature_gate_from_settings(settings, agent=resolved, dsl_port=dsl)
    return DatasetResearchUseCase(
        resolved,
        extractor,
        repo,
        settings,
        literature_gate=gate,
        dsl_port=dsl,
    )


def build_optimize_use_case(
    settings: Settings,
    agent: AgentPort | None = None,
    dsl_port: AsyncThrottledDimensionsDsl | None = None,
) -> QueryOptimizationUseCase:
    """Dimensions query optimization (aliases → variants → gate evaluation → selection)."""
    resolved = agent or _build_agent(settings)
    dsl = dsl_port or _shared_dsl(settings)
    research = build_use_case(settings, agent=resolved, dsl_port=dsl)
    return QueryOptimizationUseCase(
        dsl_port=dsl,
        research=research,
        agent=resolved,
        settings=settings,
    )


async def run_full_pipeline(
    request: ResearchRequest,
    settings: Settings,
) -> tuple[object, object]:
    """Orchestrator: research → optimization+gate → combined result."""
    key = (settings.dimensions_api_key or "").strip()
    if not key:
        raise ValueError(
            "Dimensions API key is not configured (set DATASET_AGENT_DIMENSIONS_API_KEY)"
        )

    agent = _build_agent(settings)
    dsl = _shared_dsl(settings)
    repo = JsonDatasetRepository(settings.output_dir)

    # Research
    research_uc = build_use_case(settings, agent=agent, dsl_port=dsl)
    record, path = research_uc.execute(request)

    # Optimization (seeded with research terms to avoid double research)
    optimize_uc = build_optimize_use_case(settings, agent=agent, dsl_port=dsl)
    # Flatten alias_collisions dict-of-lists into a single exclude list
    collision_excludes = []
    for terms in record.alias_collisions.values():
        collision_excludes.extend(terms)
    opt_req = OptimizeRequest(
        dataset_name=request.dataset_name,
        dataset_url=request.dataset_url,
        dataset_names=record.dataset_names,
        flag_terms=record.flag_terms,
        exclude_terms=list(record.exclude_terms) + collision_excludes[:25],
    )
    opt_record = await optimize_uc.execute(opt_req)

    record.query_optimization = opt_record
    path = repo.save(record)
    return record, path

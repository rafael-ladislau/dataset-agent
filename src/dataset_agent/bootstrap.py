"""Wire concrete adapters from Settings."""

from __future__ import annotations

import logging

from dataset_agent.adapters.agent_anthropic import AnthropicAgent
from dataset_agent.adapters.agent_openai import OpenAICompatibleAgent
from dataset_agent.adapters.dimensions_dsl import AsyncThrottledDimensionsDsl
from dataset_agent.adapters.proposals_json import JsonProposalRepository
from dataset_agent.adapters.extractor import HeuristicTextExtractor
from dataset_agent.adapters.literature import literature_gate_from_settings
from dataset_agent.adapters.storage_json import JsonDatasetRepository
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.application.research import DatasetResearchUseCase
from dataset_agent.domain.models import LLMSampling, OptimizeRequest, ResearchRequest
from dataset_agent.domain.ports import AgentPort
from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)


def _build_agent(settings: Settings) -> AgentPort:
    """Create the LLM agent adapter for the configured provider and API style."""
    endpoint = settings.llm_endpoint()
    agent_cls = OpenAICompatibleAgent if endpoint.api_style == "openai" else AnthropicAgent
    return agent_cls(
        model_name=endpoint.model,
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        max_tokens=settings.llm_max_tokens,
        max_iterations=settings.agent_max_iterations,
        timeout_seconds=settings.agent_timeout_seconds,
    )


#: Temperature of the main agent when a request does not override it (the adapters' default).
MAIN_AGENT_TEMPERATURE = 0.6

_SAMPLING_FIELDS = ("model", "temperature", "top_p", "top_k")


def has_sampling_overrides(request: ResearchRequest) -> bool:
    return any(getattr(request, f) is not None for f in _SAMPLING_FIELDS)


def build_request_agent(
    settings: Settings, request: ResearchRequest
) -> tuple[AgentPort, LLMSampling]:
    """The research/optimize agent for one request, with the request's model and sampling
    overrides applied, and a record of what it runs with. Settings are never changed, so an
    override lasts for this request only. ``top_k`` is not sent by the OpenAI-compatible
    style and is recorded as not applied."""
    endpoint = settings.llm_endpoint()
    openai_style = endpoint.api_style == "openai"
    requested = [f for f in _SAMPLING_FIELDS if getattr(request, f) is not None]
    not_applied = ["top_k"] if openai_style and request.top_k is not None else []
    sampling = LLMSampling(
        api_style=endpoint.api_style,
        model=request.model or endpoint.model,
        temperature=(MAIN_AGENT_TEMPERATURE if request.temperature is None
                     else request.temperature),
        top_p=request.top_p,
        top_k=None if openai_style else request.top_k,
        overridden=[f for f in requested if f not in not_applied],
        not_applied=not_applied,
    )
    if not requested:
        return _build_agent(settings), sampling
    common = dict(
        model_name=sampling.model,
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        max_tokens=settings.llm_max_tokens,
        max_iterations=settings.agent_max_iterations,
        timeout_seconds=settings.agent_timeout_seconds,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
    )
    if openai_style:
        return OpenAICompatibleAgent(**common), sampling
    return AnthropicAgent(**common, top_k=sampling.top_k), sampling


def build_referent_agent(settings: Settings) -> AgentPort:
    """The agent for the full-text referent question: the main agent, unless
    ``referent_thinking_budget`` asks for thinking on that question only."""
    if not settings.referent_thinking_budget:
        return _build_agent(settings)
    endpoint = settings.llm_endpoint()
    if endpoint.api_style != "anthropic":
        logger.warning("referent_thinking_budget needs the anthropic API style; the referent "
                       "question runs on the main agent without thinking")
        return _build_agent(settings)
    return AnthropicAgent(
        model_name=endpoint.model,
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        max_tokens=settings.llm_max_tokens,
        max_iterations=settings.agent_max_iterations,
        timeout_seconds=settings.agent_timeout_seconds,
        temperature=0.6 if settings.referent_temperature is None else settings.referent_temperature,
        thinking_budget=settings.referent_thinking_budget,
        top_p=settings.referent_top_p,
        top_k=settings.referent_top_k,
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
    referent_agent: AgentPort | None = None,
    main_overridden: bool = False,
) -> QueryOptimizationUseCase:
    """Dimensions query optimization (aliases → variants → gate evaluation → selection).

    ``referent_agent`` runs the full-text referent judge and the scope check; when not given
    it is built from Settings only if ``referent_thinking_budget`` asks for it (otherwise
    those questions run on ``agent``)."""
    resolved = agent or _build_agent(settings)
    dsl = dsl_port or _shared_dsl(settings)
    research = build_use_case(settings, agent=resolved, dsl_port=dsl)
    return QueryOptimizationUseCase(
        dsl_port=dsl,
        research=research,
        agent=resolved,
        settings=settings,
        proposal_repo=JsonProposalRepository(settings.output_dir),
        referent_agent=referent_agent or (
            build_referent_agent(settings) if settings.referent_thinking_budget else None
        ),
        main_overridden=main_overridden,
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

    agent, sampling = build_request_agent(settings, request)
    # Request overrides stop at the research/optimize agent: the full-text referent judge
    # and the scope check keep the configured model and sampling.
    referent_agent = (build_referent_agent(settings) if has_sampling_overrides(request)
                      else None)
    dsl = _shared_dsl(settings)
    repo = JsonDatasetRepository(settings.output_dir)

    # Research
    research_uc = build_use_case(settings, agent=agent, dsl_port=dsl)
    record, path = research_uc.execute(request)

    # Optimization (seeded with research terms to avoid double research)
    optimize_uc = build_optimize_use_case(settings, agent=agent, dsl_port=dsl,
                                          referent_agent=referent_agent,
                                          main_overridden=referent_agent is not None)
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
        resource_map=record.resource_map,
        resource_map_checked=record.resource_map_checked,
        resource_map_error=record.resource_map_error,
        resource_map_decision=record.resource_map_decision,
        refresh_resource_map=request.refresh_resource_map,
        dataset_description=record.description or "",
    )
    opt_record = await optimize_uc.execute(opt_req)

    record.llm_sampling = sampling
    if not record.resource_map_checked:
        # Research could not decide the map and optimize drew it (and stored it): carry
        # the stored decision onto the record saved below, or that save would drop it.
        stored_fn = getattr(research_uc, "stored_map_decision", None)
        stored = stored_fn(record.main_dataset_name) if callable(stored_fn) else None
        if isinstance(stored, type(record)):
            record.resource_map = stored.resource_map
            record.resource_map_checked = True
            record.resource_map_error = None
            record.resource_map_decision = stored.resource_map_decision
            record.program_datasets = stored.program_datasets

    record.query_optimization = opt_record
    path = repo.save(record)
    return record, path

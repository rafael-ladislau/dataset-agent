"""Wire concrete adapters from Settings."""

from __future__ import annotations

from dataset_agent.adapters.agent_anthropic import AnthropicAgent
from dataset_agent.adapters.dimensions_dsl import AsyncThrottledDimensionsDsl
from dataset_agent.adapters.extractor import HeuristicTextExtractor
from dataset_agent.adapters.literature import literature_gate_from_settings
from dataset_agent.adapters.storage_json import JsonDatasetRepository
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.application.research import DatasetResearchUseCase
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


def build_use_case(settings: Settings, agent: AgentPort | None = None) -> DatasetResearchUseCase:
    resolved = agent or _build_agent(settings)
    extractor = HeuristicTextExtractor()
    repo = JsonDatasetRepository(settings.output_dir)
    gate = literature_gate_from_settings(settings, agent=resolved)
    return DatasetResearchUseCase(
        resolved,
        extractor,
        repo,
        settings,
        literature_gate=gate,
    )


def build_optimize_use_case(settings: Settings) -> QueryOptimizationUseCase:
    """Dimensions query optimization (aliases → variants → FP proxy → selection)."""
    agent = _build_agent(settings)
    dsl = AsyncThrottledDimensionsDsl(settings)
    research = build_use_case(settings, agent=agent)
    return QueryOptimizationUseCase(
        dsl_port=dsl,
        research=research,
        agent=agent,
        settings=settings,
    )

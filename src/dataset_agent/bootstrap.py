"""Wire concrete adapters from Settings."""

from __future__ import annotations

from dataset_agent.adapters.agent_anthropic import AnthropicAgent
from dataset_agent.adapters.extractor import HeuristicTextExtractor
from dataset_agent.adapters.literature import literature_gate_from_settings
from dataset_agent.adapters.storage_json import JsonDatasetRepository
from dataset_agent.application.research import DatasetResearchUseCase
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


def build_use_case(settings: Settings) -> DatasetResearchUseCase:
    agent = _build_agent(settings)
    extractor = HeuristicTextExtractor()
    repo = JsonDatasetRepository(settings.output_dir)
    gate = literature_gate_from_settings(settings, agent=agent)
    return DatasetResearchUseCase(
        agent,
        extractor,
        repo,
        settings,
        literature_gate=gate,
    )

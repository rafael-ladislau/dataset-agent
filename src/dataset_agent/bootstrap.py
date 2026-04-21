"""Wire concrete adapters from Settings."""

from __future__ import annotations

from dataset_agent.adapters.agent_langchain import LangChainAgent
from dataset_agent.adapters.extractor import HeuristicTextExtractor
from dataset_agent.adapters.literature import literature_gate_from_settings
from dataset_agent.adapters.storage_json import JsonDatasetRepository
from dataset_agent.application.research import DatasetResearchUseCase
from dataset_agent.settings import Settings


def build_use_case(settings: Settings) -> DatasetResearchUseCase:
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

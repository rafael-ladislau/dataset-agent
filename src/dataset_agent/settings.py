"""Application settings (environment + defaults)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATASET_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["ollama", "lmstudio"] = "lmstudio"
    ollama_model: str = "gemma4-31b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    lmstudio_base_url: str = "http://127.0.0.1:1234"
    lmstudio_model: str = "gemma-4-31b-it-mlx"
    llm_max_tokens: int = 4096
    llm_api_key: str = "local"

    agent_max_iterations: int = 5
    agent_timeout_seconds: int = 120

    output_dir: Path = Path("./output")
    tasks_db: Path = Path("./tasks.db")

    default_engine: str = "dimensions"
    default_group_name: str = "default"
    default_years_start: int = 2015
    default_years_end: int = 2025
    default_publication_types: list[str] = Field(default_factory=lambda: ["article"])
    default_filter_us_affiliation: bool = False

    literature_gate: Literal["noop", "dimensions"] = "noop"
    literature_max_attempts: int = 3
    literature_threshold: float = 0.5
    dimensions_api_key: str = ""

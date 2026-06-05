"""Application settings (environment + defaults)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
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
    lmstudio_base_url: str = Field(
        default="http://127.0.0.1:1234",
        validation_alias=AliasChoices(
            "LMSTUDIO_BASE_URL",
            "DATASET_AGENT_LMSTUDIO_BASE_URL",
        ),
    )
    lmstudio_model: str = Field(
        default="gemma-4-31b-it-mlx",
        validation_alias=AliasChoices(
            "DATASET_AGENT_LLM_MODEL",
            "DATASET_AGENT_LMSTUDIO_MODEL",
        ),
    )
    llm_max_tokens: int = 4096
    llm_api_key: str = Field(
        default="local",
        validation_alias=AliasChoices("LMSTUDIO_API_KEY", "DATASET_AGENT_LLM_API_KEY"),
    )

    #: Web search backend (from .env without DATASET_AGENT_ prefix).
    web_search_provider: str = Field(
        default="duckduckgo",
        validation_alias="WEB_SEARCH_PROVIDER",
    )
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")

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

    #: Minimum interval between consecutive Dimensions DSL query *starts* (serial throttle).
    dimensions_rate_limit_seconds: float = 2.1
    #: Publications sample size for false-positive verification (capped at 10_000).
    fp_sample_size: int = 1000
    #: Hard cap on Dimensions calls per optimize run (0 = no extra cap beyond use-case logic).
    optimize_max_dimensions_calls: int = Field(default=64, ge=0)
    #: Top publication titles scored with LLM Signal 3 on the primary variant (0 = skip).
    optimize_signal3_max_titles: int = Field(default=5, ge=0, le=20)
    #: Titles with mention_score + context_score below this are treated as likely FP (0–20 scale).
    optimize_signal3_min_relevance_sum: int = Field(default=10, ge=0, le=20)
    #: Max abstract snippets sent to LLM to derive NOT terms (0 = skip abstract→exclude step).
    optimize_abstract_exclude_max_hits: int = Field(default=16, ge=0, le=80)
    #: Max abstract hits passed through ``classify_alias_mention`` before derive (0 = skip filtering).
    optimize_abstract_classify_max_calls: int = Field(default=12, ge=0, le=80)
    #: When true, try suffix disambiguation (dataset/survey/…) for risky aliases in optimize Phase 1.
    optimize_suffix_check_enabled: bool = True
    #: Max risky aliases to test with suffix candidates (0 = skip).
    optimize_suffix_check_max_aliases: int = Field(default=3, ge=0, le=10)
    #: Bare/suffixed count ratio required to promote a normal risky alias.
    optimize_suffix_ratio_threshold: float = Field(default=100.0, gt=0.0)
    #: Lower ratio for short acronyms (≤4 chars), e.g. ONET vs ONET dataset (~9×).
    optimize_suffix_ratio_short_acronym: float = Field(default=10.0, gt=0.0)
    #: Max risky aliases for LLM FP domain analysis in optimize Phase 3 (0 = crude keyword fallback).
    optimize_fp_domains_max_aliases: int = Field(default=3, ge=0, le=10)
    #: Max risky aliases for web-search disambiguation in optimize Phase 3 (0 = skip).
    optimize_web_disambig_max_aliases: int = Field(default=2, ge=0, le=5)
    #: Append multilingual / unaccented alias variants during research Step 5.
    research_multilingual_aliases: bool = True

    @field_validator("lmstudio_base_url")
    @classmethod
    def _normalize_lmstudio_base_url(cls, v: str) -> str:
        """Anthropic SDK expects host root without /v1 (it appends /v1/messages)."""
        base = v.rstrip("/")
        if base.endswith("/v1"):
            return base[:-3]
        return base

    @field_validator("fp_sample_size")
    @classmethod
    def _fp_sample_size_bounds(cls, v: int) -> int:
        if v < 1 or v > 10_000:
            raise ValueError("fp_sample_size must be between 1 and 10000")
        return v

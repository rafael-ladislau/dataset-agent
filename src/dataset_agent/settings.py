"""Application settings (environment + defaults)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "lmstudio", "openrouter", "openllm", "openai_compatible"]
LLMApiStyle = Literal["anthropic", "openai"]

#: API style used when ``llm_api_style`` is not set.
DEFAULT_LLM_API_STYLE: dict[str, LLMApiStyle] = {
    "lmstudio": "openai",
    "ollama": "anthropic",
    "openrouter": "anthropic",
    "openllm": "openai",
    "openai_compatible": "openai",
}


@dataclass(frozen=True)
class LLMEndpoint:
    """Resolved LLM connection: which adapter to build and where it points."""

    provider: str
    api_style: LLMApiStyle
    #: Anthropic style: host root (the SDK appends ``/v1/messages``).
    #: OpenAI style: API root, usually ending in ``/v1``.
    base_url: str
    model: str
    api_key: str


def _with_v1(url: str) -> str:
    base = url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _without_v1(url: str) -> str:
    base = url.rstrip("/")
    return base.removesuffix("/v1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATASET_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProvider = "lmstudio"
    #: Wire protocol: ``anthropic`` (/v1/messages) or ``openai`` (/v1/chat/completions).
    #: Unset uses the provider default (``openai`` for lmstudio, openllm and
    #: openai_compatible; ``anthropic`` for ollama and openrouter).
    llm_api_style: LLMApiStyle | None = None
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
    #: OpenRouter (Anthropic-compatible) endpoint and credentials.
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices(
            "OPENROUTER_BASE_URL",
            "DATASET_AGENT_OPENROUTER_BASE_URL",
        ),
    )
    openrouter_model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        validation_alias=AliasChoices(
            "DATASET_AGENT_OPENROUTER_MODEL",
        ),
    )
    openrouter_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "OPENROUTER_API_KEY",
            "DATASET_AGENT_OPENROUTER_API_KEY",
        ),
    )
    #: OpenLLM (OpenAI-compatible) endpoint; the base URL includes ``/v1``.
    openllm_base_url: str = Field(
        default="http://127.0.0.1:3000/v1",
        validation_alias=AliasChoices(
            "OPENLLM_BASE_URL",
            "DATASET_AGENT_OPENLLM_BASE_URL",
        ),
    )
    openllm_model: str = ""
    openllm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "OPENLLM_API_KEY",
            "DATASET_AGENT_OPENLLM_API_KEY",
        ),
    )
    #: Any other OpenAI-compatible server (vLLM, llama.cpp, hosted APIs); base URL as given.
    openai_compatible_base_url: str = ""
    openai_compatible_model: str = ""
    openai_compatible_api_key: str = ""
    llm_max_tokens: int = 8192
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

    #: Comma-separated API keys that secure write endpoints (empty disables auth).
    api_keys: str = Field(default="", validation_alias=AliasChoices("API_KEYS", "DATASET_AGENT_API_KEYS"))
    #: Comma-separated CORS allowed origins ("*" allows all).
    cors_origins: str = Field(
        default="*",
        validation_alias=AliasChoices("CORS_ORIGINS", "DATASET_AGENT_CORS_ORIGINS"),
    )

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

    literature_gate: Literal["noop", "dimensions"] = "dimensions"
    literature_max_attempts: int = 3
    literature_threshold: float = 0.5
    dimensions_api_key: str = ""

    #: Minimum interval between consecutive Dimensions DSL query *starts* (serial throttle).
    dimensions_rate_limit_seconds: float = 2.1
    #: (connect, read) timeout for every Dimensions HTTP call; dimcli sets none, so a
    #: stalled server response would otherwise block forever.
    dimensions_connect_timeout_seconds: float = Field(default=15.0, gt=0)
    dimensions_timeout_seconds: float = Field(default=180.0, gt=0)
    #: Publications sample size for false-positive verification (capped at 10_000).
    fp_sample_size: int = 1000
    #: Hard cap on Dimensions calls per optimize run (0 = no extra cap beyond use-case logic).
    optimize_max_dimensions_calls: int = Field(default=96, ge=0)
    #: When the dataset title has no Dimensions hits and its aliases together find fewer
    #: than ``optimize_grounding_min_hits`` papers, ask the LLM (one structured call) for
    #: the proper names the description refers to and keep those Dimensions finds.
    optimize_grounding_enabled: bool = True
    optimize_grounding_min_hits: int = Field(default=100, ge=0)
    #: A grounded name is kept only with at least this many Dimensions hits (names only
    #: found defined with an acronym in web results need ``optimize_grounding_min_hits``).
    optimize_grounding_min_alias_hits: int = Field(default=5, ge=1)
    #: Max grounded names counted in Dimensions per run.
    optimize_grounding_max_candidates: int = Field(default=8, ge=0, le=20)
    #: An unvouched (web-defined) grounded name with more hits than this is a common
    #: term, not a collection name (0 = no ceiling).
    optimize_grounding_unvouched_max_hits: int = Field(default=20000, ge=0)
    #: Look up former/successor names of the collection (renames,
    #: merges) after phase 1 counting; one web search per leading name plus one LLM call.
    optimize_name_history_enabled: bool = True
    #: Max history names counted in Dimensions per run.
    optimize_name_history_max_candidates: int = Field(default=6, ge=0, le=20)
    #: Research flags with more Dimensions full-data hits than this ("GAGE": 5.3M, as
    #: "gage" also means gauge) admit any paper; dropped when other flags remain (0 = off).
    optimize_flag_max_hits: int = Field(default=500_000, ge=0)
    #: Drop bare short-acronym research flags (<= 4 characters) when other flags remain.
    optimize_drop_short_acronym_flags: bool = True
    #: A grounded or history name is searched without flags only when at least this share
    #: of its Dimensions hits also contain a flag term; below it the name is flag-gated.
    optimize_grounded_safe_min_flag_share: float = Field(default=0.5, ge=0.0, le=1.0)
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
    #: Max top aliases to run web disambiguation collision check on (0 = skip).
    research_collision_max_aliases: int = Field(default=3, ge=0, le=10)
    #: Max risky/short aliases to run suffix-necessity check on during research (0 = skip).
    research_suffix_check_max_aliases: int = Field(default=3, ge=0, le=10)

    #: FP rate % threshold that triggers refinement in the optimization loop.
    optimize_fp_threshold_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    #: Max optimization loop iterations (build DSL → gate → refine).
    optimize_max_iterations: int = Field(default=3, ge=1, le=10)

    #: Enable reference-id resolution in the literature gate evaluator.
    gate_ref_resolution_enabled: bool = True
    #: Coverage threshold (% of pulled pubs with alias stems in title+abstract) to skip reference resolution.
    gate_ref_min_title_abstract_match_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    #: Hard cap on unique reference ids resolved per gate evaluation run.
    gate_ref_max_unique_ids: int = Field(default=2000, ge=0)
    #: Batch size for reference-id resolution queries (Dimensions limit).
    gate_ref_batch_size: int = Field(default=512, ge=1, le=1000)

    # --- Full-text query refinement (extraction pipeline probes) ---------------
    #: Run full-text probes against the extraction pipeline during Phase 3.
    fulltext_refinement_enabled: bool = False
    #: Papers with delivered open full text a probe tries to collect (top-up target).
    fulltext_target_delivered: int = Field(default=30, ge=1, le=500)
    #: Publications sent per analyze batch, next in citation order not yet attempted.
    fulltext_batch_size: int = Field(default=30, ge=1, le=500)
    #: Hard stop on analyze batches per probe (batch_size x max_batches candidates).
    fulltext_max_batches: int = Field(default=8, ge=1, le=50)
    #: Max publications pulled from Dimensions by a probe task (extraction ``probe_pull_cap``).
    #: This is the pool the top-up batches draw from.
    probe_pull_cap: int = Field(default=300, ge=1, le=1000)
    #: Wall-clock budget for the probe search and for each analyze batch (full text + LLM).
    probe_poll_timeout_seconds: float = Field(default=3600.0, gt=0.0)
    probe_poll_interval_seconds: float = Field(default=3.0, gt=0.0)
    probe_poll_max_interval_seconds: float = Field(default=20.0, gt=0.0)
    #: Min full-text-delivered probe papers needed before a term may be changed.
    fulltext_min_evidence: int = Field(default=3, ge=1)
    #: Recall check after the precision loop: propose adding missing canonical names as
    #: safe aliases when a complement probe shows they are precise. Needs full-text refinement.
    fulltext_recall_enabled: bool = True
    #: Max recall candidates evaluated (one complement probe each) per run.
    fulltext_recall_max_candidates: int = Field(default=2, ge=0, le=5)
    #: Min complement precision (papers matching the candidate but not the current query);
    #: also the MEDIUM confidence floor.
    fulltext_recall_min_precision: float = Field(default=0.8, ge=0.0, le=1.0)
    #: Min full-text papers evaluated in a complement probe (and min Dimensions hits added);
    #: also the sample needed for MEDIUM or HIGH confidence.
    fulltext_recall_min_evidence: int = Field(default=10, ge=1)
    #: A promotion adding more than ratio x the selected query's count is a large expansion.
    fulltext_recall_expansion_ratio: float = Field(default=1.0, ge=0.0)
    #: For a large expansion the Wilson 95% lower bound of complement precision must reach
    #: fulltext_recall_min_precision minus this margin (a point estimate on ~25 top-cited
    #: papers cannot justify multiplying the query).
    fulltext_recall_expansion_margin: float = Field(default=0.1, ge=0.0, le=1.0)
    #: Delivered full texts a complement probe tries to collect (raised to the min evidence).
    fulltext_recall_target_delivered: int = Field(default=20, ge=1, le=500)
    #: Extraction instance used for probes ("dev" or "prod"; probes never publish).
    probe_target: Literal["dev", "prod"] = "dev"
    #: Extraction instance a proposal is submitted to. "prod" also requires an approved proposal.
    submit_target: Literal["dev", "prod"] = "dev"
    #: Empty by default: dataset-agent itself listens on 127.0.0.1:8000, so there is
    #: no safe default; a probe with no URL fails clearly ("not configured").
    extraction_dev_url: str = ""
    extraction_dev_user: str = ""
    extraction_dev_password: str = ""
    extraction_prod_url: str = ""
    extraction_prod_user: str = ""
    extraction_prod_password: str = ""
    #: Deterministic refinement thresholds (see docs/fulltext-refinement.md).
    refine_min_alias_papers: int = Field(default=3, ge=1)
    refine_drop_fp_share: float = Field(default=0.8, ge=0.0, le=1.0)
    refine_demote_fp_share: float = Field(default=0.5, ge=0.0, le=1.0)
    # --- AI-model queries (OptimizeRequest.entity_type == "ai_model") ---
    #: Extra Dimensions calls an AI-model run may spend on measuring candidate terms.
    ai_model_extra_dimensions_calls: int = Field(default=140, ge=0)
    ai_model_max_names: int = Field(default=14, ge=1, le=40)
    ai_model_max_context_terms: int = Field(default=12, ge=0, le=30)
    ai_model_max_exclude_terms: int = Field(default=8, ge=0, le=30)
    #: Context terms mined from the abstracts of papers that certainly refer to the
    #: model (their paper titles or safe names), on top of the researched ones.
    ai_model_max_mined_context_terms: int = Field(default=15, ge=0, le=30)
    ai_model_context_mining_sample: int = Field(default=200, ge=0, le=1000)
    #: A paper title not containing the model name is an anchor only if this share of
    #: the papers quoting it also name the model.
    ai_model_title_min_name_share: float = Field(default=0.25, ge=0.0, le=1.0)
    #: ...and only if no more papers than this quote it (a card may cite a famous
    #: paper that many unrelated works quote too).
    ai_model_title_max_hits: int = Field(default=50_000, ge=1)
    #: A context term gates an ambiguous name only if at least this share of the papers
    #: it admits ("<name>" AND "<term>") already match a precise name or paper title.
    ai_model_flag_min_anchor_precision: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Below the flag bar but above this one, a term is reported as "context" (related
    #: to the model, too loose to gate a name); below it the term is rejected.
    ai_model_context_min_anchor_precision: float = Field(default=0.1, ge=0.0, le=1.0)
    #: A flag must be specific in its own right: a term with more Dimensions hits than
    #: this is a topic or a household name, however well it scores next to the model.
    ai_model_flag_max_hits: int = Field(default=50_000, ge=1)
    #: Min papers matching both a precise name/title and the context term.
    ai_model_flag_min_anchor_hits: int = Field(default=5, ge=1)
    #: A name is a safe (ungated) alias only if this share of its hits carries context.
    ai_model_safe_min_context_share: float = Field(default=0.5, ge=0.0, le=1.0)
    #: A plain word ("Aurora") with more hits than this is never a safe alias.
    ai_model_plain_word_max_safe_hits: int = Field(default=20_000, ge=0)
    #: An exclude is kept only if it removes at most this share of the precise matches.
    ai_model_exclude_max_anchor_loss: float = Field(default=0.03, ge=0.0, le=1.0)
    #: Excludes derived from abstract false positives: at most this many per iteration,
    #: each removing at most this share of the current result.
    ai_model_refine_max_excludes: int = Field(default=3, ge=0, le=10)
    ai_model_refine_exclude_max_share: float = Field(default=0.02, ge=0.0, le=1.0)
    refine_max_new_aliases: int = Field(default=3, ge=0, le=10)
    #: Min false-positive papers an exclude must occur in (title or full-text snippet).
    refine_min_exclude_papers: int = Field(default=2, ge=1)
    #: Max exclude candidates whose impact is checked in Dimensions per refine pass.
    refine_max_exclude_checks: int = Field(default=5, ge=0, le=20)
    #: Dimensions calls kept free for the next gate evaluation; impact checks stop here.
    refine_impact_budget_reserve: int = Field(default=14, ge=0)
    #: Min share of matching papers whose quote/snippets carry a new flag term.
    refine_flag_min_share: float = Field(default=0.2, ge=0.0, le=1.0)
    # A new flag term must be a proper name (agency, program, product): any lower-case
    # word other than a joiner ("state", "administrative data", "state UI") rejects it
    # outright, and any other flag with more Dimensions full-data hits than
    # optimize_flag_max_hits (or an unknown count) is rejected. At most
    # refine_max_flag_checks counts per pass.
    refine_max_flag_checks: int = Field(default=5, ge=0, le=20)
    #: Min false positives in one field of research before it is reported as a pattern.
    refine_category_min_fp_papers: int = Field(default=3, ge=1)

    @field_validator("lmstudio_base_url", "openrouter_base_url")
    @classmethod
    def _normalize_lmstudio_base_url(cls, v: str) -> str:
        """Anthropic SDK expects host root without /v1 (it appends /v1/messages)."""
        base = v.rstrip("/")
        if base.endswith("/v1"):
            return base[:-3]
        return base

    def llm_endpoint(self) -> LLMEndpoint:
        """Resolve provider + API style into a base URL, model and key.

        Raises ``ValueError`` when the selected provider's base URL or model is
        empty: openllm and openai_compatible have no default model (nor, for
        openai_compatible, base URL), and for the other providers an env var
        set to an empty value overrides the default.
        """
        provider = self.llm_provider
        style = self.llm_api_style or DEFAULT_LLM_API_STYLE[provider]
        api_key = self.llm_api_key
        if provider == "lmstudio":
            root, model = self.lmstudio_base_url, self.lmstudio_model
        elif provider == "openrouter":
            root, model = self.openrouter_base_url, self.openrouter_model
            api_key = self.openrouter_api_key or self.llm_api_key
        elif provider == "ollama":
            root, model = self.ollama_base_url, self.ollama_model
        elif provider == "openllm":
            root, model = self.openllm_base_url, self.openllm_model
            api_key = self.openllm_api_key or self.llm_api_key
        else:
            root, model = self.openai_compatible_base_url, self.openai_compatible_model
            api_key = self.openai_compatible_api_key or self.llm_api_key

        env = provider.upper()
        if not root.strip():
            raise ValueError(f"DATASET_AGENT_{env}_BASE_URL must be set for provider {provider}")
        if not model.strip():
            raise ValueError(f"DATASET_AGENT_{env}_MODEL must be set for provider {provider}")

        if provider in ("openllm", "openai_compatible"):
            # Configured as the OpenAI API root; used as given.
            base_url = root.rstrip("/") if style == "openai" else _without_v1(root)
        else:
            # Configured as the host root (the Anthropic SDK appends /v1/messages).
            base_url = _with_v1(root) if style == "openai" else root
        return LLMEndpoint(
            provider=provider,
            api_style=style,
            base_url=base_url,
            model=model,
            api_key=api_key,
        )

    @field_validator("llm_api_style", mode="before")
    @classmethod
    def _empty_api_style_is_unset(cls, v: object) -> object:
        """``DATASET_AGENT_LLM_API_STYLE=`` (empty) means "use the provider default"."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("fp_sample_size")
    @classmethod
    def _fp_sample_size_bounds(cls, v: int) -> int:
        if v < 1 or v > 10_000:
            raise ValueError("fp_sample_size must be between 1 and 10000")
        return v

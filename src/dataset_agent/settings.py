"""Application settings (environment + defaults)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "lmstudio", "openrouter", "openllm", "openai_compatible"]
LLMApiStyle = Literal["anthropic", "openai"]
#: Header that carries the key on the Anthropic style (the OpenAI style always sends Bearer).
LLMAuthMode = Literal["x-api-key", "bearer"]

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


#: The referent-judge configuration the reported figures were measured with (Qwen thinking
#: recipe), for the AI-model referent question and the dataset scope check (one ask).
MEASURED_REFERENT_JUDGE: dict[str, Any] = {
    "thinking_budget": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
}
MEASURED_SCOPE_VOTES = 1


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
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("DATASET_AGENT_OLLAMA_BASE_URL", "DATASET_AGENT_URL"),
    )
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
    #: Anthropic style only: ``x-api-key`` (default; Sandstone/LiteLLM, LM Studio, Ollama) or
    #: ``bearer`` (``Authorization: Bearer``; a vLLM started with ``--api-key``).
    llm_auth_mode: LLMAuthMode = "x-api-key"
    #: ``reasoning_effort`` sent with every LLM call when set (e.g. ``low``, ``medium``,
    #: ``high``); unset sends nothing. Free string: the gateway or server decides what it accepts.
    llm_reasoning_effort: str | None = None
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
    default_years_end: int = Field(default_factory=lambda: time.gmtime().tm_year)
    default_publication_types: list[str] = Field(default_factory=lambda: ["article"])
    default_filter_us_affiliation: bool = False

    #: "noop" accepts research results without a literature lookup; "dimensions" validates
    #: them against the Dimensions API and needs DATASET_AGENT_DIMENSIONS_API_KEY.
    literature_gate: Literal["noop", "dimensions"] = "noop"
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
    #: Runs only when the dataset title has no Dimensions hits.
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
    #: Excludes guard: when V4 keeps less than ``1 - this`` of V3's results, each exclude
    #: is measured against V3. 1.0 disables the guard.
    optimize_exclude_max_total_share: float = Field(default=0.5, ge=0.0, le=1.0)
    #: An exclude that removes more than this share of V3's results is dropped.
    optimize_exclude_max_term_share: float = Field(default=0.10, ge=0.0, le=1.0)
    #: Max optimization loop iterations (build DSL → gate → refine).
    optimize_max_iterations: int = Field(default=2, ge=1, le=10)

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
    #: This is the pool the top-up batches draw from. The random release sample does not
    #: use it: its per-year probes pull up to the year's count + 50 (at most 1,000) so the
    #: draw covers the whole year (``adapters.release_sample``).
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
    #: also the MEDIUM confidence floor and the early-stop bar of the full-text loop.
    fulltext_recall_min_precision: float = Field(default=0.8, ge=0.0, le=1.0)
    #: Release bar for a dataset the resource map splits (the broad program query): max
    #: share of unrelated papers (not data, program-only or references-only citers; the
    #: judge removes citers and program-only papers downstream). Replaces the precision bar
    #: above for split datasets only; the same minimum sample applies.
    fulltext_split_unrelated_max: float = Field(default=0.20, ge=0.0, le=1.0)
    #: Data floor of the split release bar: min confirmed data share (genuine papers the
    #: scope check labelled ``data`` / papers evaluated; ``unclear`` does not count). Below
    #: it the full-text loop keeps refining, for at most ``optimize_max_iterations`` rounds.
    fulltext_split_data_min: float = Field(default=0.50, ge=0.0, le=1.0)
    #: End state when the rounds run out below the floor. False (default, the owner's
    #: decision): the best iteration is released with a warning (MEDIUM at most;
    #: release_bar_met and data_floor_met false). True: not released (LOW).
    fulltext_split_floor_required: bool = False
    #: Min full-text papers evaluated in a complement probe (and min Dimensions hits added);
    #: also the sample needed for MEDIUM or HIGH confidence and the early-stop bar of the
    #: full-text loop.
    fulltext_recall_min_evidence: int = Field(default=10, ge=1)
    #: Which sample the release decision reads (``adapters.release_sample``).
    #: ``random_split`` (default): a split dataset is decided on a random, year-stratified
    #: sample of the selected query's papers with full text; every other dataset keeps the
    #: top-cited sample of the selected iteration (its figures were measured that way).
    #: ``random_all``: every dataset is decided on the random sample. ``top_cited``: none
    #: is. The refinement loop always probes top-cited papers (the refiner's input). AI
    #: models are not covered: they stay on the top-cited sample in every mode, and the
    #: shadow below is not drawn for them.
    fulltext_release_sampling: Literal["random_split", "random_all", "top_cited"] = "random_split"
    #: Also draw the random sample when the top-cited sample decides, and record it next to
    #: the decision (``release_sample.shadow``) without changing it: measures the difference.
    #: Datasets only; AI models record no shadow.
    fulltext_release_shadow: bool = False
    #: Seed of the random release sample; unset draws a fresh seed per run. The seed used is
    #: always recorded (``release_sample.seed``), so a run can be reproduced.
    fulltext_release_seed: Optional[int] = Field(default=None, ge=1)
    #: Min evaluated papers for a split dataset's release bar. 21 is the smallest n from
    #: which, for every larger n too, a true unrelated share of 0.10 passes the 0.20 limit
    #: with probability >= 0.8 and a true 0.30 fails it with probability >= 0.8 (binomial).
    #: The random sample aims at max(this, fulltext_target_delivered) papers.
    fulltext_split_min_sample: int = Field(default=21, ge=1, le=500)
    #: Analyze batches per year of the random release sample (each sends about twice the
    #: papers the year still needs, within fulltext_batch_size); per month slice when the
    #: year is sliced.
    fulltext_release_max_batches_per_year: int = Field(default=3, ge=1, le=20)
    #: Month-count Dimensions calls the random release sample may spend on slicing its
    #: large years by month (twelve per large year: one count per month, so the year's
    #: quota is allocated across the months and each month is probed with an inclusive
    #: ``date_range`` window under its own count + 50 cap). Never the optimizer's
    #: ``optimize_max_dimensions_calls``: this budget is the release sample's own. A
    #: budget spent before a year's twelve counts are complete leaves that year on its
    #: whole-year probe and the record marks the degraded coverage. 0: large years are
    #: never sliced. (All of Us' large years need about 48 calls.)
    fulltext_release_month_count_calls: int = Field(default=60, ge=0)
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
    #: Paper-title candidates per run (name-bearing first; the rest get a decision).
    ai_model_max_paper_titles: int = Field(default=6, ge=0, le=20)
    #: Name-bearing titles taken from the top-cited hits of a safe coined base name
    #: (each passes the model-aware check before it anchors), and how many hits are read.
    ai_model_top_cited_titles: int = Field(default=2, ge=0, le=10)
    ai_model_top_cited_sample: int = Field(default=20, ge=0, le=200)
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
    #: The model's paper title may be one of the flags of an own name that is gated or
    #: re-gated: an anchor title that names the model, or a card title passing the
    #: name-share check above. It never gates a name that is safe.
    ai_model_title_flag_enabled: bool = True
    #: A joined version of a coined name is also searched in its other spelling
    #: ("FourCastNet2" / "FourCastNetv2").
    ai_model_version_siblings: bool = True
    #: AI models: the default window ends next year (papers already dated next year).
    ai_model_window_next_year: bool = True
    #: Full-text referent check (AI models): votes per paper, and thinking for that
    #: question only (0 = off; needs the Anthropic API style). Every other LLM step
    #: keeps the main agent's settings.
    #: Resource map (program versus data): one LLM step plus a deterministic split rule;
    #: on by default (owner decision 2026-11-14). The scope check runs only with a map.
    resource_map_enabled: bool = True
    resource_map_min_cooccurrence: int = 1
    #: The release year moves the window start only when at most this share of
    #: program-and-data papers comes before it.
    resource_map_max_share_before_release: float = 0.05
    #: A data resource name is specific (it may stand alone as a safe alias, subject to its
    #: probe) when at least this share of its Dimensions hits also carry a program name.
    resource_map_specific_min_share: float = 0.5
    #: For a split dataset only: look for more data resource names in the program's data
    #: documentation (web) and in the passages of papers judged "used" (after the first
    #: probe). Every candidate passes the same deterministic checks, then the probe.
    resource_map_discovery: bool = True
    #: Literature discovery: a name must appear in passages of at least this many used papers.
    resource_map_literature_min_papers: int = 2
    #: Asks per paper in the scope check (thinking through the referent settings).
    scope_votes: int = Field(default=1, ge=1, le=9)
    referent_votes: int = Field(default=3, ge=1, le=9)
    #: S4: after the abstract rule, probe the shipped AI-model seed on full text; with at
    #: least ai_model_fulltext_min_papers judged, corrected precision decides release both
    #: ways at ai_model_fulltext_min_precision (one re-gate and re-probe below it).
    ai_model_fulltext_enabled: bool = True
    ai_model_fulltext_min_papers: int = Field(default=10, ge=1)
    ai_model_fulltext_min_precision: float = Field(default=0.80, ge=0.0, le=1.0)
    #: On by default at the values the reported AI-model figures were measured with (Qwen
    #: thinking recipe: budget 4096, temperature 0.6, top_p 0.95, top_k 20); 0 = off.
    referent_thinking_budget: int = Field(default=4096, ge=0)
    referent_temperature: float | None = 0.6
    referent_top_p: float | None = 0.95
    referent_top_k: int | None = 20
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
    #: A coined name ("FourCastNet") is gated only when fewer than this share of its hits
    #: cite the model's papers; a partial share above it is not evidence against the name
    #: (a versioned card's papers are cited by only part of the family's literature).
    ai_model_coined_min_title_share: float = Field(default=0.05, ge=0.0, le=1.0)
    #: Reserve flags for a coined base name that is safe on weak evidence: at most this
    #: many candidates are measured, stopping once this many pass the flag bar.
    ai_model_reserve_flag_max_candidates: int = Field(default=10, ge=0, le=30)
    ai_model_reserve_flag_max_passing: int = Field(default=3, ge=0, le=10)
    #: A plain word ("Aurora") with more hits than this is never a safe alias.
    ai_model_plain_word_max_safe_hits: int = Field(default=20_000, ge=0)
    #: An exclude is kept only if it removes at most this share of the precise matches.
    ai_model_exclude_max_anchor_loss: float = Field(default=0.03, ge=0.0, le=1.0)
    #: Excludes derived from abstract false positives: at most this many per iteration,
    #: each removing at most this share of the current result.
    ai_model_refine_max_excludes: int = Field(default=3, ge=0, le=10)
    ai_model_refine_exclude_max_share: float = Field(default=0.02, ge=0.0, le=1.0)
    #: When a false-positive name is in the reference set, its excludes are validated
    #: against the reference without that name if it holds at least this many papers
    #: (then the result share cap above does not apply).
    ai_model_refine_reference_min_papers: int = Field(default=10, ge=1)
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

    def referent_judge_config(self) -> dict[str, Any]:
        """The configuration the referent agent (``bootstrap.build_referent_agent``) runs
        with: ``agent`` is ``referent`` (thinking on) or ``main`` (budget 0, or the
        fallback when the API style is not anthropic: thinking and sampling dropped).
        ``measured_configuration`` says whether it is the measured one."""
        style = self.llm_api_style or DEFAULT_LLM_API_STYLE[self.llm_provider]
        budget = int(self.referent_thinking_budget or 0)
        fallback = bool(budget) and style != "anthropic"
        if budget and not fallback:
            cfg: dict[str, Any] = {
                "agent": "referent", "api_style": style, "fallback": False,
                "thinking_budget": budget,
                "temperature": 0.6 if self.referent_temperature is None else self.referent_temperature,
                "top_p": self.referent_top_p, "top_k": self.referent_top_k,
            }
        else:
            cfg = {"agent": "main", "api_style": style, "fallback": fallback,
                   "thinking_budget": 0, "temperature": None, "top_p": None, "top_k": None}
        cfg["measured_configuration"] = cfg["agent"] == "referent" and all(
            cfg[k] == v for k, v in MEASURED_REFERENT_JUDGE.items())
        return cfg

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

    @field_validator("llm_reasoning_effort", mode="before")
    @classmethod
    def _empty_reasoning_effort_is_unset(cls, v: object) -> object:
        """Empty means unset; the value is stripped and lowercased."""
        if isinstance(v, str):
            v = v.strip().lower()
            return v or None
        return v

    @field_validator("fp_sample_size")
    @classmethod
    def _fp_sample_size_bounds(cls, v: int) -> int:
        if v < 1 or v > 10_000:
            raise ValueError("fp_sample_size must be between 1 and 10000")
        return v

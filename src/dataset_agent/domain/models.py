"""Pydantic models: canonical JSON record and research request."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from dataset_agent.domain.proposals import IterationMetrics, QueryProposal


class AccessType(str, Enum):
    OPEN = "Open"
    RESTRICTED = "Restricted"
    UNKNOWN = "Unknown"


class Group(BaseModel):
    name: str


class YearsRange(BaseModel):
    start_year: int
    end_year: int


class LiteratureValidation(BaseModel):
    """Result of literature validation gate."""

    passed: bool
    indicator: float  # 0.0 - 1.0
    attempts: int
    detail: dict = Field(default_factory=dict)


class TermsEvaluation(BaseModel):
    """LLM evaluation of dataset_names and flag_terms quality."""

    is_effective: bool
    dataset_names_score: int  # 0-10
    flag_terms_score: int  # 0-10
    issues: list[str] = Field(default_factory=list)
    suggested_dataset_names: list[str] = Field(default_factory=list)
    suggested_flag_terms: list[str] = Field(default_factory=list)
    suggested_exclude_terms: list[str] = Field(
        default_factory=list,
        description="Terms/phrases to avoid in a follow-up literature search when off-domain noise appears",
    )
    reasoning: str = ""


class ResearchRequest(BaseModel):
    """Minimal input: dataset name; other parameters come from Settings."""

    dataset_name: str = Field(..., min_length=1, description="Dataset name to research")
    webhook_url: Optional[str] = Field(
        default=None,
        description="Optional URL that receives a POST with the JSON when done (background API)",
    )
    sample_size: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of publications to sample for validation (1-1000)",
    )
    llm_batch_size: int = Field(
        default=25,
        ge=0,
        le=1000,
        description="Maximum lexical candidates sent to LLM validation (0-1000)",
    )
    dataset_url: Optional[str] = Field(
        default=None,
        description="Optional official dataset URL (used for alias discovery / optimize pipeline)",
    )

    @field_validator("dataset_name", mode="before")
    @classmethod
    def _strip_dataset_name(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    @field_validator("dataset_url", mode="before")
    @classmethod
    def _strip_dataset_url(cls, v: object) -> object:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s or None
        return v


class ValidationRequest(BaseModel):
    """Direct validation request with pre-defined terms."""

    main_dataset_name: str = Field(..., min_length=1, description="Main dataset name")
    dataset_names: list[str] = Field(default_factory=list, description="Alternative names/aliases")
    flag_terms: list[str] = Field(default_factory=list, description="Organizations/flag terms")
    description: Optional[str] = Field(default=None, description="Optional dataset description")
    sample_size: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of publications to sample for validation (1-1000)",
    )
    llm_batch_size: int = Field(
        default=25,
        ge=0,
        le=1000,
        description="Maximum lexical candidates sent to LLM validation (0-1000)",
    )
    exclude_terms: list[str] = Field(
        default_factory=list,
        description="Terms to exclude from Dimensions query / filtering on retry",
    )


class ConfidenceLevel(str, Enum):
    """Aggregate confidence for the selected Dimensions query variant."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OptimizeRequest(BaseModel):
    """Input for the Query Optimization pipeline (POST /optimize)."""

    dataset_name: str = Field(..., min_length=1)
    dataset_url: Optional[str] = Field(
        default=None,
        description="Official dataset page URL (strong signal for aliases and heuristics)",
    )
    dataset_names: list[str] = Field(
        default_factory=list,
        description="Optional override: seed aliases instead of running full research",
    )
    flag_terms: list[str] = Field(
        default_factory=list,
        description="Optional override: seed flag terms",
    )
    exclude_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Optional seed exclusions before FP analysis. These terms are candidates for "
            "the resulting `exclusion_terms` and can be forwarded to `POST /validate` as "
            "`exclude_terms` in follow-up validation runs."
        ),
    )
    entity_type: Literal["dataset", "ai_model"] = Field(
        default="dataset",
        description=(
            "What `dataset_name` names. With `ai_model` it is an AI model (family) name and "
            "the model research path runs: Hugging Face / arXiv evidence, qualified names, "
            "context flags measured in Dimensions. `dataset_names`, `flag_terms` and "
            "`exclude_terms` are then extra candidates, measured like every other term."
        ),
    )
    hugging_face_id: Optional[str] = Field(
        default=None, description="AI models only: Hugging Face model id, e.g. `microsoft/aurora`."
    )
    model_family: Optional[str] = Field(
        default=None, description="AI models only: family name; defaults to `dataset_name`."
    )
    years_start: Optional[int] = Field(
        default=None, description="AI models only: first publication year for the seed."
    )
    years_end: Optional[int] = Field(
        default=None, description="AI models only: last publication year for the seed."
    )

    model_config = {"protected_namespaces": ()}

    @field_validator("hugging_face_id", "model_family", mode="before")
    @classmethod
    def _strip_opt_text(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip() or None
        return v

    @field_validator("dataset_name", mode="before")
    @classmethod
    def _strip_opt_dataset_name(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    @field_validator("dataset_url", mode="before")
    @classmethod
    def _strip_opt_dataset_url(cls, v: object) -> object:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s or None
        return v


class QueryOptimizationAliases(BaseModel):
    """Alias buckets after risk classification."""

    safe: list[str] = Field(default_factory=list)
    risky: list[str] = Field(default_factory=list)


class AliasCountEntry(BaseModel):
    """Per-alias publication count and risk label."""

    alias: str
    count: int = Field(ge=0)
    risk: Literal["safe", "risky"]


class VariantTestedEntry(BaseModel):
    """One tested query variant (V1/V3/V4, etc.)."""

    label: str
    for_clause: str
    expected_count: Optional[int] = Field(default=None, ge=0)
    fp_rate_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    top_titles: list[str] = Field(default_factory=list)


class TermDecision(BaseModel):
    """Why one candidate term was kept or rejected (AI-model runs)."""

    term: str
    #: ``abstract_fp``: one paper the abstract check judged another meaning (``stats``
    #: holds publication_id, year, title, matched_alias and iteration).
    kind: Literal["alias", "paper_title", "flag", "exclude", "abstract_fp"]
    decision: Literal[
        "safe", "gated", "flag", "context", "exclude", "anchor", "rejected", "false_positive"
    ]
    count: Optional[int] = Field(default=None, ge=0, description="Dimensions full-data hits")
    reason: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)


class QueryOptimizationRecord(BaseModel):
    """Output of the Query Optimization Agent (see docs/spec-gap-analysis.md §2.1)."""

    schema_version: int = Field(default=1, ge=1, description="JSON contract version for clients")
    success: bool = Field(default=True, description="False if the pipeline aborted or degraded")
    failed_phase: Optional[str] = Field(
        default=None,
        description="Last completed phase name when success is false (e.g. phase1_alias_counting)",
    )
    errors: list[str] = Field(default_factory=list, description="Human-readable error messages")

    selected_variant: Optional[str] = None
    for_clause: Optional[str] = None
    dsl_query: Optional[str] = None
    expected_count: Optional[int] = Field(default=None, ge=0)
    fp_rate_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    confidence: Optional[ConfidenceLevel] = None

    aliases: QueryOptimizationAliases = Field(default_factory=QueryOptimizationAliases)
    alias_counts: list[AliasCountEntry] = Field(default_factory=list)
    flag_terms: list[str] = Field(default_factory=list)
    exclusion_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Suggested NOT terms for follow-up filtering. This list is not persisted "
            "automatically; pass it to `POST /validate` as `exclude_terms` when you want "
            "to validate with these exclusions."
        ),
    )
    all_variants_tested: list[VariantTestedEntry] = Field(default_factory=list)
    fulltext_iterations: list[IterationMetrics] = Field(
        default_factory=list, description="Per-iteration full-text metrics (refinement enabled)"
    )
    proposal: Optional[QueryProposal] = Field(
        default=None, description="Full-text-backed query proposal awaiting human approval"
    )
    entity_type: Literal["dataset", "ai_model"] = "dataset"
    term_decisions: list[TermDecision] = Field(
        default_factory=list,
        description="AI models: every candidate term with its counts and the decision taken",
    )
    ai_model_seed: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "AI models: `POST /ai-models/start` body for the extraction pipeline. Its flag "
            "terms apply to every alias there, so they carry the safe names too and the "
            "seed finds what the selected query finds."
        ),
    )
    abstract_fp_count: Optional[int] = Field(
        default=None, ge=0, description="AI models: abstracts judged another meaning (k)"
    )
    abstract_n_judged: Optional[int] = Field(
        default=None, ge=0, description="AI models: abstracts the LLM judged (n)"
    )
    abstract_n_counted_without_judgement: Optional[int] = Field(
        default=None, ge=0,
        description="AI models: papers counted genuine with no alias in title or abstract",
    )
    abstract_fp_wilson_95: Optional[list[float]] = Field(
        default=None, description="AI models: Wilson 95% interval of k / n (0-1)"
    )
    abstract_coverage_pct: Optional[float] = Field(
        default=None,
        description="AI models: share of the pulled papers that name the model in title or abstract",
    )
    ai_model_seed_clause: Optional[str] = Field(
        default=None, description="AI models: the `for` clause the extraction pipeline builds from the seed"
    )
    ai_model_seed_count: Optional[int] = Field(
        default=None, ge=0, description="AI models: hits of the seed clause inside the year window"
    )
    ai_model_query_count: Optional[int] = Field(
        default=None, ge=0,
        description=(
            "AI models: hits of the selected `for_clause` inside the year window of "
            "`dsl_query`, next to `ai_model_seed_count`"
        ),
    )
    ai_model_judge_context: Optional[str] = Field(
        default=None,
        description="AI models: the model description given to the abstract-check judge",
    )
    ai_model_query_count_all_years: Optional[int] = Field(
        default=None, ge=0, description="AI models: all-years hits of the selected `for_clause`"
    )
    notes: str = ""


class DatasetRecord(BaseModel):
    """Canonical JSON record (LOGICA_DO_PROJETO.md)."""

    engine: str
    group: Group
    main_dataset_name: str
    home_url: Optional[str] = None
    description: str = ""
    dataset_names: list[str] = Field(default_factory=list)
    flag_terms: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    years_range: YearsRange
    filter_us_affiliation: bool = False
    publication_types: list[str] = Field(default_factory=lambda: ["article"])
    access_type: str = AccessType.UNKNOWN.value
    data_url: Optional[str] = None
    schema_url: Optional[str] = None
    documentation_url: Optional[str] = None
    official_name: str = ""
    relationship_type: str = "official_name"
    official_name_reasoning: str = ""
    webhook_url: Optional[str] = None
    literature_validation: Optional[LiteratureValidation] = None
    terms_evaluation: Optional[TermsEvaluation] = None
    publications_total: Optional[int] = None
    dsl_query: Optional[str] = Field(
        default=None,
        description="Dimensions DSL used for literature/publication count lookup",
    )
    alias_collisions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Collision domains from web disambiguation → candidate exclude terms",
    )
    query_optimization: Optional[QueryOptimizationRecord] = None
    retry_validation: Optional[ValidationRequest] = Field(
        default=None,
        description="Suggested POST /validate body for a follow-up run (set by /validate only)",
    )
    links: Optional[dict[str, Any]] = Field(
        default=None,
        description="HATEOAS-style links (e.g. retry -> POST /validate)",
    )

    def model_dump_json_pretty(self) -> str:
        return self.model_dump_json(indent=2)


def build_record_from_pipeline(
    request: ResearchRequest,
    *,
    description: str,
    dataset_names: list[str],
    flag_terms: list[str],
    access_type: str,
    data_url: Optional[str],
    schema_url: Optional[str],
    documentation_url: Optional[str],
    home_url: Optional[str] = None,
    defaults: Any,
) -> DatasetRecord:
    """Merge pipeline outputs with request + Settings defaults (defaults must expose fields)."""
    years = YearsRange(start_year=defaults.default_years_start, end_year=defaults.default_years_end)
    group = Group(name=defaults.default_group_name)
    pub_types = list(defaults.default_publication_types)
    filt_us = bool(defaults.default_filter_us_affiliation)

    main = request.dataset_name
    reasoning = (
        "Heuristic v1: official_name matches main_dataset_name until a dedicated "
        "resolution step exists."
    )

    eng = defaults.default_engine

    return DatasetRecord(
        engine=eng,
        group=group,
        main_dataset_name=main,
        home_url=home_url,
        description=description,
        dataset_names=dataset_names,
        flag_terms=flag_terms,
        exclude_terms=[],
        years_range=years,
        filter_us_affiliation=filt_us,
        publication_types=pub_types,
        access_type=access_type,
        data_url=data_url,
        schema_url=schema_url,
        documentation_url=documentation_url,
        official_name=main,
        relationship_type="official_name",
        official_name_reasoning=reasoning,
        webhook_url=request.webhook_url,
        alias_collisions={},
        query_optimization=None,
    )

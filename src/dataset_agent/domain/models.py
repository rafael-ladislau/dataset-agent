"""Pydantic models: canonical JSON record and research request."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("dataset_name", mode="before")
    @classmethod
    def _strip_dataset_name(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v


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
    )

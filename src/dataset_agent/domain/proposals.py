"""Models for full-text-backed query proposals (propose -> approve -> submit)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceQuote(BaseModel):
    publication_id: str = ""
    doi: Optional[str] = None
    title: Optional[str] = None
    quote: str = ""
    #: ``used`` or ``mentioned`` (both correct matches) or ``not_mentioned`` (false positive).
    verdict: Literal["used", "mentioned", "not_mentioned"] = "used"


class IterationMetrics(BaseModel):
    """Metrics for one probed candidate. ``None`` means not measurable."""

    iteration: int
    for_clause: str = ""
    n_publications: int = 0
    n_fulltext_delivered: int = 0
    n_evaluated: int = 0
    fulltext_precision: Optional[float] = Field(
        default=None, description="match precision: papers mentioning the dataset / evaluated (0-1)"
    )
    usage_rate: Optional[float] = Field(default=None, description="used / mentioned-or-used (0-1)")
    mention_rate: Optional[float] = Field(default=None, description="(used|mentioned) / evaluated (0-1)")
    used_rate: Optional[float] = Field(default=None, description="used / evaluated (0-1)")
    oa_coverage: Optional[float] = Field(default=None, description="delivered / attempted (0-1)")
    fulltext_target: Optional[int] = Field(default=None, description="delivered full texts aimed for")
    fulltext_batches: int = Field(default=0, description="analyze batches sent")
    fulltext_stop_reason: Optional[str] = Field(
        default=None, description="target_met | pool_exhausted | max_batches"
    )
    fulltext_budget_exhausted: bool = Field(
        default=False, description="OpenAlex refused at least one file for budget"
    )
    abstract_fp_rate_pct: Optional[float] = Field(default=None, description="secondary signal")
    papers: list[tuple[str, str]] = Field(
        default_factory=list,
        description="(publication id, used | mentioned | not_mentioned) for every evaluated paper",
    )
    cited_only_count: Optional[int] = Field(
        default=None,
        description="AI models: genuine papers naming the model only in the reference list "
        "(counted as matches); None for datasets",
    )
    cited_rate: Optional[float] = Field(
        default=None, description="AI models: cited only / evaluated (0-1); None for datasets"
    )
    paper_details: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="AI models: one entry per evaluated paper (id, title, verdict, location, "
        "cited_only, matched_name); None for datasets",
    )
    error: Optional[str] = None


class TermDiff(BaseModel):
    kind: Literal["alias", "flag", "exclude"]
    term: str
    #: keep | add | drop | demote (safe->risky)
    action: Literal["keep", "add", "drop", "demote"]
    #: verified (full-text evidence), unverified (no evidence, never changed),
    #: suspect (abstract-only signal, never changed).
    status: Literal["verified", "unverified", "suspect"]
    reason: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceQuote] = Field(default_factory=list)


class SeedBuckets(BaseModel):
    main_dataset_name: str
    safe: list[str] = Field(default_factory=list)
    risky: list[str] = Field(default_factory=list)
    flag_terms: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    raw_clause: Optional[str] = None
    for_clause: str = ""
    seed: dict[str, Any] = Field(default_factory=dict, description="/pipelines/start body")


class QueryProposal(BaseModel):
    id: str
    status: Literal["pending", "approved", "submitted"] = "pending"
    dataset_name: str
    created_at: str = Field(default_factory=utc_now_iso)
    initial: SeedBuckets
    proposed: SeedBuckets
    diffs: list[TermDiff] = Field(default_factory=list)
    iterations: list[IterationMetrics] = Field(default_factory=list)
    #: Ready ``/pipelines/start`` payload (probe=false).
    start_payload: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None
    submitted_at: Optional[str] = None
    submit_target: Optional[str] = None
    submit_result: Optional[dict[str, Any]] = None

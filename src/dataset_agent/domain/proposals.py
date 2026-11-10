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
    #: ``used`` (genuine), ``mentioned`` (mentioned, not used) or ``not_mentioned``.
    verdict: Literal["used", "mentioned", "not_mentioned"] = "used"


class IterationMetrics(BaseModel):
    """Metrics for one probed candidate. ``None`` means not measurable."""

    iteration: int
    for_clause: str = ""
    n_publications: int = 0
    n_fulltext_delivered: int = 0
    n_evaluated: int = 0
    fulltext_precision: Optional[float] = Field(default=None, description="used / evaluated (0-1)")
    usage_rate: Optional[float] = Field(default=None, description="used / mentioned-or-used (0-1)")
    mention_rate: Optional[float] = Field(default=None, description="(used|mentioned) / evaluated (0-1)")
    oa_coverage: Optional[float] = Field(default=None, description="delivered / sampled (0-1)")
    abstract_fp_rate_pct: Optional[float] = Field(default=None, description="secondary signal")
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

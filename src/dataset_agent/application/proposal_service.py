"""Build, approve and submit full-text-backed query proposals."""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from dataset_agent.adapters.extraction_client import ExtractionClient
from dataset_agent.adapters.seed_mapping import QueryCandidate, candidate_clause, candidate_to_seed
from dataset_agent.domain.proposals import (
    IterationMetrics,
    QueryProposal,
    SeedBuckets,
    TermDiff,
    utc_now_iso,
)
from dataset_agent.settings import Settings


class ProposalError(Exception):
    """Base class; ``status_code`` maps to the HTTP response."""

    status_code = 400


class ProposalNotFound(ProposalError):
    status_code = 404


class ProposalStateError(ProposalError):
    status_code = 409


class ProposalForbidden(ProposalError):
    status_code = 403


class ProposalRepo(Protocol):
    def save(self, proposal: QueryProposal) -> None: ...
    def get(self, proposal_id: str) -> QueryProposal | None: ...


def _buckets(c: QueryCandidate, settings: Settings, **seed_kw: Any) -> SeedBuckets:
    return SeedBuckets(
        main_dataset_name=c.main_dataset_name,
        safe=list(c.safe),
        risky=list(c.risky),
        flag_terms=list(c.flag_terms),
        exclude_terms=list(c.exclude_terms),
        raw_clause=c.raw_clause,
        for_clause=candidate_clause(c),
        seed=candidate_to_seed(c, **seed_kw),
    )


def merge_diffs(per_iteration: list[list[TermDiff]]) -> list[TermDiff]:
    """Collapse per-iteration diffs: a real change beats ``keep``; later beats earlier."""
    merged: dict[tuple[str, str], TermDiff] = {}
    for diffs in per_iteration:
        for d in diffs:
            key = (d.kind, d.term.lower())
            prev = merged.get(key)
            if prev is None or d.action != "keep" or prev.action == "keep":
                merged[key] = d
    return list(merged.values())


def build_proposal(
    *,
    dataset_name: str,
    initial: QueryCandidate,
    proposed: QueryCandidate,
    diffs: list[TermDiff],
    iterations: list[IterationMetrics],
    settings: Settings,
    notes: list[str] | None = None,
    seed_kw: dict[str, Any] | None = None,
) -> QueryProposal:
    kw = dict(seed_kw or {})
    kw.setdefault("group_name", settings.default_group_name)
    kw.setdefault("years", (settings.default_years_start, settings.default_years_end))
    kw.setdefault("publication_types", list(settings.default_publication_types))
    kw.setdefault("filter_us_affiliation", settings.default_filter_us_affiliation)
    prop_buckets = _buckets(proposed, settings, **kw)
    return QueryProposal(
        id=uuid.uuid4().hex,
        dataset_name=dataset_name,
        initial=_buckets(initial, settings, **kw),
        proposed=prop_buckets,
        diffs=diffs,
        iterations=iterations,
        start_payload={**prop_buckets.seed, "probe": False},
        notes=list(notes or []),
    )


class ProposalService:
    def __init__(
        self,
        repo: ProposalRepo,
        settings: Settings,
        client_factory=None,
    ) -> None:
        self._repo = repo
        self._settings = settings
        self._client_factory = client_factory or (
            lambda target: ExtractionClient.for_target(settings, target)
        )

    def get(self, proposal_id: str) -> QueryProposal:
        p = self._repo.get(proposal_id)
        if p is None:
            raise ProposalNotFound(proposal_id)
        return p

    def approve(self, proposal_id: str, approved_by: str | None = None) -> QueryProposal:
        p = self.get(proposal_id)
        if p.status == "submitted":
            raise ProposalStateError("proposal already submitted")
        if p.status == "pending":
            p.status = "approved"
            p.approved_at = utc_now_iso()
            p.approved_by = approved_by
            self._repo.save(p)
        return p

    def submit(self, proposal_id: str) -> QueryProposal:
        p = self.get(proposal_id)
        if p.status == "submitted":
            raise ProposalStateError("proposal already submitted")
        if p.status != "approved":
            raise ProposalStateError("proposal must be approved before submit")
        target = self._settings.submit_target
        if target == "prod" and p.status != "approved":  # defence in depth
            raise ProposalForbidden("prod submit requires an approved proposal")
        payload = {**p.start_payload, "probe": False}
        client = self._client_factory(target)
        try:
            result = client.submit(payload)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        p.status = "submitted"
        p.submitted_at = utc_now_iso()
        p.submit_target = target
        p.submit_result = result if isinstance(result, dict) else {"result": result}
        self._repo.save(p)
        return p

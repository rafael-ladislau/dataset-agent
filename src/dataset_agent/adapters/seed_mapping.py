"""Map a query candidate onto the extraction pipeline's ``/pipelines/start`` seed.

Mapping rules (extraction contract v1):

* safe aliases   -> ``standalone_names`` (match without flags)
* risky aliases  -> ``dataset_names`` (+ ``flag_terms``)
* excludes       -> ``exclude_terms``
* shapes the structured fields cannot express (explicit raw clause, or only safe
  aliases, which would leave ``dataset_names`` empty) -> raw ``dimensions_query``
  clause. The clause must contain every alias in ``dataset_names`` +
  ``standalone_names`` (the extraction API rejects it otherwise, HTTP 422).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from dataset_agent.adapters.query_builder import (
    _clean,
    _or_group,
    build_for_clause,
    drop_flags_equal_to_aliases,
)
from dataset_agent.adapters.text_processing import dedupe_strings_ci_preserve_order


@dataclass
class QueryCandidate:
    """One candidate Dimensions query, expressed as term buckets."""

    main_dataset_name: str
    safe: list[str] = field(default_factory=list)
    risky: list[str] = field(default_factory=list)
    flag_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    #: Explicit raw ``for "..."`` clause for shapes the buckets cannot express.
    raw_clause: str | None = None
    description: str = ""

    def all_aliases(self) -> list[str]:
        return dedupe_strings_ci_preserve_order(list(self.safe) + list(self.risky))


def candidate_clause(c: QueryCandidate) -> str:
    """The ``for`` clause equivalent to the candidate (raw clause wins)."""
    if c.raw_clause:
        return c.raw_clause
    return build_for_clause(
        safe=c.safe,
        variant="V4",
        risky=c.risky,
        flag_terms=c.flag_terms,
        exclusion_terms=c.exclude_terms,
    )


# Characters Dimensions' lexer treats as operators inside a quoted phrase. A raw
# ``dimensions_query`` cannot carry backslash escapes (the server rejects a stray
# backslash and never escapes raw clauses), so terms containing any of them go
# through the structured seed fields, which the server escapes when it builds the clause.
_DSL_SPECIAL_TERM_CHARS = frozenset('&|!(){}[]^~*?:/+\\')


def _has_dsl_special_terms(c: QueryCandidate) -> bool:
    terms = (*c.all_aliases(), *c.safe, *c.risky, *c.flag_terms, *c.exclude_terms)
    return any(ch in _DSL_SPECIAL_TERM_CHARS for t in terms for ch in t)


def seed_clause(c: QueryCandidate) -> str:
    """Raw ``dimensions_query`` clause using the extraction server's quoting convention.

    Each alias becomes ``\\"alias\\"``; only a literal ``"`` inside an alias is
    escaped. No other DSL escaping is applied (unlike the agent's own queries).
    """
    if c.raw_clause:
        return c.raw_clause
    return build_for_clause(
        safe=c.safe,
        variant="V4",
        risky=c.risky,
        flag_terms=c.flag_terms,
        exclusion_terms=c.exclude_terms,
        server_quoting=True,
    )


def alias_coverage_missing(clause: str, aliases: list[str]) -> list[str]:
    """Aliases (case-insensitive) that the server would not find in *clause*.

    Mirrors the server: bare double quotes in the clause are escaped, then each
    alias must appear verbatim (lower-cased) as a substring. Backslashes are
    not stripped.
    """
    hay = re.sub(r'(?<!\\)"', '\\\\"', (clause or "").strip()).lower()
    return [a for a in aliases if a and a.lower() not in hay]


def candidate_to_seed(
    c: QueryCandidate,
    *,
    group_name: str = "default",
    years: tuple[int, int] = (2015, 2025),
    publication_types: list[str] | None = None,
    filter_us_affiliation: bool = False,
    dimensions_limit: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``/pipelines/start`` body (without ``probe`` fields)."""
    use_raw = bool(c.raw_clause) or (
        bool(c.safe) and not c.risky and not _has_dsl_special_terms(c)
    )
    seed: dict[str, Any] = {
        "seed_type": "dataset",
        "engine": "dimensions",
        "group": {"name": group_name},
        "main_dataset_name": c.main_dataset_name,
        "description": c.description or c.main_dataset_name,
        "dataset_names": list(c.risky),
        "standalone_names": list(c.safe),
        "flag_terms": drop_flags_equal_to_aliases(c.flag_terms, c.risky),
        "exclude_terms": list(c.exclude_terms),
        "webhook_url": None,
        "years_range": {"start_year": years[0], "end_year": years[1]},
        "filter_us_affiliation": filter_us_affiliation,
        "publication_types": list(publication_types or ["article"]),
        "home_url": None,
        "access_type": None,
        "data_url": None,
        "schema_url": None,
        "documentation_url": None,
    }
    quoted = [
        t
        for t in (*seed["dataset_names"], *seed["standalone_names"], *seed["flag_terms"], *seed["exclude_terms"])
        if '"' in t
    ]
    if quoted:
        # The server rejects a double quote in any seed term (HTTP 422).
        raise ValueError(f"terms must not contain a double quote: {quoted}")
    if dimensions_limit is not None:
        seed["dimensions_limit"] = dimensions_limit
    if use_raw:
        clause = seed_clause(c)
        # With a raw clause the structured buckets only feed validation; keep every
        # alias in dataset_names so downstream matching sees them all.
        seed["dataset_names"] = c.all_aliases()
        missing = alias_coverage_missing(clause, seed["dataset_names"] + seed["standalone_names"])
        if missing:
            raise ValueError(f"raw dimensions_query does not contain aliases: {missing}")
        seed["dimensions_query"] = clause
    else:
        seed["dimensions_query"] = None
    if extra:
        seed.update(extra)
    return seed


def _gated(current: QueryCandidate) -> bool:
    return bool(current.risky and current.flag_terms)


def complement_clause(term: str, current: QueryCandidate, *, server_quoting: bool = False) -> str:
    """``("term") NOT ((<positive body of current>) OR <current excludes>)``.

    The final query is ``(P OR term) NOT E``, so the papers the term adds are exactly
    ``term NOT (P OR E)``. *P* is built like :func:`candidate_clause` (V3 body)."""
    sq = server_quoting
    pos = build_for_clause(
        safe=current.safe, variant="V3", risky=current.risky,
        flag_terms=current.flag_terms, server_quoting=sq,
    )
    neg = [f"({pos})"] if pos else []
    ex = _or_group(_clean(current.exclude_terms), sq)
    if ex:
        neg.append(ex)
    head = _or_group([term], sq)
    return f"{head} NOT ({' OR '.join(neg)})" if neg else head


def complement_candidate(term: str, current: QueryCandidate) -> QueryCandidate:
    """Candidate whose probe returns only papers *term* adds beyond *current*.

    The term is sent as the (only) alias so the extraction validator looks for it.
    Raises ``ValueError`` when the shape cannot be expressed safely."""
    desc = f"recall complement of '{term}' beyond the current query"
    if not _gated(current):
        # No flag-gated group: a plain alias with the whole current query as excludes.
        excl = list(current.safe) + list(current.risky) + list(current.exclude_terms)
        if not current.safe and not current.risky:
            excl += list(current.flag_terms)  # flags alone form the query body
        return QueryCandidate(
            main_dataset_name=current.main_dataset_name,
            safe=[term],
            exclude_terms=dedupe_strings_ci_preserve_order(excl),
            description=desc,
        )
    probe = QueryCandidate(
        main_dataset_name=current.main_dataset_name,
        safe=[term], risky=list(current.risky), flag_terms=list(current.flag_terms),
        exclude_terms=list(current.exclude_terms),
    )
    if _has_dsl_special_terms(probe) or any(
        ch in _DSL_SPECIAL_TERM_CHARS for t in current.safe for ch in t
    ):
        raise ValueError("complement not expressible: flag-gated query with DSL special characters")
    return QueryCandidate(
        main_dataset_name=current.main_dataset_name,
        safe=[term],
        raw_clause=complement_clause(term, current, server_quoting=True),
        description=desc,
    )

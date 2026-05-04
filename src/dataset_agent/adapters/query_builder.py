"""Dimensions ``for`` clause builders (V1–V5) and variant probe execution.

See ``dataset_query_agent_spec.md`` §4 and ``docs/spec-gap-analysis.md`` §2.3.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from dataset_agent.adapters.dimensions_dsl import (
    build_search_publications_dsl,
    inner_quoted_term,
    parse_total_count,
    top_titles_from_result,
)
from dataset_agent.domain.ports import DimensionsDslPort

VariantId = Literal["V1", "V2", "V3", "V4", "V5"]


def _clean(terms: Sequence[str] | None) -> list[str]:
    if not terms:
        return []
    out: list[str] = []
    for t in terms:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if s:
            out.append(s)
    return out


def _q(term: str) -> str:
    """Escaped quoted token as used inside the outer ``for \"...\"`` string."""
    inner = inner_quoted_term(term)
    return f'\\"{inner}\\"'


def _or_group(terms: list[str]) -> str:
    if not terms:
        return ""
    parts = [_q(t) for t in terms]
    return "(" + " OR ".join(parts) + ")"


def build_for_clause(
    *,
    safe: Sequence[str],
    variant: VariantId = "V1",
    risky: Sequence[str] | None = None,
    flag_terms: Sequence[str] | None = None,
    exclusion_terms: Sequence[str] | None = None,
    tier_hybrid_terms: Sequence[str] | None = None,
    tier_hybrid_qualifier: str | None = None,
) -> str:
    """Assemble a boolean ``for`` sub-expression (UPPERCASE ``OR`` / ``AND`` / ``NOT``)."""
    st = _clean(safe)
    rk = _clean(risky)
    fl = _clean(flag_terms)
    ex = _clean(exclusion_terms)
    th = _clean(tier_hybrid_terms)
    tq = tier_hybrid_qualifier.strip() if isinstance(tier_hybrid_qualifier, str) else ""

    if variant == "V1":
        return _or_group(st)

    if variant == "V2":
        return _or_group(st + rk)

    if variant == "V5":
        return _or_group(st)

    # V3 / V4: safe OR hybrid; optional tier-hybrid OR arm; optional NOT exclusions (V4).
    chunks: list[str] = []
    safe_expr = _or_group(st)
    if safe_expr:
        chunks.append(safe_expr)

    if rk and fl:
        hybrid = f"({_or_group(rk)} AND {_or_group(fl)})"
        chunks.append(hybrid)
    elif rk:
        chunks.append(_or_group(rk))
    elif fl and not st:
        chunks.append(_or_group(fl))

    body = " OR ".join(c for c in chunks if c)
    if th and tq:
        tier = f"({_or_group(th)} AND {_or_group([tq])})"
        body = f"{body} OR {tier}" if body else tier

    if variant == "V4" and ex:
        ex_g = _or_group(ex)
        if ex_g:
            body = f"{body} NOT {ex_g}" if body else f"NOT {ex_g}"

    return body


def build_readable_for_clause(
    *,
    safe: Sequence[str],
    variant: VariantId = "V1",
    risky: Sequence[str] | None = None,
    flag_terms: Sequence[str] | None = None,
    exclusion_terms: Sequence[str] | None = None,
    tier_hybrid_terms: Sequence[str] | None = None,
    tier_hybrid_qualifier: str | None = None,
) -> str:
    """Human-friendly view of :func:`build_for_clause` (no DSL escape backslashes)."""
    raw = build_for_clause(
        safe=safe,
        variant=variant,
        risky=risky,
        flag_terms=flag_terms,
        exclusion_terms=exclusion_terms,
        tier_hybrid_terms=tier_hybrid_terms,
        tier_hybrid_qualifier=tier_hybrid_qualifier,
    )
    return raw.replace("\\", "")


def default_variant_build_order(
    safe: Sequence[str],
    risky: Sequence[str],
    flag_terms: Sequence[str],
    exclusion_terms: Sequence[str],
    *,
    include_v2_diagnostic: bool = False,
    tier_hybrid_terms: Sequence[str] | None = None,
    tier_hybrid_qualifier: str | None = None,
) -> list[tuple[VariantId, str]]:
    """Return ``(variant, for_clause)`` pairs to probe in a typical optimization pass."""
    st = _clean(safe)
    rk = _clean(risky)
    fl = _clean(flag_terms)
    ex = _clean(exclusion_terms)
    th = tier_hybrid_terms
    tq = tier_hybrid_qualifier

    out: list[tuple[VariantId, str]] = []

    if st:
        out.append(
            (
                "V1",
                build_for_clause(
                    safe=st,
                    variant="V1",
                    tier_hybrid_terms=th,
                    tier_hybrid_qualifier=tq,
                ),
            )
        )

    if include_v2_diagnostic and (st or rk):
        out.append(
            (
                "V2",
                build_for_clause(
                    safe=st,
                    variant="V2",
                    risky=rk,
                    tier_hybrid_terms=th,
                    tier_hybrid_qualifier=tq,
                ),
            )
        )

    if rk and fl:
        out.append(
            (
                "V3",
                build_for_clause(
                    safe=st,
                    variant="V3",
                    risky=rk,
                    flag_terms=fl,
                    tier_hybrid_terms=th,
                    tier_hybrid_qualifier=tq,
                ),
            )
        )
        if ex:
            out.append(
                (
                    "V4",
                    build_for_clause(
                        safe=st,
                        variant="V4",
                        risky=rk,
                        flag_terms=fl,
                        exclusion_terms=ex,
                        tier_hybrid_terms=th,
                        tier_hybrid_qualifier=tq,
                    ),
                )
            )

    return out


async def run_variant(
    dsl_port: DimensionsDslPort,
    label: str,
    for_clause: str,
    *,
    search_in: str = "full_data",
    limit: int = 20,
    title_sample: int = 10,
) -> dict[str, Any]:
    """Execute a short Dimensions search for sanity-check counts and top titles."""
    dsl = build_search_publications_dsl(for_clause, search_in=search_in, limit=limit)
    result = await dsl_port.execute_dsl(dsl)
    return {
        "label": label,
        "for_clause": for_clause,
        "dsl_query": dsl,
        "expected_count": parse_total_count(result),
        "top_titles": top_titles_from_result(result, max_titles=title_sample),
    }

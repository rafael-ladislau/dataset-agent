"""Dimensions ``for`` clause builders (V1–V5) and variant probe execution.

See ``dataset_query_agent_spec.md`` §4 and ``docs/spec-gap-analysis.md`` §2.3. Operator
precedence, the parenthesisation used here and the measured phrase rules behind the
escaping and same-search checks: docs/dimensions-search-semantics.md.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from dataset_agent.adapters.dimensions_dsl import (
    build_search_publications_dsl,
    inner_quoted_term,
    parse_total_count,
    top_titles_from_result,
)
from dataset_agent.adapters.dimensions_phrase import phrase_contains
from dataset_agent.adapters.query_heuristics import apply_short_acronym_heuristic
from dataset_agent.adapters.term_sanitizer import sanitize_term
from dataset_agent.domain.ports import DimensionsDslPort

VariantId = Literal["V1", "V2", "V3", "V4", "V5"]


def _clean(terms: Sequence[str] | None) -> list[str]:
    if not terms:
        return []
    out: list[str] = []
    for t in terms:
        if not isinstance(t, str):
            continue
        s = sanitize_term(t)
        if s:
            out.append(s)
    return out


def _q(term: str, server_quoting: bool = False) -> str:
    """Escaped quoted token as used inside the outer ``for \"...\"`` string.

    With ``server_quoting`` the extraction server's raw-clause convention is used:
    only a literal ``"`` inside the term is escaped (no other DSL escaping).
    """
    if server_quoting:
        return '\\"' + term.replace('"', '\\"') + '\\"'
    inner = inner_quoted_term(term)
    return f'\\"{inner}\\"'


def _or_group(terms: list[str], server_quoting: bool = False) -> str:
    if not terms:
        return ""
    parts = [_q(t, server_quoting) for t in terms]
    return "(" + " OR ".join(parts) + ")"


def drop_flags_equal_to_aliases(
    flag_terms: Sequence[str] | None, risky: Sequence[str] | None
) -> list[str]:
    """Flags without any term contained in a risky alias as a token sequence, under the
    measured Dimensions folding, plurals included
    (:func:`~dataset_agent.adapters.dimensions_phrase.phrase_contains`): "GPT 4" in
    "GPT-4", "Research Programs" in "All of Us Research Program". Equality is the
    special case.

    Such a flag makes the gate ``(A OR B) AND (.. OR F)`` satisfy itself, since every
    ``A`` paper also matches ``F``, so every bare ``A`` paper matches.

    The flag group is shared by every risky alias (the extraction seed has one gate), so
    a flag inside one alias leaves the gate of all of them, also of an alias that does
    not contain it. If every flag would be dropped the flags are returned unchanged:
    they still gate the aliases that do not contain them, but an alias that contains
    every flag is in effect searched without a gate. :func:`flag_gate_notes` reports
    both cases for the output.
    """
    fl = _clean(flag_terms)
    rk = _clean(risky)
    if not fl or not rk:
        return fl
    kept = [t for t in fl if not any(phrase_contains(r, t) for r in rk)]
    return kept or fl


def _names(terms: Sequence[str]) -> str:
    return ", ".join(repr(t) for t in terms)


def flag_gate_notes(
    flag_terms: Sequence[str] | None, risky: Sequence[str] | None
) -> list[str]:
    """Output notes for what :func:`drop_flags_equal_to_aliases` does to the shared gate:
    each dropped flag with the aliases that contain it and the aliases that no longer
    match through it, and each risky alias that contains every kept flag (its gate
    matches every one of its papers, so it is in effect ungated)."""
    fl = _clean(flag_terms)
    rk = _clean(risky)
    if not fl or not rk:
        return []
    kept = drop_flags_equal_to_aliases(fl, rk)
    notes: list[str] = []
    for flag in fl:
        if flag in kept:
            continue
        holders = [r for r in rk if phrase_contains(r, flag)]
        others = [r for r in rk if r not in holders]
        note = (f"flag gate: {flag!r} dropped from the shared gate because it occurs inside "
                f"{_names(holders)}")
        if others:
            verb = "no longer matches" if len(others) == 1 else "no longer match"
            note += f"; {_names(others)} {verb} through it"
        notes.append(note)
    for alias in rk:
        if all(phrase_contains(alias, f) for f in kept):
            notes.append(
                f"flag gate: risky alias {alias!r} is in effect ungated: every flag "
                f"({_names(kept)}) occurs inside it, so its gate matches every {alias!r} paper"
            )
    return notes


def drop_contradicting_excludes(
    exclusion_terms: Sequence[str] | None,
    *,
    names: Sequence[str] | None,
    flag_terms: Sequence[str] | None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split excludes into ``(kept, dropped)``; *dropped* holds ``(exclude, inclusion term)``.

    An exclude is dropped when it and a dataset name or flag term are one phrase or one
    contains the other (token-based, case, hyphen and plural folding:
    :func:`~dataset_agent.adapters.dimensions_phrase.phrase_contains`). ``NOT "census"``
    next to the flag "Census Bureau" removes the very papers the flag asks for. Order is
    preserved. This is not part of :func:`build_for_clause`: the recall-complement probes
    deliberately place aliases in ``exclusion_terms``.
    """
    inclusion = _clean(names) + _clean(flag_terms)
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for ex in _clean(exclusion_terms):
        hit = next(
            (inc for inc in inclusion if phrase_contains(inc, ex) or phrase_contains(ex, inc)),
            None,
        )
        if hit is None:
            kept.append(ex)
        else:
            dropped.append((ex, hit))
    return kept, dropped


def build_for_clause(
    *,
    safe: Sequence[str],
    variant: VariantId = "V1",
    risky: Sequence[str] | None = None,
    flag_terms: Sequence[str] | None = None,
    exclusion_terms: Sequence[str] | None = None,
    tier_hybrid_terms: Sequence[str] | None = None,
    tier_hybrid_qualifier: str | None = None,
    server_quoting: bool = False,
) -> str:
    """Assemble a boolean ``for`` sub-expression (UPPERCASE ``OR`` / ``AND`` / ``NOT``)."""
    sq = server_quoting
    st = _clean(safe)
    rk = _clean(risky)
    fl = _clean(flag_terms)
    if rk and fl:
        fl = drop_flags_equal_to_aliases(fl, rk)
    ex = _clean(exclusion_terms)
    th = _clean(tier_hybrid_terms)
    tq = tier_hybrid_qualifier.strip() if isinstance(tier_hybrid_qualifier, str) else ""

    if variant == "V1":
        return _or_group(st, sq)

    if variant == "V2":
        return _or_group(st + rk, sq)

    if variant == "V5":
        return _or_group(st, sq)

    # V3 / V4: safe OR hybrid; optional tier-hybrid OR arm; optional NOT exclusions (V4).
    chunks: list[str] = []
    safe_expr = _or_group(st, sq)
    if safe_expr:
        chunks.append(safe_expr)

    if rk and fl:
        hybrid = f"({_or_group(rk, sq)} AND {_or_group(fl, sq)})"
        chunks.append(hybrid)
    elif rk:
        chunks.append(_or_group(rk, sq))
    elif fl and not st:
        chunks.append(_or_group(fl, sq))

    if th and tq:
        chunks.append(f"({_or_group(th, sq)} AND {_or_group([tq], sq)})")
    chunks = [c for c in chunks if c]
    body = " OR ".join(chunks)

    if variant == "V4" and ex and body:
        # Every group is parenthesised: Dimensions drops a group of an unparenthesised
        # "A OR B AND C" (it returns B AND C), so the result must never depend on
        # precedence. With no positive part there is nothing to search: a bare
        # "NOT (E)" would match everything outside E.
        positive = chunks[0] if len(chunks) == 1 else f"({body})"
        body = f"{positive} NOT {_or_group(ex, sq)}"

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
    st, rk = apply_short_acronym_heuristic(st, rk)
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

"""Recall check: propose missing canonical names as safe aliases, with full-text evidence.

The precision loop only measures how many matched papers are correct, so a final query can
be precise but tiny (a program's long name sitting in the flag terms never matches on its
own). This module selects name-like candidates deterministically and judges them on a
*complement* probe: top-cited papers that match the candidate but not the current query.

Deterministic rules decide; no LLM is involved. The complement probe is the real gate,
the filters below only save probe slots.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any, Literal

from dataset_agent.adapters.dimensions_phrase import search_key
from dataset_agent.adapters.fulltext_evidence import Evidence, _fold, _phrase_in, term_in
from dataset_agent.adapters.fulltext_refiner import _NAME_JOINERS, _generic_word
from dataset_agent.adapters.query_heuristics import (
    detect_rhetorical_name,
    head_acronym,
    is_acronym_suffix_form,
    is_short_acronym,
    looks_like_acronym_head,
)
from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.adapters.term_sanitizer import sanitize_term
from dataset_agent.settings import Settings

_MAX_TERM_LEN = 120
# a candidate that adds more Dimensions hits than this is not probed
MAX_ADDED_HITS = 500_000
_MAX_QUOTES = 3
# Words that describe a kind of resource without naming one ("Data Commons", "Research Center").
_GENERIC_NAME_WORDS = frozenset({
    "data", "dataset", "datasets", "database", "databases", "commons", "research", "program",
    "programme", "project", "study", "studies", "survey", "center", "centre", "resource",
    "resources", "network", "initiative", "repository", "archive", "portal", "platform",
    "system", "service", "services", "collection", "registry", "cohort", "consortium",
    "observatory", "atlas", "hub", "bank", "biobank",
})


def _all_generic_words(term: str) -> bool:
    toks = [t for t in _fold(term).split() if t not in _NAME_JOINERS]
    return bool(toks) and all(t in _GENERIC_NAME_WORDS for t in toks)


@dataclass
class RecallCandidate:
    term: str
    source: Literal["flag", "name", "alias"]
    tier: int


def _initials(term: str, *, drop_joiners: bool) -> str:
    toks = _fold(term).split()
    if drop_joiners:
        toks = [t for t in toks if t not in _NAME_JOINERS]
    return "".join(t[0] for t in toks).upper()


def dataset_acronyms(
    names: list[str], aliases: list[str], *, head_aliases: list[str] | None = None
) -> set[str]:
    """Upper-case acronyms that stand for the dataset: single-token acronym-shaped names
    or aliases, the acronym head of a multi-word name or of a multi-word alias in
    *head_aliases* (vetted aliases only; all *aliases* when omitted) — head position only,
    so "All of Us NIH" never makes NIH related — and initials of names with 3+ words."""
    out: set[str] = set()
    for t in [*names, *aliases]:
        toks = (t or "").split()
        if len(toks) == 1 and looks_like_acronym_head(toks[0], max_len=10):
            out.add(_fold(toks[0]).upper())
    for t in [*names, *(aliases if head_aliases is None else head_aliases)]:
        head = head_acronym(t)
        if head:
            out.add(head.upper())
    for n in names:
        if len(_fold(n).split()) >= 3:
            for dj in (False, True):
                ini = _initials(n, drop_joiners=dj)
                if len(ini) >= 3:
                    out.add(ini)
    out.discard("")
    return out


# Words too generic (or agency-like) to anchor a shared name phrase on their own.
_PHRASE_GENERIC_WORDS = _GENERIC_NAME_WORDS | frozenset({
    "national", "institute", "institutes", "health", "medical", "science", "sciences",
    "foundation", "agency", "department", "office", "bureau", "federal", "united", "states",
    "american", "international", "center", "centers", "university", "college", "council",
    "association", "society", "laboratory", "laboratories", "information", "management",
    "analysis", "statistics", "services", "public", "general",
})


def _shared_name_phrase(flag: str, forms: list[str]) -> bool:
    """True when *flag* and some name or multi-word alias share a contiguous token phrase
    with 2+ non-joiner words that are not all generic ("All of Us" in "All of Us NIH" and
    "All of Us Research Program"; never "National ... Health" overlaps)."""
    ftoks = _fold(flag).split()

    def grams(toks: list[str]) -> set[tuple[str, ...]]:
        out: set[tuple[str, ...]] = set()
        for i in range(len(toks)):
            for j in range(i + 2, len(toks) + 1):
                g = toks[i:j]
                if g[0] in _NAME_JOINERS or g[-1] in _NAME_JOINERS:
                    continue
                words = [t for t in g if t not in _NAME_JOINERS]
                # Single-letter words are fold fragments ("Alzheimer's" -> "alzheimer s",
                # "U.S." -> "u s"), not name words.
                if any(len(w) < 2 for w in words):
                    continue
                if len(words) >= 2 and not all(w in _PHRASE_GENERIC_WORDS for w in words):
                    out.add(tuple(g))
        return out

    # A phrase equal to the whole flag is a containment case, handled by the name rule.
    fg = {g for g in grams(ftoks) if len(g) < len(ftoks)}
    return bool(fg) and any(fg & grams(_fold(f).split()) for f in forms if f)


def _flag_names_dataset(flag: str, names: list[str], acronyms: set[str]) -> bool:
    if any(_phrase_in(n, flag, plural_ok=False) for n in names if n):
        return True
    if _shared_name_phrase(flag, names):
        return True
    toks = _fold(flag).split()
    if not toks:
        return False
    if len(toks) == 1 and toks[0].upper() in acronyms:
        return True
    if len(toks) > 1 and toks[0].upper() in acronyms:
        return True
    return any(
        len(ini) >= 3 and ini in acronyms
        for ini in (_initials(flag, drop_joiners=False), _initials(flag, drop_joiners=True))
    )


def recall_candidates(
    current: QueryCandidate,
    *,
    main_names: list[str],
    known_aliases: list[str],
    flag_terms: list[str],
    evidence_against: set[str] | None = None,
    data_phrases: list[str] | None = None,
) -> tuple[list[RecallCandidate], list[str]]:
    """``(eligible in pool order, rejection notes)``.

    Pool: tier 0 = *flag_terms* then *main_names*; tier 1 = *known_aliases* that are not
    safe now (risky, or dropped by phase 1). *evidence_against*: aliases the refiner dropped
    or demoted on full-text evidence. *data_phrases*: generated data phrases of a resource
    map ("<program> data", "data from the <program>"); they are data references by
    construction, so the proper-name, generic-phrase and rhetorical filters do not apply."""
    phrase_f = {search_key(p) for p in (data_phrases or [])}
    against = {search_key(a) for a in (evidence_against or set())}
    safe_f = {search_key(s) for s in current.safe}
    all_names = [n for n in (sanitize_term(x) for x in main_names) if n]
    # Name-relatedness only: a multi-word alias (even a rhetorical one such as "All of Us"
    # when every main name is "All of Us NIH") also makes a containing flag a name of the dataset.
    related_forms = list(all_names)
    for a in [*current.safe, *current.risky, *known_aliases]:
        sa = sanitize_term(a)
        if sa and len(sa.split()) >= 2:
            related_forms.append(sa)
    # A flag-gated or dropped alias such as "UNAVCO geodetic data" must not make its head
    # organization ("UNAVCO") a name of the dataset; only safe aliases lend their head.
    acronyms = dataset_acronyms(
        all_names, [*current.safe, *current.risky, *known_aliases], head_aliases=list(current.safe)
    )
    pool: list[tuple[str, str, int]] = []
    pool += [(t, "flag", 0) for t in flag_terms]
    pool += [(t, "name", 0) for t in main_names]
    pool += [(t, "alias", 1) for t in known_aliases]

    eligible: list[RecallCandidate] = []
    rejected: list[str] = []
    seen: set[str] = set()
    risky_f = {search_key(r) for r in current.risky}
    for raw, source, tier in pool:
        term = sanitize_term(raw)
        if not term or search_key(term) in seen:
            continue
        seen.add(search_key(term))
        if search_key(term) in safe_f:
            continue

        def reject(why: str, term: str = term) -> None:
            rejected.append(f"{term} ({why})")

        if '"' in term or "\\" in term or len(term) > _MAX_TERM_LEN:
            reject("unusable characters or too long")
        elif is_short_acronym(term):
            reject("short acronym")
        elif is_acronym_suffix_form(term):
            reject("short acronym plus generic suffix")
        elif _generic_word(term) and search_key(term) not in phrase_f:
            reject("not a proper name")
        elif _all_generic_words(term):
            reject("generic phrase")
        elif search_key(term) in against:
            reject("full text already rejected it")
        elif detect_rhetorical_name(term) and search_key(term) not in phrase_f:
            reject("rhetorical phrase")
        elif search_key(term) in risky_f and not current.flag_terms:
            reject("already matches without flags")
        elif source == "flag" and not _flag_names_dataset(term, related_forms, acronyms):
            reject("not a name of the dataset")
        else:
            eligible.append(RecallCandidate(term=term, source=source, tier=tier))  # type: ignore[arg-type]
    return eligible, rejected


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound of the proportion *k*/*n* (0.0 when *n* is 0)."""
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - spread) / denom)


def judge_complement(
    term: str,
    evidence: Evidence,
    added_hits: int,
    settings: Settings,
    *,
    base_count: int | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """``(promote, reason, stats)`` for one complement probe.

    *base_count*: Dimensions count of the selected query. A promotion adding more than
    ``fulltext_recall_expansion_ratio`` x that count must also clear the Wilson lower bound."""
    n = evidence.n_evaluated
    genuine = evidence.genuine
    precision = (len(genuine) / n) if n else None
    lower = wilson_lower(len(genuine), n) if n else None
    stats: dict[str, Any] = {
        "added_hits": added_hits,
        "n_evaluated": n,
        "matched": len(genuine),
        "used": sum(1 for v in evidence.verdicts if v.used),
        "complement_precision": round(precision, 4) if precision is not None else None,
        "n_delivered": evidence.n_delivered,
        "n_sampled": evidence.n_sampled,
        "stop_reason": evidence.stop_reason,
        "budget_exhausted": evidence.budget_exhausted,
        "term_in_text": sum(1 for v in evidence.verdicts if term_in(term, v)),
        "precision_lower_bound": round(lower, 4) if lower is not None else None,
        "base_count": base_count,
    }
    need_n = settings.fulltext_recall_min_evidence
    need_p = settings.fulltext_recall_min_precision
    if n < need_n:
        why = f"only {n} full-text paper(s) evaluated (need {need_n})"
        if evidence.shortfall_note:
            why += f"; {evidence.shortfall_note}"
        return False, why, stats
    assert precision is not None
    if precision < need_p:
        return False, f"complement precision {precision:.2f} on {n} papers (need {need_p:.2f})", stats
    if base_count and added_hits > settings.fulltext_recall_expansion_ratio * base_count:
        need_lb = need_p - settings.fulltext_recall_expansion_margin
        assert lower is not None
        if lower < need_lb:
            # The point precision met the need; only the interval rejects it.
            stats["rejected_on_lower_bound"] = True
            return False, (
                f"complement precision {precision:.2f} on {n} papers is too uncertain for "
                f"+{added_hits} hits on a query of {base_count} "
                f"(lower bound {lower:.2f}, need {need_lb:.2f})"
            ), stats
    return True, (
        f"recall: complement precision {precision:.2f} on {n} papers; +{added_hits} Dimensions hits"
        f"; {stats['used']} of {n} use the data"
    ), stats


def evidence_quotes(term: str, evidence: Evidence) -> list:
    """Up to three quotes: genuine papers whose text shows the term first."""
    ranked = sorted(
        evidence.verdicts,
        key=lambda v: (not v.genuine, not term_in(term, v)),
    )
    return [v.as_quote(term) for v in ranked[:_MAX_QUOTES]]

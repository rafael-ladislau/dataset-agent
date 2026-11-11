"""Recall check: propose missing canonical names as safe aliases, with full-text evidence.

The precision loop only measures how many matched papers are correct, so a final query can
be precise but tiny (a program's long name sitting in the flag terms never matches on its
own). This module selects name-like candidates deterministically and judges them on a
*complement* probe: top-cited papers that match the candidate but not the current query.

Deterministic rules decide; no LLM is involved. The complement probe is the real gate,
the filters below only save probe slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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


def _acronym_compound(term: str) -> bool:
    """Two or more short acronym-shaped tokens ("PBO GPS"): no distinctive name."""
    toks = term.split()
    return len(toks) >= 2 and all(
        looks_like_acronym_head(t, max_len=5) and len(t) <= 5 for t in toks
    )


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


def dataset_acronyms(names: list[str], aliases: list[str]) -> set[str]:
    """Upper-case acronyms that stand for the dataset: single-token acronym-shaped names
    or aliases, the acronym head of any multi-word name or alias (head position only, so
    "All of Us NIH" never makes NIH related), and initials of names with 3+ words."""
    out: set[str] = set()
    for t in [*names, *aliases]:
        toks = (t or "").split()
        if len(toks) == 1 and looks_like_acronym_head(toks[0], max_len=10):
            out.add(_fold(toks[0]).upper())
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


def _flag_names_dataset(flag: str, names: list[str], acronyms: set[str]) -> bool:
    if any(_phrase_in(n, flag, plural_ok=False) for n in names if n):
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
) -> tuple[list[RecallCandidate], list[str]]:
    """``(eligible in pool order, rejection notes)``.

    Pool: tier 0 = *flag_terms* then *main_names*; tier 1 = *known_aliases* that are not
    safe now (risky, or dropped by phase 1). *evidence_against*: aliases the refiner dropped
    or demoted on full-text evidence."""
    against = {_fold(a) for a in (evidence_against or set())}
    safe_f = {_fold(s) for s in current.safe}
    all_names = [n for n in (sanitize_term(x) for x in main_names) if n]
    # Name-relatedness only: a multi-word alias (even a rhetorical one such as "All of Us"
    # when every main name is "All of Us NIH") also makes a containing flag a name of the dataset.
    related_forms = list(all_names)
    for a in [*current.safe, *current.risky, *known_aliases]:
        sa = sanitize_term(a)
        if sa and len(sa.split()) >= 2:
            related_forms.append(sa)
    acronyms = dataset_acronyms(
        all_names, [*current.safe, *current.risky, *known_aliases]
    )
    pool: list[tuple[str, str, int]] = []
    pool += [(t, "flag", 0) for t in flag_terms]
    pool += [(t, "name", 0) for t in main_names]
    pool += [(t, "alias", 1) for t in known_aliases]

    eligible: list[RecallCandidate] = []
    rejected: list[str] = []
    seen: set[str] = set()
    risky_f = {_fold(r) for r in current.risky}
    for raw, source, tier in pool:
        term = sanitize_term(raw)
        if not term or _fold(term) in seen:
            continue
        seen.add(_fold(term))
        if _fold(term) in safe_f:
            continue

        def reject(why: str, term: str = term) -> None:
            rejected.append(f"{term} ({why})")

        if '"' in term or "\\" in term or len(term) > _MAX_TERM_LEN:
            reject("unusable characters or too long")
        elif is_short_acronym(term):
            reject("short acronym")
        elif is_acronym_suffix_form(term):
            reject("short acronym plus generic suffix")
        elif _generic_word(term):
            reject("not a proper name")
        elif _all_generic_words(term):
            reject("generic phrase")
        elif _acronym_compound(term):
            reject("acronyms only")
        elif _fold(term) in against:
            reject("full text already rejected it")
        elif detect_rhetorical_name(term):
            reject("rhetorical phrase")
        elif _fold(term) in risky_f and not current.flag_terms:
            reject("already matches without flags")
        elif source == "flag" and not _flag_names_dataset(term, related_forms, acronyms):
            reject("not a name of the dataset")
        else:
            eligible.append(RecallCandidate(term=term, source=source, tier=tier))  # type: ignore[arg-type]
    return eligible, rejected


def judge_complement(
    term: str,
    evidence: Evidence,
    added_hits: int,
    settings: Settings,
) -> tuple[bool, str, dict[str, Any]]:
    """``(promote, reason, stats)`` for one complement probe."""
    n = evidence.n_evaluated
    genuine = evidence.genuine
    precision = (len(genuine) / n) if n else None
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
    return True, f"recall: complement precision {precision:.2f} on {n} papers; +{added_hits} Dimensions hits", stats


def evidence_quotes(term: str, evidence: Evidence) -> list:
    """Up to three quotes: genuine papers whose text shows the term first."""
    ranked = sorted(
        evidence.verdicts,
        key=lambda v: (not v.genuine, not term_in(term, v)),
    )
    return [v.as_quote(term) for v in ranked[:_MAX_QUOTES]]

"""Full-text-driven query refiner.

Deterministic rules decide whether an existing alias is kept, dropped or demoted
(safe -> risky, i.e. requires flag terms); the LLM only *proposes* exclude / flag /
alias candidates from evidence quotes, and every proposal is validated against the
quotes before it is accepted. A term is changed only with full-text evidence:
terms without enough verified papers stay ``unverified`` and abstract-only signals
can at most mark a term ``suspect``.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from dataset_agent.adapters.fulltext_evidence import (
    Evidence,
    PubVerdict,
    attribute_terms,
    singular_variants,
    term_in,
    term_in_quote,
)
from dataset_agent.adapters.literature import _stem_text
from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.adapters.text_processing import dedupe_strings_ci_preserve_order
from dataset_agent.adapters.tools import EMIT_REFINEMENT_CANDIDATES
from dataset_agent.domain.ports import AgentPort
from dataset_agent.domain.proposals import TermDiff
from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)

_MAX_EVIDENCE = 3


@dataclass
class RefineResult:
    candidate: QueryCandidate
    diffs: list[TermDiff] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(d.action != "keep" for d in self.diffs)


def _quotes(verdicts: list[PubVerdict]):
    return [v.as_quote() for v in verdicts[:_MAX_EVIDENCE]]


def _is_quote_char(ch: str) -> bool:
    return ch in "\"'`" or unicodedata.category(ch) in ("Pi", "Pf")


def _unquote(term: str) -> str:
    """Strip wrapping quote characters of any kind (straight, curly, guillemets)."""
    t = term.strip()
    while t and _is_quote_char(t[0]):
        t = t[1:].strip()
    while t and _is_quote_char(t[-1]):
        t = t[:-1].strip()
    return t


def propose_refinements(
    agent: AgentPort,
    *,
    dataset_name: str,
    genuine: list[PubVerdict],
    false_positives: list[PubVerdict],
    aliases: list[str],
    flag_terms: list[str],
    exclude_terms: list[str],
) -> dict[str, list[str]]:
    """LLM: candidate excludes / flags / new aliases from verified quotes."""

    def lines(vs: list[PubVerdict]) -> str:
        return "\n".join(f"- [{v.title[:80]}] {v.quote[:300]}" for v in vs[:12]) or "(none)"

    prompt = f"""Dataset: {dataset_name!r}.
Current aliases: {aliases!r}; flag terms: {flag_terms!r}; exclude terms: {exclude_terms!r}.

Papers whose full text GENUINELY USES the dataset (evidence quotes):
{lines(genuine)}

Papers that do NOT use it (false positives; evidence quotes):
{lines(false_positives)}

Propose: (1) short exclude phrases that occur in the false-positive quotes/titles but NOT in the genuine ones;
(2) flag terms: context words that occur around the genuine quotes; (3) new alias forms of the dataset name
that appear in the genuine quotes but are missing from the current aliases.
Only use words that literally occur in the quotes above.

Use emit_refinement_candidates exactly once."""
    try:
        raw = agent.get_structured(prompt, EMIT_REFINEMENT_CANDIDATES)
    except Exception as exc:
        logger.warning("propose_refinements failed: %s", exc)
        return {"exclude_terms": [], "flag_terms": [], "new_aliases": []}

    def clean(key: str, cap: int) -> list[str]:
        v = raw.get(key)
        if not isinstance(v, list):
            return []
        # Strip wrapping quotes the LLM may add; a term that still contains a double
        # quote cannot be sent to extraction (seed terms reject it), so drop it.
        terms = [_unquote(x) for x in v if isinstance(x, str)]
        return dedupe_strings_ci_preserve_order(
            [t for t in terms if len(t) > 2 and '"' not in t]
        )[:cap]

    return {
        "exclude_terms": clean("exclude_terms", 10),
        "flag_terms": clean("flag_terms", 10),
        "new_aliases": clean("new_aliases", 5),
    }


def _exclude_reason(term: str, hits: list[PubVerdict]) -> str:
    n_quote = sum(1 for v in hits if term_in_quote(term, v))
    n_title = len(hits) - n_quote
    where = f"{n_quote} evidence quote(s)"
    if n_title:
        where += f" and {n_title} title(s) only"
    return f"occurs in {where} of {len(hits)} false-positive paper(s) and no genuine one"


def _covered(term: str, existing: list[str]) -> bool:
    stem = _stem_text(term)
    return any(_stem_text(e) == stem for e in existing)


def refine_candidate(
    candidate: QueryCandidate,
    evidence: Evidence,
    *,
    settings: Settings,
    agent: AgentPort | None,
    dataset_name: str,
    count_fn: Callable[[str], int] | None = None,
    reference_count: int = 0,
    abstract_fp_aliases: set[str] | None = None,
) -> RefineResult:
    """Return the refined candidate plus a per-term diff with evidence and reasons."""
    abstract_fp = {a.lower() for a in (abstract_fp_aliases or set())}
    aliases = candidate.all_aliases()
    safe_l = {a.lower() for a in candidate.safe}
    diffs: list[TermDiff] = []
    enough = evidence.n_evaluated >= settings.fulltext_min_evidence

    stats = attribute_terms(evidence, aliases)
    flags_stats = attribute_terms(evidence, list(candidate.flag_terms))

    # --- LLM proposals, validated against the quotes ------------------------------
    new_excludes: list[tuple[str, list[PubVerdict]]] = []
    new_flags: list[tuple[str, list[PubVerdict]]] = []
    new_aliases: list[tuple[str, list[PubVerdict]]] = []
    if enough and agent is not None and (evidence.genuine or evidence.false_positives):
        prop = propose_refinements(
            agent,
            dataset_name=dataset_name,
            genuine=evidence.genuine,
            false_positives=evidence.false_positives,
            aliases=aliases,
            flag_terms=candidate.flag_terms,
            exclude_terms=candidate.exclude_terms,
        )
        for t in prop["exclude_terms"]:
            if _covered(t, candidate.exclude_terms):
                continue
            hit_fp = [v for v in evidence.false_positives if term_in(t, v)]
            # Over-matching the genuine side is the safe direction here: an exclude
            # is blocked when the term, or its singular, occurs in a genuine paper.
            forms = [t, *singular_variants(t)]
            hit_gen = [v for v in evidence.genuine if any(term_in(f, v) for f in forms)]
            if hit_fp and not hit_gen:
                new_excludes.append((t, hit_fp))
        # Flags need at least two genuine quotes (quote-only; titles never count).
        for t in prop["flag_terms"]:
            if _covered(t, candidate.flag_terms):
                continue
            hit_gen = [v for v in evidence.genuine if term_in_quote(t, v)]
            if len(hit_gen) >= 2:
                new_flags.append((t, hit_gen))
        for t in prop["new_aliases"]:
            if _covered(t, aliases):
                continue
            # A new alias must occur literally (no plural folding) in a genuine quote.
            hit_gen = [v for v in evidence.genuine if term_in_quote(t, v, exact=True)]
            if hit_gen:
                new_aliases.append((t, hit_gen))

    flags = list(candidate.flag_terms) + [t for t, _ in new_flags]

    # --- alias decisions (deterministic) -------------------------------------------
    keep_safe: list[str] = []
    keep_risky: list[str] = []
    n_remaining = len(aliases)
    for a in aliases:
        st = stats[a]
        is_safe = a.lower() in safe_l
        base = {"stats": st.as_dict()}
        if not enough or st.n < settings.refine_min_alias_papers:
            status = "suspect" if a.lower() in abstract_fp else "unverified"
            why = (
                "abstract-only FP signal; not enough full-text papers to change it"
                if status == "suspect"
                else f"only {st.n} verified paper(s) mention it (need {settings.refine_min_alias_papers})"
            )
            diffs.append(TermDiff(kind="alias", term=a, action="keep", status=status, reason=why, **base))
            (keep_safe if is_safe else keep_risky).append(a)
            continue
        share = st.fp_share
        if share >= settings.refine_drop_fp_share and n_remaining > 1:
            diffs.append(TermDiff(
                kind="alias", term=a, action="drop", status="verified",
                reason=f"{st.fp}/{st.n} verified papers with this term are false positives",
                evidence=_quotes(st.fp_papers), **base))
            n_remaining -= 1
        elif share >= settings.refine_demote_fp_share and is_safe and flags:
            diffs.append(TermDiff(
                kind="alias", term=a, action="demote", status="verified",
                reason=f"{st.fp}/{st.n} verified papers are false positives; now requires flag terms",
                evidence=_quotes(st.fp_papers), **base))
            keep_risky.append(a)
        else:
            reason = "verified usage in full text" if share < settings.refine_demote_fp_share else (
                "high FP share but cannot be dropped/demoted (last alias or no flag terms)")
            diffs.append(TermDiff(
                kind="alias", term=a, action="keep", status="verified", reason=reason,
                evidence=_quotes(st.used_papers), **base))
            (keep_safe if is_safe else keep_risky).append(a)

    # --- new aliases (re-probe Dimensions counts) ------------------------------------
    added_aliases = 0
    for t, hits in new_aliases:
        if added_aliases >= settings.refine_max_new_aliases or count_fn is None:
            break
        try:
            c = int(count_fn(t))
        except Exception as exc:  # budget exhausted or Dimensions error
            logger.info("new alias %r skipped: %s", t, exc)
            continue
        if c <= 0:
            continue
        risky = c > 10 * max(1, reference_count)
        if risky and not flags:
            continue
        (keep_risky if risky else keep_safe).append(t)
        added_aliases += 1
        diffs.append(TermDiff(
            kind="alias", term=t, action="add", status="verified",
            reason=f"mention form in {len(hits)} genuine quote(s); {c} Dimensions hits"
            + (" (risky: requires flags)" if risky else ""),
            stats={"dimensions_count": c, "genuine_papers": len(hits)}, evidence=_quotes(hits)))

    # --- flags / excludes --------------------------------------------------------------
    for f in candidate.flag_terms:
        st = flags_stats[f]
        diffs.append(TermDiff(
            kind="flag", term=f, action="keep",
            status="verified" if enough and st.n >= settings.refine_min_alias_papers else "unverified",
            reason="unchanged", stats=st.as_dict()))
    for t, hits in new_flags:
        diffs.append(TermDiff(
            kind="flag", term=t, action="add", status="verified",
            reason=f"occurs in {len(hits)} genuine evidence quote(s)", evidence=_quotes(hits),
            stats={"genuine_papers": len(hits)}))
    for e in candidate.exclude_terms:
        diffs.append(TermDiff(kind="exclude", term=e, action="keep", status="unverified", reason="unchanged"))
    for t, hits in new_excludes:
        diffs.append(TermDiff(
            kind="exclude", term=t, action="add", status="verified",
            reason=_exclude_reason(t, hits),
            evidence=_quotes(hits), stats={"fp_papers": len(hits)}))

    refined = QueryCandidate(
        main_dataset_name=candidate.main_dataset_name,
        safe=keep_safe,
        risky=keep_risky,
        flag_terms=dedupe_strings_ci_preserve_order(flags),
        exclude_terms=dedupe_strings_ci_preserve_order(
            list(candidate.exclude_terms) + [t for t, _ in new_excludes]),
        description=candidate.description,
    )
    # An explicit raw clause cannot be edited term by term; keep it only if unchanged.
    if candidate.raw_clause and not any(d.action != "keep" for d in diffs):
        refined.raw_clause = candidate.raw_clause
    return RefineResult(candidate=refined, diffs=diffs)

"""Full-text-driven query refiner.

A *match* is a paper whose full text mentions the dataset (used or not); a false
positive is one that does not mention it. Usage never decides a term change.

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
    _phrase_in,
    attribute_terms,
    singular_variants,
    term_in,
    term_in_evidence,
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
#: Papers per side and snippets per paper shown to the LLM pattern pass.
_MAX_PROMPT_PAPERS = 40
_MAX_PROMPT_SNIPPETS = 2


@dataclass
class RefineResult:
    candidate: QueryCandidate
    diffs: list[TermDiff] = field(default_factory=list)
    #: Patterns found across the papers (false-positive groups, fields of research).
    notes: list[str] = field(default_factory=list)

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


def _paper_line(v: PubVerdict) -> str:
    bits = [f"- [{v.title[:100]}]"]
    if v.categories:
        bits.append(f"(field: {'; '.join(v.categories[:2])})")
    if v.quote:
        bits.append(f"quote: {v.quote[:240]}")
    snippets = [c for c in v.contexts if c.get("text")][:_MAX_PROMPT_SNIPPETS]
    for c in snippets:
        bits.append(f"| {c.get('part', '')} «{c.get('term', '')}»: {str(c['text'])[:300]}")
    return " ".join(bits)


def propose_refinements(
    agent: AgentPort,
    *,
    dataset_name: str,
    genuine: list[PubVerdict],
    false_positives: list[PubVerdict],
    aliases: list[str],
    flag_terms: list[str],
    exclude_terms: list[str],
) -> dict:
    """LLM pattern pass over all probe papers: false-positive groups with excludes,
    flags and new aliases from the matching papers."""

    def lines(vs: list[PubVerdict]) -> str:
        return "\n".join(_paper_line(v) for v in vs[:_MAX_PROMPT_PAPERS]) or "(none)"

    prompt = f"""Dataset: {dataset_name!r}.
Current aliases: {aliases!r}; flag terms: {flag_terms!r}; exclude terms: {exclude_terms!r}.
Each paper below shows its title, field of research, the validator's quote when there is one, and
snippets of the full text around the query term that matched it («term»).

Papers whose full text MENTIONS the dataset (correct matches, whether or not they use it):
{lines(genuine)}

Papers matched by the query whose full text does NOT mention the dataset at all (false positives):
{lines(false_positives)}

Look at the false positives TOGETHER: group them by the other meaning of the query term that pulled
them in (for example an acronym that also names a medical condition, a technology or an author), and
for each group give the query term that caused it and short exclude phrases that occur in that
group's snippets or titles but in none of the matching papers.
Then look at the matching papers TOGETHER: give flag terms (context words they share around the
dataset name, such as the agency or program) and new alias forms of the dataset name that appear in
their quotes or snippets but are missing from the current aliases.
Only use words that literally occur in the text above.

Use emit_refinement_candidates exactly once."""
    try:
        raw = agent.get_structured(prompt, EMIT_REFINEMENT_CANDIDATES)
    except Exception as exc:
        logger.warning("propose_refinements failed: %s", exc)
        return {"exclude_terms": [], "flag_terms": [], "new_aliases": [], "groups": []}

    def clean_list(v, cap: int) -> list[str]:
        if not isinstance(v, list):
            return []
        # Strip wrapping quotes the LLM may add; a term that still contains a double
        # quote cannot be sent to extraction (seed terms reject it), so drop it.
        terms = [_unquote(x) for x in v if isinstance(x, str)]
        return dedupe_strings_ci_preserve_order(
            [t for t in terms if len(t) > 2 and '"' not in t]
        )[:cap]

    groups = []
    for g in raw.get("false_positive_groups") or []:
        if not isinstance(g, dict) or not isinstance(g.get("label"), str):
            continue
        groups.append({
            "label": g["label"][:120],
            "cause_term": str(g.get("cause_term") or "")[:80],
            "papers": int(g["papers"]) if isinstance(g.get("papers"), int) else 0,
            "exclude_terms": clean_list(g.get("exclude_terms"), 5),
        })
    group_excludes = [t for g in groups[:6] for t in g["exclude_terms"]]
    return {
        "exclude_terms": dedupe_strings_ci_preserve_order(
            clean_list(raw.get("exclude_terms"), 10) + group_excludes)[:15],
        "flag_terms": clean_list(raw.get("flag_terms"), 10),
        "new_aliases": clean_list(raw.get("new_aliases"), 5),
        "groups": groups[:6],
    }


def category_patterns(evidence: Evidence, min_fp: int) -> list[str]:
    """Fields of research concentrated among the false positives and rare among matches.

    Reported only (never applied): excluding a whole field is a strong, manual decision.
    """
    fps, matches = evidence.false_positives, evidence.genuine
    if not fps:
        return []
    counts: dict[str, list[int]] = {}
    for v in fps:
        for c in set(v.categories):
            counts.setdefault(c, [0, 0])[0] += 1
    for v in matches:
        for c in set(v.categories):
            counts.setdefault(c, [0, 0])[1] += 1
    out = []
    for name, (n_fp, n_match) in sorted(counts.items(), key=lambda kv: -kv[1][0]):
        if n_fp < min_fp or n_fp / len(fps) < 0.5:
            continue
        if matches and n_match / len(matches) > 0.1:
            continue
        out.append(
            f"full-text pattern: {n_fp}/{len(fps)} false positives are in field of research "
            f"'{name}' vs {n_match}/{len(matches)} matches; consider excluding that field (manual)."
        )
    return out


def _exclude_reason(term: str, hits: list[PubVerdict]) -> str:
    n_text = sum(1 for v in hits if term_in_evidence(term, v))
    n_title = len(hits) - n_text
    where = f"{n_text} full-text snippet(s)/quote(s)"
    if n_title:
        where += f" and {n_title} title(s) only"
    return f"occurs in {where} of {len(hits)} false-positive paper(s) and no matching one"


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
    impact_fn: Callable[[str], int] | None = None,
) -> RefineResult:
    """*impact_fn(term)*: how many current query results also contain *term* (what an
    exclude would remove); an exclude that removes nothing is rejected."""
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
    notes: list[str] = []
    impacts: dict[str, int] = {}
    if enough:
        notes.extend(category_patterns(evidence, settings.refine_category_min_fp_papers))
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
        for g in prop.get("groups") or []:
            notes.append(
                f"full-text pattern: false-positive group '{g['label']}' "
                f"(~{g['papers']} paper(s), via '{g['cause_term']}')"
                + (f"; candidate excludes {g['exclude_terms']}" if g["exclude_terms"] else "")
            )
        for t in prop["exclude_terms"]:
            if _covered(t, candidate.exclude_terms):
                continue
            # NOT ("term") runs over all of full_data: an exclude inside or containing a
            # kept alias, or inside a flag, would silently drop the query's own matches.
            if any(_phrase_in(t, a) or _phrase_in(a, t) for a in aliases) or any(
                _phrase_in(t, f) for f in candidate.flag_terms
            ):
                continue
            hit_fp = [v for v in evidence.false_positives if term_in(t, v)]
            # Over-matching the genuine side is the safe direction here: an exclude
            # is blocked when the term, or its singular, occurs in a genuine paper.
            forms = [t, *singular_variants(t)]
            hit_gen = [v for v in evidence.genuine if any(term_in(f, v) for f in forms)]
            if len(hit_fp) < settings.refine_min_exclude_papers or hit_gen:
                continue
            if impact_fn is not None:
                try:
                    removed = int(impact_fn(t))
                except Exception as exc:  # budget exhausted or Dimensions error
                    logger.info("exclude %r impact unknown: %s", t, exc)
                else:
                    if removed <= 0:
                        continue
                    impacts[t] = removed
            new_excludes.append((t, hit_fp))
        # Flags need at least two matching papers and a minimum share of them, from
        # quotes or full-text snippets (titles never count).
        n_match = len(evidence.genuine)
        for t in prop["flag_terms"]:
            if _covered(t, candidate.flag_terms):
                continue
            hit_gen = [v for v in evidence.genuine if term_in_evidence(t, v)]
            if len(hit_gen) >= 2 and len(hit_gen) >= settings.refine_flag_min_share * n_match:
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
                reason=f"{st.fp}/{st.n} verified papers with this term do not mention the dataset",
                evidence=_quotes(st.fp_papers), **base))
            n_remaining -= 1
        elif share >= settings.refine_demote_fp_share and is_safe and flags:
            diffs.append(TermDiff(
                kind="alias", term=a, action="demote", status="verified",
                reason=f"{st.fp}/{st.n} verified papers do not mention the dataset; now requires flag terms",
                evidence=_quotes(st.fp_papers), **base))
            keep_risky.append(a)
        else:
            reason = "verified mentions in full text" if share < settings.refine_demote_fp_share else (
                "high FP share but cannot be dropped/demoted (last alias or no flag terms)")
            diffs.append(TermDiff(
                kind="alias", term=a, action="keep", status="verified", reason=reason,
                evidence=_quotes(st.matched_papers), **base))
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
            reason=f"mention form in {len(hits)} matching quote(s); {c} Dimensions hits"
            + (" (risky: requires flags)" if risky else ""),
            stats={"dimensions_count": c, "matching_papers": len(hits)}, evidence=_quotes(hits)))

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
            reason=f"occurs in the quote or snippets of {len(hits)} matching paper(s)", evidence=_quotes(hits),
            stats={"matching_papers": len(hits)}))
    for e in candidate.exclude_terms:
        diffs.append(TermDiff(kind="exclude", term=e, action="keep", status="unverified", reason="unchanged"))
    for t, hits in new_excludes:
        diffs.append(TermDiff(
            kind="exclude", term=t, action="add", status="verified",
            reason=_exclude_reason(t, hits)
            + (f"; removes {impacts[t]} of the current results" if t in impacts else ""),
            evidence=_quotes(hits),
            stats={"fp_papers": len(hits), **({"removes": impacts[t]} if t in impacts else {})}))

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
    return RefineResult(candidate=refined, diffs=diffs, notes=notes)

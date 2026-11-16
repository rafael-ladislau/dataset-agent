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
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from dataset_agent.adapters.name_grounding import _PAREN_TAIL, content_tokens
from dataset_agent.adapters.term_sanitizer import sanitize_term, sanitize_terms
from dataset_agent.adapters.dimensions_phrase import same_search
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
from dataset_agent.adapters.query_heuristics import (
    detect_rhetorical_name,
    is_acronym_suffix_form,
    is_short_acronym,
)
from dataset_agent.adapters.scope_check import acronym_form, names_program
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

#: Classifications that make a name context around the dataset instead of a name of
#: it: an organization, archive, repository or host name is judged as a flag term,
#: never promoted to a standalone alias ("GAGE Facility" next to a GNSS collection).
#: The wording matches the resource map's kinds vocabulary (resource_map.KINDS).
CONTEXT_ONLY_KINDS = frozenset({"organization", "archive", "repository", "host"})
#: A bare domain ("www.example.org") is a host, never a dataset name.
_HOST_NAME = re.compile(r"(?:[A-Za-z0-9][\w'-]*\.)+[A-Za-z]{2,}")
#: A URL scheme in front of a host name ("https://www.example.org").
_HOST_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://")
#: A port after a host name ("ftp.example.org:8080").
_HOST_PORT = re.compile(r":\d+$")


def _surface_form_anchored(term: str, dataset_name: str) -> bool:
    """The name is the dataset's own surface form, or a wording of it: it denotes the
    dataset itself even when a classification says organization or repository. A
    trailing parenthetical qualifier ('Nevada Geodetic Laboratory (NGL)') is stripped
    before the check: naming the acronym the classification already saw is still the
    dataset's own name."""
    anchor = content_tokens(dataset_name)

    def anchored(t: str) -> bool:
        tokens = content_tokens(t)
        return same_search(t, dataset_name) or (bool(tokens) and tokens <= anchor)

    if anchored(term):
        return True
    stripped = _PAREN_TAIL.sub("", term).strip()
    return bool(stripped) and stripped != term and anchored(stripped)


def _context_only_kind(term: str, classified: str | None, dataset_name: str) -> str | None:
    """How the classification says the name is context around the dataset: organization,
    archive, repository or host (from the proposal's kind, or a bare-domain host in the
    prose). None for a surface-form-anchored name: the dataset's own form stays an alias."""
    if _surface_form_anchored(term, dataset_name):
        return None
    kind = (classified or "").strip().casefold()
    if kind in CONTEXT_ONLY_KINDS:
        return kind
    stripped = _HOST_SCHEME.sub("", term.strip().rstrip("/")).split("/", 1)[0]
    head = _HOST_PORT.sub("", stripped.split("?", 1)[0].split("#", 1)[0]).strip()
    if _HOST_NAME.fullmatch(head):
        return "host"
    return None


def _new_alias_entries(v: object) -> tuple[list[str], dict[str, str]]:
    """Names and the optional classification of each ``new_aliases`` entry: a string or
    a ``{"name", "kind"}`` row (the row shape the resource map uses). Names keep the
    cleaning of the plain-string path; the classification survives a cleaned name."""
    entries = v if isinstance(v, list) else []
    names: list[str] = []
    kinds: dict[str, str] = {}
    for x in entries:
        if isinstance(x, str):
            names.append(_unquote(x))
        elif isinstance(x, dict) and isinstance(x.get("name"), str):
            name = _unquote(x["name"])
            names.append(name)
            kind = x.get("kind")
            if isinstance(kind, str) and kind.strip():
                # The refine loop looks the kind up by the sanitized term, so the key
                # itself must be sanitized: a sanitizer-changed variation of the name
                # (a trailing period) would otherwise silently lose the classification.
                cleaned = sanitize_term(name)
                if cleaned is not None:
                    kinds.setdefault(" ".join(cleaned.split()).casefold(), kind.strip())
    terms = sanitize_terms(names, "refiner term")
    out = dedupe_strings_ci_preserve_order(
        [t for t in terms if len(t) > 2 and '"' not in t])[:5]
    return out, kinds


@dataclass
class RefineResult:
    candidate: QueryCandidate
    diffs: list[TermDiff] = field(default_factory=list)
    #: Patterns found across the papers (false-positive groups, fields of research).
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(d.action != "keep" for d in self.diffs)


def _quotes(verdicts: list[PubVerdict], term: str | None = None):
    return [v.as_quote(term) for v in verdicts[:_MAX_EVIDENCE]]


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
    flags and new aliases from the matching papers. Papers labelled program-only by the
    scope check are genuine and listed with the matching papers."""

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
dataset name). A flag term must be a proper name (an agency, program or product such as
"Census Bureau"), never a lower-case context word such as "state" or "survey data". Also give new
alias forms of the dataset name that appear in their quotes or snippets but are missing from the
current aliases; with each new alias give what it names (kind): dataset (the name of the dataset
or product itself), organization (an agency, institute, consortium, facility or funder), archive
or repository (a collection the data sits in), or host (a platform, portal or server that hosts
it). A name whose kind is organization, archive, repository or host is context around the
dataset, not a name of it, and is judged as a flag term.
Only use words that literally occur in the text above.

Use emit_refinement_candidates exactly once."""
    empty = {"exclude_terms": [], "flag_terms": [], "new_aliases": [], "new_alias_kinds": {},
             "groups": [], "failed": True}
    try:
        raw = agent.get_structured(prompt, EMIT_REFINEMENT_CANDIDATES)
    except Exception as exc:
        logger.warning("propose_refinements failed: %s", exc)
        return empty
    if not isinstance(raw, dict) or not raw:
        logger.warning("propose_refinements: the LLM returned no usable result")
        return empty

    def clean_list(v, cap: int) -> list[str]:
        if not isinstance(v, list):
            return []
        # Strip wrapping quotes the LLM may add; a term that still contains a double
        # quote cannot be sent to extraction (seed terms reject it), so drop it.
        terms = sanitize_terms(
            [_unquote(x) for x in v if isinstance(x, str)], "refiner term"
        )
        return dedupe_strings_ci_preserve_order(
            [t for t in terms if len(t) > 2 and '"' not in t]
        )[:cap]

    groups = []
    raw_groups = raw.get("false_positive_groups")
    for g in raw_groups if isinstance(raw_groups, list) else []:
        if not isinstance(g, dict) or not isinstance(g.get("label"), str):
            continue
        groups.append({
            "label": g["label"][:120],
            "cause_term": str(g.get("cause_term") or "")[:80],
            "papers": int(g["papers"]) if isinstance(g.get("papers"), int) else 0,
            "exclude_terms": clean_list(g.get("exclude_terms"), 5),
        })
    group_excludes = [t for g in groups[:6] for t in g["exclude_terms"]]
    new_aliases, kinds = _new_alias_entries(raw.get("new_aliases"))
    return {
        "exclude_terms": dedupe_strings_ci_preserve_order(
            clean_list(raw.get("exclude_terms"), 10) + group_excludes)[:15],
        "flag_terms": clean_list(raw.get("flag_terms"), 10),
        "new_aliases": new_aliases,
        "new_alias_kinds": kinds,
        "groups": groups[:6],
    }


def category_patterns(evidence: Evidence, min_fp: int) -> list[str]:
    """Fields of research concentrated among the false positives and rare among matches.

    Reported only (never applied): excluding a whole field is a strong, manual decision.
    """
    fps, matches = evidence.false_positives, evidence.genuine
    # Without matching papers to compare with, every field of the false positives would be
    # reported, the dataset's own field included.
    if not fps or len(matches) < min_fp:
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
        if n_match / len(matches) > 0.1:
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


#: Lower-case words allowed inside a proper name ("Bureau of Labor Statistics").
_NAME_JOINERS = frozenset({"of", "the", "and", "for", "on", "in", "to", "at", "de", "&"})


def _generic_word(term: str) -> bool:
    """Not a proper name: some word has no upper-case letter (other than a joiner such as "of").

    Flags gate risky aliases, so they must name an agency, program or product: "BLS",
    "Census Bureau", "Bureau of Labor Statistics" pass; "state", "administrative data"
    and "state UI" (a lower-case word next to an acronym) do not. A coined name with an inner
    capital ("dbGaP", "eICU", "gnomAD") passes; a digit is no exemption ("gen3", "p53")."""
    words = [w for w in term.split() if w]
    if not words:
        return True
    for w in words:
        if w.lower() in _NAME_JOINERS and len(words) > 1:
            continue
        if any(ch.isalpha() for ch in w) and not any(ch.isupper() for ch in w):
            return True
    return False


def _covered(term: str, existing: list[str]) -> bool:
    """*term* is the same Dimensions search as one of *existing*."""
    return any(same_search(term, e) for e in existing)


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
    program_names: list[str] | None = None,
) -> RefineResult:
    """Return the refined candidate plus a per-term diff with evidence and reasons.

    *impact_fn(term)*: how many current query results also contain *term* (what an
    exclude would remove). At most ``refine_max_exclude_checks`` excludes are checked per
    pass, most false-positive hits first; an exclude that removes nothing, or whose
    impact is unknown (budget reserve reached, Dimensions error), is rejected.

    *program_names* (resource map): a new alias that equals, contains or abbreviates a
    program name keeps the phase-1 guards: an acronym form or an ordinary phrase is gated;
    otherwise it routes by its count like any alias (the query is broad on the program).

    A new-alias proposal may carry a classification ("kind"); an organization, archive,
    repository or host name is context around the dataset, not a name of it, and is
    judged as a flag candidate instead (never promoted to an alias), unless it is the
    dataset's own surface form (``_surface_form_anchored``), which stays an alias.
    """
    abstract_fp = {a.lower() for a in (abstract_fp_aliases or set())}
    aliases = candidate.all_aliases()
    safe_l = {a.lower() for a in candidate.safe}
    diffs: list[TermDiff] = []
    enough = evidence.n_evaluated >= settings.fulltext_min_evidence

    stats = attribute_terms(evidence, aliases)
    flags_stats = attribute_terms(evidence, list(candidate.flag_terms), partial=False)
    # Papers labelled program-only stay genuine: never an exclude source, never against a name.
    fps = evidence.false_positives

    # --- LLM proposals, validated against the quotes ------------------------------
    rejected: dict[str, list[str]] = {"exclude": [], "flag": [], "alias": []}

    def reject(kind: str, term: str, why: str) -> None:
        logger.info("refine: %s candidate %r rejected: %s", kind, term, why)
        rejected[kind].append(f"{term} ({why})")

    new_excludes: list[tuple[str, list[PubVerdict]]] = []
    new_flags: list[tuple[str, list[PubVerdict]]] = []
    new_aliases: list[tuple[str, list[PubVerdict]]] = []
    # Alias proposals whose classification says they are context around the dataset
    # (organization, archive, repository, host): never aliases; judged on the flag path.
    routed_flags: list[str] = []
    routed_kind: dict[str, str] = {}
    notes: list[str] = []
    already: list[str] = []
    impacts: dict[str, int] = {}
    # Safe aliases that an exclude drawn from false positives overlaps ("Binex Co" next to
    # "BINEX"): the alias has another meaning, mapped to the exclude that showed it (seen in
    # at least refine_min_exclude_papers false positives, none of the genuine papers).
    overlap_demote: dict[str, str] = {}
    if enough:
        notes.extend(category_patterns(evidence, settings.refine_category_min_fp_papers))
    if enough and agent is not None and (evidence.genuine or fps):
        prop = propose_refinements(
            agent,
            dataset_name=dataset_name,
            genuine=evidence.genuine,
            false_positives=fps,
            aliases=aliases,
            flag_terms=candidate.flag_terms,
            exclude_terms=candidate.exclude_terms,
        )
        if prop.get("failed"):
            notes.append(
                "refiner LLM returned no usable result: no exclude, flag or alias was proposed"
            )
        for g in prop.get("groups") or []:
            verified = (
                sum(1 for v in fps if term_in(g["cause_term"], v))
                if g["cause_term"] else 0
            )
            notes.append(
                f"full-text pattern: false-positive group '{g['label']}' "
                f"(~{g['papers']} per LLM; cause term '{g['cause_term']}' occurs in {verified} false positive(s))"
                + (f"; candidate excludes {g['exclude_terms']}" if g["exclude_terms"] else "")
            )
        candidates_ex: list[tuple[str, list[PubVerdict]]] = []
        for t in prop["exclude_terms"]:
            if _covered(t, candidate.exclude_terms):
                already.append(t)
                continue
            # NOT ("term") runs over all of full_data: an exclude inside or containing a
            # kept alias, or inside a flag, would silently drop the query's own matches.
            overlapped = [a for a in aliases if _phrase_in(t, a) or _phrase_in(a, t)]
            if overlapped or any(_phrase_in(t, f) for f in candidate.flag_terms):
                reject("exclude", t, "overlaps an alias or flag")
                n_fp_overlap = sum(1 for v in fps if term_in(t, v))
                if n_fp_overlap >= settings.refine_min_exclude_papers and not any(
                    term_in(t, v) for v in evidence.genuine
                ):
                    for a in overlapped:
                        if a.lower() in safe_l and a.lower() != t.lower():
                            overlap_demote.setdefault(a, t)
                continue
            hit_fp = [v for v in fps if term_in(t, v)]
            # Over-matching the genuine side is the safe direction here: an exclude
            # is blocked when the term, or its singular, occurs in a genuine paper.
            forms = [t, *singular_variants(t)]
            hit_gen = [v for v in evidence.genuine if any(term_in(f, v) for f in forms)]
            if hit_gen:
                reject("exclude", t, f"in {len(hit_gen)} matching paper(s)")
                continue
            if len(hit_fp) < settings.refine_min_exclude_papers:
                reject("exclude", t, f"in {len(hit_fp)} false positive(s), need {settings.refine_min_exclude_papers}")
                continue
            candidates_ex.append((t, hit_fp))
        # Each impact check is a Dimensions call: check the strongest candidates only.
        candidates_ex.sort(key=lambda th: -len(th[1]))
        for t, _ in candidates_ex[settings.refine_max_exclude_checks:]:
            reject("exclude", t, "over the impact-check cap")
        for t, hit_fp in candidates_ex[: settings.refine_max_exclude_checks]:
            if impact_fn is not None:
                try:
                    removed = int(impact_fn(t))
                except Exception as exc:  # budget reserve reached or Dimensions error
                    reject("exclude", t, f"impact unknown: {exc}")
                    continue
                if removed <= 0:
                    reject("exclude", t, "removes no current result")
                    continue
                impacts[t] = removed
            new_excludes.append((t, hit_fp))
        # Aliases are validated before flags: a flag equal to a current or new alias
        # would make the gate satisfy itself (("X") AND (... OR "X")).
        short_aliases: list[str] = []
        for t in prop["new_aliases"]:
            if _covered(t, aliases):
                already.append(t)
                continue
            if _covered(t, candidate.flag_terms):
                reject("alias", t, "already a flag term")
                continue
            # "UI data", "WR dataset": an acronym plus a generic suffix still matches the
            # acronym's other meanings; the refiner never adds one.
            if is_acronym_suffix_form(t):
                reject("alias", t, "short acronym plus generic suffix")
                continue
            # "NOTA": a bare short acronym matches its other meanings even behind flags,
            # and grounding already refuses one; the refiner never adds one either.
            if is_short_acronym(t):
                reject("alias", t, "bare short acronym")
                short_aliases.append(t)
                continue
            # "All of Us": a phrase of ultra-common words matches ordinary prose.
            if detect_rhetorical_name(t):
                reject("alias", t, "rhetorical phrase")
                continue
            # A term the LLM lists both as an alias and as a flag is context around the
            # dataset (an organization, a facility), not a name: it is judged as a flag only.
            if _covered(t, prop["flag_terms"]):
                reject("alias", t, "also proposed as a flag term")
                continue
            # A name classified organization, archive, repository or host (by the
            # proposal's kind, or a bare host domain) is context around the dataset, not
            # a name of it: never promoted to an alias, it is judged on the flag path,
            # like the flag gauntlet's own evidence and count rules.
            ctx_kind = _context_only_kind(
                t, prop.get("new_alias_kinds", {}).get(" ".join(t.split()).casefold()),
                dataset_name)
            if ctx_kind:
                logger.info(
                    "refine: alias candidate %r classified %s, not a dataset name: judged "
                    "as a flag term", t, ctx_kind)
                routed_flags.append(t)
                routed_kind[t.casefold()] = ctx_kind
                continue
            # A new alias must occur literally (no plural folding) in a genuine quote.
            hit_gen = [v for v in evidence.genuine if term_in_quote(t, v, exact=True)]
            if hit_gen:
                new_aliases.append((t, hit_gen))
            else:
                reject("alias", t, "not literally in a matching quote")
        # Flags need at least two matching papers and a minimum share of them, from
        # quotes or full-text snippets (titles never count), and must be specific.
        # Alias proposals routed here for their classification join the same gauntlet.
        flag_props = list(prop["flag_terms"])
        for t in routed_flags:
            if not _covered(t, prop["flag_terms"]):
                flag_props.append(t)
        n_match = len(evidence.genuine)
        flag_checks = 0
        flag_hits: dict[str, int] = {}
        for t in flag_props:
            if _covered(t, candidate.flag_terms):
                already.append(t)
                continue
            # A short acronym refused as an alias must not come back as a gate term.
            if _covered(t, aliases) or _covered(t, [a for a, _ in new_aliases] + short_aliases):
                reject("flag", t, "already an alias")
                continue
            # "PMI": a bare short acronym as a gate term matches its other meanings.
            if (
                settings.optimize_drop_short_acronym_flags
                and is_short_acronym(t)
                and candidate.flag_terms
            ):
                reject("flag", t, "bare short acronym")
                continue
            # A flag that is a phrase of a flag-gated alias widens nothing and lets the alias
            # satisfy its own gate. Aliases accepted as risky this round are checked below.
            if any(_phrase_in(t, a, plural_ok=False) for a in candidate.risky):
                reject("flag", t, "contained in a flag-gated alias")
                continue
            # "EarthScope" next to "EarthScope Consortium" only widens the gate.
            if any(_phrase_in(t, f, plural_ok=False) for f in candidate.flag_terms):
                reject("flag", t, "broader form of an existing flag")
                continue
            hit_gen = [v for v in evidence.genuine if term_in_evidence(t, v)]
            if len(hit_gen) < 2 or len(hit_gen) < settings.refine_flag_min_share * n_match:
                reject("flag", t, f"in {len(hit_gen)}/{n_match} matching paper(s)")
                continue
            if _generic_word(t):
                reject("flag", t, "not a proper name (lower-case word)")
                continue
            if count_fn is not None:
                if flag_checks >= settings.refine_max_flag_checks:
                    reject("flag", t, "over the flag-check cap")
                    continue
                flag_checks += 1
                try:
                    c = int(count_fn(t))
                except Exception as exc:  # budget exhausted or Dimensions error
                    reject("flag", t, f"hit count unknown: {exc}")
                    continue
                max_hits = settings.optimize_flag_max_hits  # same cap as research flags; 0 = off
                if c <= 0 or (max_hits and c > max_hits):
                    reject("flag", t, f"{c} Dimensions hits (max {max_hits})")
                    continue
                flag_hits[t] = c
            new_flags.append((t, hit_gen))
    if already:
        logger.info("refine: proposals skipped, already in the query: %s", already)
        notes.append("proposals skipped, already in the query: " + "; ".join(already[:8]))

    flags = list(candidate.flag_terms) + [t for t, _ in new_flags]

    # --- alias decisions (deterministic) -------------------------------------------
    keep_safe: list[str] = []
    keep_risky: list[str] = []
    n_remaining = len(aliases)
    for a in aliases:
        st = stats[a]
        is_safe = a.lower() in safe_l
        base = {"stats": st.as_dict()}
        if a in overlap_demote and is_safe and flags:
            diffs.append(TermDiff(
                kind="alias", term=a, action="demote", status="verified",
                reason=(
                    f"false positives mention {overlap_demote[a]!r}, another meaning of this "
                    "alias, and no matching paper does; now requires flag terms"
                ),
                evidence=_quotes(
                    [v for v in fps if term_in(overlap_demote[a], v)],
                    overlap_demote[a],
                ), **base))
            keep_risky.append(a)
            continue
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
                evidence=_quotes(st.fp_papers, a), **base))
            n_remaining -= 1
        elif share >= settings.refine_demote_fp_share and is_safe and flags:
            diffs.append(TermDiff(
                kind="alias", term=a, action="demote", status="verified",
                reason=f"{st.fp}/{st.n} verified papers do not mention the dataset; now requires flag terms",
                evidence=_quotes(st.fp_papers, a), **base))
            keep_risky.append(a)
        else:
            reason = "verified mentions in full text" if share < settings.refine_demote_fp_share else (
                "high FP share but cannot be dropped/demoted (last alias or no flag terms)")
            diffs.append(TermDiff(
                kind="alias", term=a, action="keep", status="verified", reason=reason,
                evidence=_quotes(st.matched_papers, a), **base))
            (keep_safe if is_safe else keep_risky).append(a)

    # --- new aliases (re-probe Dimensions counts) ------------------------------------
    added_aliases = 0
    for t, hits in new_aliases:
        if added_aliases >= settings.refine_max_new_aliases or count_fn is None:
            reject("alias", t, "over the new-alias cap" if count_fn is not None else "no count available")
            continue
        try:
            c = int(count_fn(t))
        except Exception as exc:  # budget exhausted or Dimensions error
            reject("alias", t, f"hit count unknown: {exc}")
            continue
        if c <= 0:
            reject("alias", t, "0 Dimensions hits")
            continue
        risky = c > 10 * max(1, reference_count)
        prog = names_program(t, program_names or [])
        gated = bool(prog) and (acronym_form(t) or detect_rhetorical_name(t))
        if gated and not risky:
            risky = True
            logger.info("refine: alias %r names the program %r (acronym or ordinary phrase): "
                        "gated", t, prog)
        if risky and not flags:
            reject("alias", t, f"{c} Dimensions hits: risky and there are no flag terms")
            continue
        (keep_risky if risky else keep_safe).append(t)
        added_aliases += 1
        logger.info("refine: alias %r accepted: %d Dimensions hits, %s", t, c, "risky" if risky else "safe")
        diffs.append(TermDiff(
            kind="alias", term=t, action="add", status="verified",
            reason=f"mention form in {len(hits)} matching quote(s); {c} Dimensions hits"
            + (" (names the program: acronym or ordinary phrase, gated)" if gated else " (risky: requires flags)" if risky
               else ""),
            stats={"dimensions_count": c, "matching_papers": len(hits)}, evidence=_quotes(hits, t)))

    # Flags inside an alias that became flag-gated in this round (a new risky alias).
    new_risky = [t for t in keep_risky if t not in candidate.risky]
    if new_risky and new_flags:
        kept_flags = []
        for t, hits in new_flags:
            if any(_phrase_in(t, a, plural_ok=False) for a in new_risky):
                reject("flag", t, "contained in a flag-gated alias")
                flag_hits.pop(t, None)
            else:
                kept_flags.append((t, hits))
        new_flags = kept_flags
        flags = list(candidate.flag_terms) + [t for t, _ in new_flags]
        if not flags:
            # The removed flags were the only gate: a new risky alias cannot stay. A safe
            # alias demoted this round goes back to safe (its demotion needed a flag).
            safe_now = {a.lower() for a in candidate.safe}
            for t in new_risky:
                keep_risky.remove(t)
                if t.lower() in safe_now:
                    keep_safe.append(t)
                    diffs = [
                        d.model_copy(update={
                            "action": "keep",
                            "reason": "demotion undone: its only new flag was contained in it, "
                                      "so no flag terms are left to gate it",
                        })
                        if d.kind == "alias" and d.action == "demote" and d.term == t else d
                        for d in diffs
                    ]
                    continue
                diffs = [d for d in diffs
                         if not (d.kind == "alias" and d.action == "add" and d.term == t)]
                reject("alias", t, "risky and there are no flag terms")

    for d in diffs:
        if d.kind == "alias" and d.action in ("drop", "demote"):
            logger.info("refine: alias %r %s: %s", d.term, d.action, d.reason)
    for t, _ in new_flags:
        logger.info("refine: flag %r accepted: %s", t,
                    f"{flag_hits[t]} Dimensions hits" if t in flag_hits else "count not checked")
    for t, _ in new_excludes:
        logger.info("refine: exclude %r accepted: %s", t,
                    f"removes {impacts[t]} results" if t in impacts else "impact not checked")
    added_parts = [
        f"{label} [{', '.join(items)}]"
        for label, items in (
            ("aliases", [d.term for d in diffs if d.kind == "alias" and d.action == "add"]),
            ("flags", [t for t, _ in new_flags]),
            ("excludes", [t for t, _ in new_excludes]),
        )
        if items
    ]
    if added_parts:
        notes.append("refiner added: " + "; ".join(added_parts))

    for kind, items in rejected.items():
        if items:
            notes.append(f"rejected {kind} candidates: " + "; ".join(items[:8])
                         + (f"; +{len(items) - 8} more" if len(items) > 8 else ""))

    # --- flags / excludes --------------------------------------------------------------
    for f in candidate.flag_terms:
        st = flags_stats[f]
        diffs.append(TermDiff(
            kind="flag", term=f, action="keep",
            status="verified" if enough and st.n >= settings.refine_min_alias_papers else "unverified",
            reason="unchanged", stats=st.as_dict()))
    for t, hits in new_flags:
        routed = routed_kind.get(t.casefold())
        diffs.append(TermDiff(
            kind="flag", term=t, action="add", status="verified",
            reason=f"occurs in the quote or snippets of {len(hits)} matching paper(s)"
            + (f"; classified {routed}: context around the dataset, not a name of it"
               if routed else "")
            + (f"; {flag_hits[t]} Dimensions hits" if t in flag_hits else ""), evidence=_quotes(hits, t),
            stats={"matching_papers": len(hits), **({"dimensions_count": flag_hits[t]} if t in flag_hits else {})}))
    for e in candidate.exclude_terms:
        diffs.append(TermDiff(kind="exclude", term=e, action="keep", status="unverified", reason="unchanged"))
    for t, hits in new_excludes:
        diffs.append(TermDiff(
            kind="exclude", term=t, action="add", status="verified",
            reason=_exclude_reason(t, hits)
            + (f"; removes {impacts[t]} of the current results" if t in impacts else ""),
            evidence=_quotes(hits, t),
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

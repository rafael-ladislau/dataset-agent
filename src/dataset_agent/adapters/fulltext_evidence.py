"""Turn extraction ``llm-results`` into full-text evidence for a candidate query.

Settled rules (the probe measures the *search query*, so a paper that mentions the
dataset is a correct match even when it does not use it):

* match (``genuine``) = the dataset is mentioned (``is_mentioned`` or ``is_used``);
  false positive = the dataset is not mentioned at all (``not_mentioned``);
* usage (``is_used``) is reported separately and never decides a term change;
* publications without delivered full text, not analyzed, or with no verdict at
  all are excluded from the full-text metrics (they only lower OA coverage);
* a term is attributed to a paper when it appears in the paper's evidence quote,
  title or full-text snippets of the query's terms (``term_contexts``);
* a dataset name found only in the paper's reference list is not a mention: when every
  non-flag snippet is in ``references`` and no flag snippet lies outside it, the
  verdict is ``not_mentioned`` whatever the validator said (see _references_only).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

from dataset_agent.adapters.extraction_client import is_attempted, is_delivered
from dataset_agent.adapters.literature import _stem_text
from dataset_agent.adapters.query_heuristics import head_acronym
from dataset_agent.domain.proposals import EvidenceQuote, IterationMetrics


def _fold(text: str) -> str:
    """Normalize like the extraction side: NFKC, lower case, and every non-alphanumeric
    character (underscore included) becomes a space. ``literature._stem_text`` keeps
    ``_`` and phase 1 depends on that, so the fold is local to the full-text evidence."""
    return _stem_text(unicodedata.normalize("NFKC", text or "").replace("_", " "))


@dataclass
class PubVerdict:
    publication_id: str
    doi: str | None
    title: str
    citation_count: int
    verdict: str  # used | mentioned | not_mentioned
    quote: str
    #: Snippets of the query's terms in the full text (``term``, ``role``, ``part``, ``text``).
    contexts: list[dict] = field(default_factory=list)
    #: Dimensions field-of-research names.
    categories: list[str] = field(default_factory=list)
    #: The validator's verdict when a deterministic rule replaced it, else None.
    overridden_from: str | None = None

    @property
    def genuine(self) -> bool:
        """Correct match for the query: the paper mentions the dataset (used or not)."""
        return self.verdict in ("used", "mentioned")

    @property
    def used(self) -> bool:
        return self.verdict == "used"

    def evidence_quote_text(self, term: str | None = None) -> str:
        """The validator quote, or when it is empty the snippet for *term* (else the
        first snippet), so exclude/alias evidence is never empty."""
        if self.quote.strip():
            return self.quote
        snippets = [str(c.get("text") or "") for c in self.contexts if c.get("text")]
        if term:
            for sn in snippets:
                if _phrase_in(term, sn):
                    return sn
        return snippets[0] if snippets else self.quote

    def as_quote(self, term: str | None = None) -> EvidenceQuote:
        return EvidenceQuote(
            publication_id=self.publication_id,
            doi=self.doi,
            title=self.title or None,
            quote=self.evidence_quote_text(term),
            verdict=self.verdict,  # type: ignore[arg-type]
        )


@dataclass
class Evidence:
    verdicts: list[PubVerdict] = field(default_factory=list)
    #: Publications with a full-text outcome (delivered or a real miss).
    n_sampled: int = 0
    n_delivered: int = 0
    target: int | None = None
    batches: int = 0
    stop_reason: str | None = None
    budget_exhausted: bool = False

    @property
    def shortfall_note(self) -> str | None:
        """Human note when fewer than the target papers had full text, else None."""
        if self.target is None or self.n_delivered >= self.target:
            return None
        return (
            f"{self.n_delivered}/{self.target} papers with full text "
            f"({self.stop_reason or 'stopped'}; {self.n_sampled} attempted in "
            f"{self.batches} batch(es){'; OpenAlex budget exhausted' if self.budget_exhausted else ''})"
        )

    @property
    def n_evaluated(self) -> int:
        return len(self.verdicts)

    @property
    def genuine(self) -> list[PubVerdict]:
        return [v for v in self.verdicts if v.genuine]

    @property
    def false_positives(self) -> list[PubVerdict]:
        return [v for v in self.verdicts if not v.genuine]

    def metrics(self, iteration: int, for_clause: str, abstract_fp_pct: float | None) -> IterationMetrics:
        n = self.n_evaluated
        used = sum(1 for v in self.verdicts if v.used)
        matched = len(self.genuine)
        return IterationMetrics(
            iteration=iteration,
            for_clause=for_clause,
            n_publications=self.n_sampled,
            n_fulltext_delivered=self.n_delivered,
            n_evaluated=n,
            fulltext_precision=(matched / n) if n else None,
            usage_rate=(used / matched) if matched else None,
            mention_rate=(matched / n) if n else None,
            used_rate=(used / n) if n else None,
            oa_coverage=(self.n_delivered / self.n_sampled) if self.n_sampled else None,
            abstract_fp_rate_pct=abstract_fp_pct,
            fulltext_target=self.target,
            fulltext_batches=self.batches,
            fulltext_stop_reason=self.stop_reason,
            fulltext_budget_exhausted=self.budget_exhausted,
            papers=[(v.publication_id, v.verdict) for v in self.verdicts],
        )


def _verdict_for(pub: dict[str, Any]) -> tuple[str, str] | None:
    """Collapse a publication's validated datasets into (verdict, quote); None = no verdict."""
    entries = [e for e in (pub.get("validated_datasets") or []) if isinstance(e, dict)]
    if not entries:
        # Analyzed, but the LLM found no dataset mention at all.
        return ("not_mentioned", "")
    if all(e.get("is_used") is None and e.get("is_mentioned") is None for e in entries):
        return None
    used = [e for e in entries if e.get("is_used") is True]
    if used:
        return ("used", str(used[0].get("text_quote") or ""))
    mentioned = [e for e in entries if e.get("is_mentioned") is True]
    if mentioned:
        return ("mentioned", str(mentioned[0].get("text_quote") or ""))
    return ("not_mentioned", str(entries[0].get("text_quote") or ""))


def _references_only(contexts: list[dict]) -> bool:
    """True when the paper names the dataset only in its reference list: at least one
    non-flag snippet, all non-flag snippets in ``references``, and no flag snippet
    outside ``references``. A cited title is not a mention."""
    names = [c for c in contexts if c.get("role") != "flag"]
    if not names or any(c.get("part") != "references" for c in names):
        return False
    return all(c.get("part") == "references" for c in contexts if c.get("role") == "flag")


def build_evidence(results: dict[str, Any]) -> Evidence:
    """Build :class:`Evidence` from a ``GET /pipelines/{id}/llm-results`` body.

    The sample is every publication that has a full-text outcome; publications never
    sent for full text are not part of it.
    """
    pubs = [p for p in (results.get("publications") or []) if isinstance(p, dict)]
    pubs.sort(key=lambda p: -(p.get("citation_count") or 0))
    sample = [p for p in pubs if is_attempted(p)]
    ev = Evidence(n_sampled=len(sample))
    for p in sample:
        if not is_delivered(p):
            continue
        ev.n_delivered += 1
        if not p.get("analyzed") or p.get("validation_failed"):
            # Unanalyzed, or validation errored: no verdict (never not_mentioned).
            continue
        v = _verdict_for(p)
        if v is None:
            continue
        contexts = [c for c in (p.get("term_contexts") or []) if isinstance(c, dict)]
        overridden_from = None
        if v[0] != "not_mentioned" and _references_only(contexts):
            overridden_from, v = v[0], ("not_mentioned", v[1])
        ev.verdicts.append(
            PubVerdict(
                publication_id=str(p.get("publication_id") or ""),
                doi=p.get("doi"),
                title=str(p.get("title") or ""),
                citation_count=int(p.get("citation_count") or 0),
                verdict=v[0],
                quote=v[1],
                contexts=contexts,
                categories=[c for c in (p.get("category_for") or []) if isinstance(c, str)],
                overridden_from=overridden_from,
            )
        )
    return ev


def _tokens(text: str) -> list[str]:
    return _fold(text).split()


def _token_matches(term_token: str, text_token: str, plural_ok: bool) -> bool:
    """The term token itself, or (when allowed) its plural in the text. The term is
    never altered, so ``NHIS`` never matches ``NHI``; tokens of 3 letters or fewer
    (``ACE``, ``MEP``) only match exactly."""
    if text_token == term_token:
        return True
    if not plural_ok or len(term_token) <= 3:
        return False
    plurals = {term_token + "s", term_token + "es"}
    if term_token.endswith("y") and term_token[-2] not in "aeiou":
        plurals.add(term_token[:-1] + "ies")  # study -> studies
    return text_token in plurals


def _in_tokens(want: list[str], have: list[str], plural_ok: bool) -> bool:
    n = len(want)
    return any(
        all(_token_matches(w, h, plural_ok) for w, h in zip(want, have[i : i + n]))
        for i in range(len(have) - n + 1)
    )


def _phrase_in(term: str, text: str, *, plural_ok: bool = True) -> bool:
    """Whole-word phrase match on normalized tokens: ``ACE`` never matches ``placebo``,
    and ``Family Survey`` matches ``family surveys`` (text-side plural only)."""
    want = _tokens(term)
    return bool(want) and _in_tokens(want, _tokens(text), plural_ok)


def _in_segments(term: str, texts: list[str], snippets: list[str], plural_ok: bool = True) -> bool:
    """Match inside any of *texts* (whole) or *snippets*; a phrase never spans two segments.

    Extraction cuts snippets on word boundaries, so the tokens next to an ellipsis are
    whole words."""
    want = _tokens(term)
    if not want:
        return False
    return any(_in_tokens(want, _tokens(s), plural_ok) for s in [*texts, *snippets])


def _snippets(verdict: PubVerdict, *, with_title: bool) -> list[str]:
    return [
        str(c.get("text") or "")
        for c in verdict.contexts
        if c.get("text") and (with_title or c.get("part") != "title")
    ]


def term_in(term: str, verdict: PubVerdict) -> bool:
    """Quote, title or a full-text snippet of the query's terms."""
    return _in_segments(term, [verdict.quote, verdict.title], _snippets(verdict, with_title=True))


def term_in_evidence(term: str, verdict: PubVerdict) -> bool:
    """Quote or full-text snippets; the title never counts (neither the paper title nor
    a snippet whose part is ``title``)."""
    return _in_segments(term, [verdict.quote], _snippets(verdict, with_title=False))


def singular_variants(term: str) -> list[str]:
    """Every plausible singular of the term's last word (len > 3): drop ``s``, drop
    ``es``, ``ies``->``y``, ``es``->``is``. Over-generating is deliberate: this is only
    for checks where over-matching is the safe direction (the exclude guard)."""
    words = term.split()
    if not words:
        return []
    last = words[-1]
    low = last.lower()
    stems: list[str] = []
    if len(low) > 3 and low.endswith("s") and not low.endswith("ss"):
        stems.append(last[:-1])
    if len(low) > 4 and low.endswith("es"):
        stems += [last[:-2], last[:-2] + "is"]
    if len(low) > 4 and low.endswith("ies"):
        stems.append(last[:-3] + "y")
    return [" ".join([*words[:-1], s]) for s in dict.fromkeys(stems)]


def term_in_quote(term: str, verdict: PubVerdict, *, exact: bool = False) -> bool:
    """Quote-only match (the title never counts). ``exact`` disables the text-side
    plural, for checks that need the literal form (new aliases)."""
    return _in_segments(term, [verdict.quote], [], plural_ok=not exact)


@dataclass
class TermStats:
    """Per-term counts. ``matched`` = papers that mention the dataset (correct
    matches); ``used`` is the subset that uses it; ``fp`` = papers that do not
    mention it at all."""

    term: str
    matched: int = 0
    used: int = 0
    fp: int = 0
    matched_papers: list[PubVerdict] = field(default_factory=list)
    fp_papers: list[PubVerdict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.matched + self.fp

    @property
    def fp_share(self) -> float:
        return self.fp / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"papers": self.n, "matched": self.matched, "used": self.used, "fp": self.fp,
                "fp_share": round(self.fp_share, 3)}


def _head(term: str) -> str:
    words = term.split()
    return words[0].casefold() if len(words) >= 2 else ""


def _alias_head(term: str) -> str:
    """Head of an alias that may own a partial snippet: only a real leading acronym."""
    h = head_acronym(term)
    return h.casefold() if h else ""


def _partial_owners(terms: list[str], verdict: PubVerdict) -> set[str]:
    """Aliases a ``match=partial`` snippet is attributed to.

    Extraction searches each leading acronym once and tags the snippet with the first
    alias that has it, so the tag may name a precise alias ("WR Longitudinal Panel")
    rather than the ambiguous one ("WR dataset"). No alias occurs in the paper's text,
    so the snippet goes to the alias(es) with that acronym and the fewest words: the
    shortest phrase is the one most likely to match by accident.
    """
    owners: set[str] = set()
    for c in verdict.contexts:
        if c.get("match") != "partial":
            continue
        head = _head(str(c.get("term") or ""))
        sharing = [t for t in terms if head and _alias_head(t) == head]
        if not sharing:
            continue
        fewest = min(len(t.split()) for t in sharing)
        owners.update(t for t in sharing if len(t.split()) == fewest)
    return owners


def attribute_terms(evidence: Evidence, terms: list[str], *, partial: bool = True) -> dict[str, TermStats]:
    """Per-term attribution: how many verified papers carry each term in quote, title or
    snippets, or (aliases only, ``partial=True``) a ``partial`` leading-acronym snippet
    (see _partial_owners)."""
    out = {t: TermStats(t) for t in terms}
    for v in evidence.verdicts:
        owners = _partial_owners(terms, v) if partial else set()
        for t in terms:
            if term_in(t, v) or t in owners:
                st = out[t]
                if v.genuine:
                    st.matched += 1
                    st.matched_papers.append(v)
                    if v.used:
                        st.used += 1
                else:
                    st.fp += 1
                    st.fp_papers.append(v)
    return out

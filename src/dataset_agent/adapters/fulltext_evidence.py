"""Turn extraction ``llm-results`` into full-text evidence for a candidate query.

Settled rules:

* genuine paper = ``is_used == True``; false positive = ``is_used == False``
  (``is_mentioned == False`` is also a false positive);
* publications without delivered full text, not analyzed, or with no verdict at
  all are excluded from the full-text metrics (they only lower OA coverage);
* a term is attributed to a paper when its stem appears in the paper's evidence
  quote or title.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

from dataset_agent.adapters.literature import _stem_text
from dataset_agent.domain.proposals import EvidenceQuote, IterationMetrics


@dataclass
class PubVerdict:
    publication_id: str
    doi: str | None
    title: str
    citation_count: int
    verdict: str  # used | mentioned | not_mentioned
    quote: str

    @property
    def genuine(self) -> bool:
        return self.verdict == "used"

    @property
    def text(self) -> str:
        return _stem_text(f"{self.quote} {self.title}")

    @property
    def quote_text(self) -> str:
        return _stem_text(self.quote)

    def as_quote(self) -> EvidenceQuote:
        return EvidenceQuote(
            publication_id=self.publication_id,
            doi=self.doi,
            title=self.title or None,
            quote=self.quote,
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
        used = len(self.genuine)
        mentioned = sum(1 for v in self.verdicts if v.verdict in ("used", "mentioned"))
        return IterationMetrics(
            iteration=iteration,
            for_clause=for_clause,
            n_publications=self.n_sampled,
            n_fulltext_delivered=self.n_delivered,
            n_evaluated=n,
            fulltext_precision=(used / n) if n else None,
            usage_rate=(used / mentioned) if mentioned else None,
            mention_rate=(mentioned / n) if n else None,
            oa_coverage=(self.n_delivered / self.n_sampled) if self.n_sampled else None,
            abstract_fp_rate_pct=abstract_fp_pct,
            fulltext_target=self.target,
            fulltext_batches=self.batches,
            fulltext_stop_reason=self.stop_reason,
            fulltext_budget_exhausted=self.budget_exhausted,
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


def _attempted(pub: dict[str, Any]) -> bool:
    ft = pub.get("fulltext") or {}
    return ft.get("status") == "delivered" or (bool(ft) and ft.get("reason") != "not_attempted")


def build_evidence(results: dict[str, Any], top_n: int | None = None) -> Evidence:
    """Build :class:`Evidence` from a ``GET /pipelines/{id}/llm-results`` body.

    With *top_n* ``None`` (the top-up probe), the sample is every publication that has a
    full-text outcome; publications never sent for full text are not part of it. An
    explicit *top_n* keeps the legacy flat top-N sample.
    """
    pubs = [p for p in (results.get("publications") or []) if isinstance(p, dict)]
    pubs.sort(key=lambda p: -(p.get("citation_count") or 0))
    if top_n is None:
        sample = [p for p in pubs if _attempted(p)]
    else:
        sample = pubs[: max(1, int(top_n))]
    ev = Evidence(n_sampled=len(sample))
    for p in sample:
        ft = p.get("fulltext") or {}
        if ft.get("status") != "delivered":
            continue
        ev.n_delivered += 1
        if not p.get("analyzed") or p.get("validation_failed"):
            # Unanalyzed, or validation errored: no verdict (never not_mentioned).
            continue
        v = _verdict_for(p)
        if v is None:
            continue
        ev.verdicts.append(
            PubVerdict(
                publication_id=str(p.get("publication_id") or ""),
                doi=p.get("doi"),
                title=str(p.get("title") or ""),
                citation_count=int(p.get("citation_count") or 0),
                verdict=v[0],
                quote=v[1],
            )
        )
    return ev


def _tokens(text: str) -> list[str]:
    return _stem_text(unicodedata.normalize("NFKC", text or "")).split()


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


def _phrase_in(term: str, text: str, *, plural_ok: bool = True) -> bool:
    """Whole-word phrase match on normalized tokens: ``ACE`` never matches ``placebo``,
    and ``Family Survey`` matches ``family surveys`` (text-side plural only)."""
    want = _tokens(term)
    have = _tokens(text)
    if not want:
        return False
    n = len(want)
    return any(
        all(_token_matches(w, h, plural_ok) for w, h in zip(want, have[i : i + n]))
        for i in range(len(have) - n + 1)
    )


def term_in(term: str, verdict: PubVerdict) -> bool:
    return _phrase_in(term, f"{verdict.quote} {verdict.title}")


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
    return _phrase_in(term, verdict.quote, plural_ok=not exact)


@dataclass
class TermStats:
    term: str
    used: int = 0
    fp: int = 0
    mentioned_only: int = 0
    used_papers: list[PubVerdict] = field(default_factory=list)
    fp_papers: list[PubVerdict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.used + self.fp

    @property
    def fp_share(self) -> float:
        return self.fp / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"papers": self.n, "used": self.used, "fp": self.fp, "fp_share": round(self.fp_share, 3)}


def attribute_terms(evidence: Evidence, terms: list[str]) -> dict[str, TermStats]:
    """Per-term attribution: how many verified papers carry each term in quote/title."""
    out = {t: TermStats(t) for t in terms}
    for v in evidence.verdicts:
        for t in terms:
            if term_in(t, v):
                st = out[t]
                if v.genuine:
                    st.used += 1
                    st.used_papers.append(v)
                else:
                    st.fp += 1
                    st.fp_papers.append(v)
                    if v.verdict == "mentioned":
                        st.mentioned_only += 1
    return out

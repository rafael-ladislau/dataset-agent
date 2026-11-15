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
* a dataset name found only in the paper's reference list is not a mention. When the
  served row carries ``bibliography_excluded`` True the pipeline cut the reference list
  before the judge read the paper and the judge's verdict stands. Otherwise, when every
  non-flag snippet is in ``references``, no flag snippet lies outside it, and the
  validator's quote is empty or lies inside one references snippet with more words than
  any context term it matches, the verdict is
  ``not_mentioned`` (see references_only_override); anything else keeps the verdict.

AI models (body ``seed_type == "ai_model"``, model task v2 / v2.1) read
``validated_ai_models`` rows instead, with these differences (owner decision: a model name
found only in the reference list IS a mention):

* no references-only override: ``overridden_from`` stays None;
* a row with ``is_mentioned`` False is never a mention or a use, whatever its
  ``is_used`` or ``usage_type`` say (such a row with either set is counted in
  ``n_incoherent``); with ``is_mentioned`` None, ``is_used`` True or
  ``usage_type == "cited"`` makes it a mention;
* *cited only* (``PubVerdict.cited_only``, genuine papers only) = the model's names occur
  only in the reference list. With at least one non-flag name snippet: true when every
  one is in ``references``, false when any is in the title, abstract or body (flag
  snippets ignored). With none, the served ``mention_location`` (v2.1) decides
  (``references_only`` true, ``text`` false); otherwise it is unknown (None).
  ``usage_type`` never decides it; a genuine paper with ``usage_type == "cited"`` is
  reported apart as *not run* (discussed, not run). Shares divide by genuine papers;
* an empty row list, or only ``validation_error`` / ``processing_error`` rows, is no
  verdict (never ``not_mentioned``): the validator emits one row per seed.
"""

from __future__ import annotations

import re
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


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of k / n."""
    import math

    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


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
    #: AI models only: the deciding row's ``usage_type`` / ``match_type``.
    usage_type: str | None = None
    match_type: str | None = None
    #: AI models only: the names occur only in the reference list (True), elsewhere too
    #: (False), or unknown (None); see the module rule. Read through ``cited_only``.
    names_in_references_only: bool | None = False
    #: AI models only (contract v2.1): ``text`` | ``references_only`` | ``unknown`` as
    #: served, and the model name or alias the verdict rests on. None when not served.
    mention_location: str | None = None
    matched_name: str | None = None
    #: AI models only: the deciding row's raw ``is_mentioned`` / ``is_used`` flags.
    is_mentioned: bool | None = None
    is_used: bool | None = None
    #: AI models only, set by the referent check: ``anchor`` | ``confirmed`` |
    #: ``rejected`` | ``unclear``; None when not checked.
    referent: str | None = None
    #: Datasets only, set by the scope check when the resource map split a program from
    #: its data: ``data`` | ``program_only`` | ``unclear``; None when not checked. A label
    #: only: a ``program_only`` paper keeps the judge's verdict (genuine for the release
    #: bar) and is counted in the program-only share.
    scope: str | None = None
    #: Datasets: the full-text validator's own ``mention_scope`` label when served
    #: (``data`` | ``program_only`` | ``not_applicable``); a label only, never decisive.
    validator_scope: str | None = None
    #: Datasets: the scope check turned this paper under its references-only rule.
    scope_references_only: bool = False
    #: Datasets: the pipeline cut the reference list before the judge read the paper
    #: (served ``bibliography_excluded``); the references-only rules then never apply.
    bibliography_excluded: bool = False

    @property
    def genuine(self) -> bool:
        """Correct match for the query: the paper mentions the dataset (used or not)."""
        return self.verdict in ("used", "mentioned")

    @property
    def used(self) -> bool:
        return self.verdict == "used"

    @property
    def four_way(self) -> str:
        """Datasets: ``data`` | ``program_only`` | ``references_only`` | ``unrelated``, the
        split the reports and the split-dataset release bar use. Genuine papers are data or
        program-only (scope label). A ``not_mentioned`` paper (the judge's verdict, or one
        the references-only override or the scope check turned) whose non-flag snippets all
        sit in ``references``, with no flag snippet outside them, is references-only,
        whatever ``bibliography_excluded`` says. Every other paper is unrelated. Known
        limit: a homonym that occurs only in a cited title is references-only too."""
        if self.genuine:
            return "program_only" if self.scope == "program_only" else "data"
        if self.verdict == "not_mentioned" and _references_only(self.contexts):
            return "references_only"
        return "unrelated"

    @property
    def cited_only(self) -> bool | None:
        """AI models: a genuine paper naming the model only in its reference list; None
        when unknown; False for papers that are not genuine."""
        return self.names_in_references_only if self.genuine else False

    @property
    def not_run(self) -> bool:
        """AI models: a genuine paper the judge labelled ``cited`` (discussed, not run)."""
        return self.genuine and self.usage_type == "cited"

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


#: The four-way split of evaluated dataset papers (``PubVerdict.four_way``).
FOUR_WAY = ("data", "program_only", "references_only", "unrelated")


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
    #: ``dataset`` or ``ai_model`` (from the body's ``seed_type``).
    entity: str = "dataset"
    #: Delivered, analyzed papers with no verdict because validation errored.
    n_error_rows: int = 0
    #: AI models: rows judged not this model that still claim a use or a usage type.
    n_incoherent: int = 0
    #: AI models: delivered, analyzed papers whose rows hold no judgement at all.
    n_no_verdict: int = 0
    #: Datasets: the scope check ran on this evidence (set by ``scope_check``).
    scope_ran: bool = False

    @property
    def n_referent_rejected(self) -> int:
        return sum(1 for v in self.verdicts if v.referent == "rejected")

    @property
    def n_referent_unclear(self) -> int:
        return sum(1 for v in self.verdicts if v.referent == "unclear")

    @property
    def scope_checked(self) -> bool:
        return self.scope_ran or any(v.scope is not None for v in self.verdicts)

    @property
    def n_scope_references_only(self) -> int | None:
        """Papers the scope check turned under its references-only rule; None when not run."""
        if not self.scope_checked:
            return None
        return sum(1 for v in self.verdicts if v.scope_references_only)

    @property
    def n_program_only(self) -> int | None:
        """Papers the scope check found to discuss only the program; None when it did not run."""
        if not self.scope_checked:
            return None
        return sum(1 for v in self.verdicts if v.scope == "program_only")

    @property
    def n_scope_unclear(self) -> int | None:
        """Papers the scope check could not decide (kept genuine); None when it did not run."""
        if not self.scope_checked:
            return None
        return sum(1 for v in self.verdicts if v.scope == "unclear")

    @property
    def raw_precision(self) -> float | None:
        """Judge precision before the referent check and the references-only rule (an upper
        bound): every paper either removed counts again. Program-only papers are already
        genuine (a label only)."""
        n = self.n_evaluated
        extra = self.n_referent_rejected + (self.n_scope_references_only or 0)
        return (len(self.genuine) + extra) / n if n else None

    @property
    def n_data(self) -> int | None:
        """Genuine papers not labelled program-only; None when the scope check did not run."""
        if not self.scope_checked:
            return None
        return sum(1 for v in self.verdicts if v.genuine and v.scope != "program_only")

    def four_way_counts(self) -> dict[str, int]:
        """Papers per ``PubVerdict.four_way`` class, in the order of ``FOUR_WAY``."""
        counts = dict.fromkeys(FOUR_WAY, 0)
        for v in self.verdicts:
            counts[v.four_way] += 1
        return counts

    def four_way_summary(self) -> dict[str, dict[str, Any]] | None:
        """Split datasets: count, share and Wilson 95% interval per four-way class; None
        when the scope check did not run or nothing was evaluated."""
        n = self.n_evaluated
        if not self.scope_checked or not n:
            return None
        return {
            k: {"count": c, "share": c / n, "wilson95": _wilson(c, n)}
            for k, c in self.four_way_counts().items()
        }

    @property
    def corrected_precision(self) -> float | None:
        """Precision after the referent check (equal to raw when it did not run)."""
        n = self.n_evaluated
        return len(self.genuine) / n if n else None

    @property
    def n_cited_only(self) -> int | None:
        """AI models: genuine papers that cite the model only; None for datasets."""
        if self.entity != "ai_model":
            return None
        return sum(1 for v in self.verdicts if v.cited_only)

    @property
    def n_cited_only_unknown(self) -> int | None:
        """AI models: genuine papers whose name location is unknown; None for datasets."""
        if self.entity != "ai_model":
            return None
        return sum(1 for v in self.verdicts if v.genuine and v.cited_only is None)

    @property
    def n_not_run(self) -> int | None:
        """AI models: genuine papers labelled ``cited`` (not run); None for datasets."""
        if self.entity != "ai_model":
            return None
        return sum(1 for v in self.verdicts if v.not_run)

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
        cited = self.n_cited_only
        not_run = self.n_not_run
        four_way = self.four_way_summary()
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
            cited_only_count=cited,
            cited_rate=(cited / matched) if cited is not None and matched else None,
            not_run_count=not_run,
            not_run_rate=(not_run / matched) if not_run is not None and matched else None,
            cited_only_unknown_count=self.n_cited_only_unknown,
            raw_fulltext_precision=(
                self.raw_precision if self.entity == "ai_model" or self.scope_checked else None
            ),
            program_only_count=self.n_program_only,
            data_share=(self.n_data / n if self.n_data is not None and n else None),
            fulltext_precision_wilson95=(_wilson(matched, n) if self.scope_checked and n else None),
            data_share_wilson95=(_wilson(self.n_data, n)
                                 if self.n_data is not None and n else None),
            program_only_wilson95=(_wilson(self.n_program_only, n)
                                   if self.n_program_only is not None and n else None),
            used_share_wilson95=(_wilson(used, n) if self.scope_checked and n else None),
            program_only_rate=(
                self.n_program_only / n if self.n_program_only is not None and n else None
            ),
            scope_unclear_count=self.n_scope_unclear,
            scope_references_only_count=self.n_scope_references_only,
            four_way=four_way,
            unrelated_share=four_way["unrelated"]["share"] if four_way else None,
            referent_rejected_count=(
                self.n_referent_rejected if self.entity == "ai_model" else None
            ),
            referent_unclear_count=(
                self.n_referent_unclear if self.entity == "ai_model" else None
            ),
            paper_details=(
                [
                    {
                        "id": v.publication_id,
                        "title": v.title,
                        "verdict": v.verdict,
                        "location": v.mention_location,
                        "cited_only": v.cited_only,
                        "not_run": v.not_run,
                        "matched_name": v.matched_name,
                    }
                    for v in self.verdicts
                ]
                if self.entity == "ai_model"
                else [
                    {
                        "id": v.publication_id,
                        "verdict": v.verdict,
                        "overridden_from": v.overridden_from,
                        "scope": v.scope,
                        "references_only": v.scope_references_only,
                        "validator_scope": v.validator_scope,
                        "terms": sorted({str(c.get("term") or "") for c in v.contexts} - {""}),
                    }
                    for v in self.verdicts
                ]
                if self.scope_checked
                else None
            ),
        )


def _deciding_row(pub: dict[str, Any]) -> dict | None:
    """The validated-dataset row the verdict rests on: the first used row, else the first
    mentioned row, else the first row; None when there are no rows."""
    entries = [e for e in (pub.get("validated_datasets") or []) if isinstance(e, dict)]
    if not entries:
        return None
    used = [e for e in entries if e.get("is_used") is True]
    if used:
        return used[0]
    mentioned = [e for e in entries if e.get("is_mentioned") is True]
    return mentioned[0] if mentioned else entries[0]


def _verdict_for(pub: dict[str, Any]) -> tuple[str, str] | None:
    """Collapse a publication's validated datasets into (verdict, quote); None = no verdict."""
    entries = [e for e in (pub.get("validated_datasets") or []) if isinstance(e, dict)]
    if not entries:
        # Analyzed, but the LLM found no dataset mention at all.
        return ("not_mentioned", "")
    if all(e.get("is_used") is None and e.get("is_mentioned") is None for e in entries):
        return None
    row = _deciding_row(pub)
    verdict = ("used" if row.get("is_used") is True
               else "mentioned" if row.get("is_mentioned") is True else "not_mentioned")
    return (verdict, str(row.get("text_quote") or ""))


_ERROR_MATCH_TYPES = frozenset({"validation_error", "processing_error"})


@dataclass
class _ModelVerdict:
    verdict: str
    quote: str
    usage_type: str | None
    match_type: str | None
    mention_location: str | None = None
    matched_name: str | None = None
    is_mentioned: bool | None = None
    is_used: bool | None = None


_LOCATIONS = ("text", "references_only")


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _model_rows(pub: dict[str, Any]) -> list[dict]:
    return [e for e in (pub.get("validated_ai_models") or []) if isinstance(e, dict)]


def _is_error_row(row: dict) -> bool:
    return row.get("match_type") in _ERROR_MATCH_TYPES


def _model_verdict_for(pub: dict[str, Any]) -> _ModelVerdict | None:
    """Collapse a publication's validated AI models: used > mentioned > not_mentioned.

    None = no verdict: an empty list, only error rows, or rows without any judgement.
    ``usage_type == "cited"`` is a mention (reference-list-only), never a use."""
    rows = [r for r in _model_rows(pub) if not _is_error_row(r)]
    if not rows:
        return None
    if all(r.get("is_used") is None and r.get("is_mentioned") is None and not r.get("usage_type")
           for r in rows):
        return None

    def mk(verdict: str, r: dict) -> _ModelVerdict:
        return _ModelVerdict(verdict, str(r.get("text_quote") or ""), r.get("usage_type"),
                             r.get("match_type"), _text_or_none(r.get("mention_location")),
                             _text_or_none(r.get("matched_name")),
                             r.get("is_mentioned"), r.get("is_used"))

    # A row judged "not this model" blocks its own use and usage type.
    for r in rows:
        if (r.get("is_mentioned") is not False and r.get("is_used") is True
                and r.get("usage_type") != "cited"):
            return mk("used", r)
    for r in rows:
        if r.get("is_mentioned") is True or (
            r.get("is_mentioned") is None
            and (r.get("is_used") is True or r.get("usage_type") == "cited")
        ):
            return mk("mentioned", r)
    return mk("not_mentioned", rows[0])


def _is_incoherent(row: dict) -> bool:
    """Judged not this model, yet claiming a use or a usage type."""
    return row.get("is_mentioned") is False and (
        row.get("is_used") is True or bool(row.get("usage_type"))
    )


def _model_names_in_references_only(contexts: list[dict], location: str | None) -> bool | None:
    """AI models: True when every non-flag name snippet is in ``references`` (at least
    one), False when any is elsewhere. Flag snippets are ignored (another model named in
    the body says nothing about this one). With no name snippet the served location
    decides; otherwise None (unknown)."""
    names = [c for c in contexts if c.get("role") != "flag"]
    if names:
        return all(c.get("part") == "references" for c in names)
    if location in _LOCATIONS:
        return location == "references_only"
    return None


def _references_only(contexts: list[dict]) -> bool:
    """True when the paper names the dataset only in its reference list: at least one
    non-flag snippet, all non-flag snippets in ``references``, and no flag snippet
    outside ``references``. A cited title is not a mention."""
    names = [c for c in contexts if c.get("role") != "flag"]
    if not names or any(c.get("part") != "references" for c in names):
        return False
    return all(c.get("part") == "references" for c in contexts if c.get("role") == "flag")


#: Citation markers: "[62]", "[3, 5-7]", "(ref 62)", "(refs. 3,4)" and superscript digits.
_CITATION_MARKER = re.compile(
    r"\[\s*\d+(?:\s*[-\u2013,]\s*\d+)*\s*\]"
    r"|\(\s*(?:refs?|reference)\.?\s*\d+(?:\s*[-\u2013,]\s*\d+)*\s*\)"
    r"|[\u00b9\u00b2\u00b3\u2070\u2074-\u2079]+",
    re.IGNORECASE,
)
#: A leading entry number ("62. ", "62) ").
_LEAD_NUMBER = re.compile(r"^\s*\d+\s*[.)]\s+")
#: Joiners between excerpts or cited titles: "|", "…", "..." and spaced ". . .".
_JOINER = re.compile(r"\||\u2026|\.(?:\s*\.){2,}")
#: Leading articles of a normalized quote ("the", "a", "an").
_LEAD_ARTICLES = re.compile(r"^(?:(?:the|a|an)\s+)+")
#: Digits glued to the end of a word ("Us41" -> "Us").
_GLUED_DIGITS = re.compile(r"(?<=[^\W\d_])\d+\b")


def _ref_norm(text: str) -> str:
    """Normalize a quote or a references snippet for the references-only containment test:
    citation markers and a leading entry number removed, case-folded, joiners as
    whitespace, digits glued to a word's end removed, whitespace collapsed, and the
    punctuation around the whole text trimmed."""
    t = _LEAD_NUMBER.sub("", _CITATION_MARKER.sub(" ", text or ""))
    t = _JOINER.sub(" ", t.casefold())
    t = _GLUED_DIGITS.sub("", t)
    return " ".join(t.split()).strip(" .,;:")


def _whole_words_in(needle: str, hay: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay) is not None


def quote_allows_references_only(quote: str | None, contexts: list[dict]) -> bool:
    """The references-only override may apply: the validator's quote is empty once
    normalized (``_ref_norm``), or it lies entirely inside one single ``references``
    snippet (whole words, after the same normalization) and has strictly more words than
    every context term it matches (holds it, or lies inside it). Anything else (an author
    or journal line the snippet does not hold, a body sentence, a quote spanning two
    snippets, a quote that is only the name or a piece of it, with or without leading
    articles) keeps the judge's verdict. Known limit: the context terms are the only names
    known here, so a quote that is the dataset's spelled-out name (no context term) counts
    as longer than the name. General rule: no dataset names."""
    q = _ref_norm(quote or "")
    if not q:
        return True
    # Leading articles do not make a quote longer than the name ("the NHANES").
    n_words = len(_LEAD_ARTICLES.sub("", q).split())
    for c in contexts:
        t = _ref_norm(str(c.get("term") or ""))
        if t and (_whole_words_in(t, q) or _whole_words_in(q, t)) and n_words <= len(t.split()):
            return False
    inside = re.compile(rf"(?<!\w){re.escape(q)}(?!\w)")
    return any(
        inside.search(_ref_norm(str(c.get("text") or "")))
        for c in contexts if c.get("part") == "references"
    )


def references_only_override(quote: str | None, contexts: list[dict], *,
                             bibliography_excluded: bool) -> bool:
    """Turn a dataset verdict to ``not_mentioned``: the judge read the reference list (no
    ``bibliography_excluded`` marker), the paper names the dataset only there
    (``_references_only``) and the quote allows it (``quote_allows_references_only``).
    With the bibliography cut by the pipeline the judge is trusted."""
    return (not bibliography_excluded and _references_only(contexts)
            and quote_allows_references_only(quote, contexts))


_ENTITIES = ("dataset", "ai_model")


def _entity_of(results: dict[str, Any], entity: str | None) -> str:
    served = results.get("seed_type")
    if served is not None and served not in _ENTITIES:
        raise ValueError(f"unknown llm-results seed_type {served!r}")
    if entity is not None and entity not in _ENTITIES:
        raise ValueError(f"unknown entity {entity!r}")
    if entity is not None and served is not None and served != entity:
        raise ValueError(f"llm-results seed_type {served!r} does not match entity {entity!r}")
    return entity or served or "dataset"


def build_evidence(results: dict[str, Any], *, entity: str | None = None) -> Evidence:
    """Build :class:`Evidence` from a ``GET /pipelines/{id}/llm-results`` body.

    The sample is every publication that has a full-text outcome; publications never
    sent for full text are not part of it. The entity is the body's ``seed_type``
    (default ``dataset``); an explicit *entity* that contradicts it raises ValueError.
    """
    kind = _entity_of(results, entity)
    pubs = [p for p in (results.get("publications") or []) if isinstance(p, dict)]
    pubs.sort(key=lambda p: -(p.get("citation_count") or 0))
    sample = [p for p in pubs if is_attempted(p)]
    ev = Evidence(n_sampled=len(sample), entity=kind)
    for p in sample:
        if not is_delivered(p):
            continue
        ev.n_delivered += 1
        if not p.get("analyzed"):
            continue
        if p.get("validation_failed"):
            # Validation errored: no verdict (never not_mentioned).
            ev.n_error_rows += 1
            continue
        contexts = [c for c in (p.get("term_contexts") or []) if isinstance(c, dict)]
        if kind == "ai_model":
            mv = _model_verdict_for(p)
            rows = _model_rows(p)
            if mv is None:
                if rows and all(_is_error_row(r) for r in rows):
                    ev.n_error_rows += 1
                else:
                    ev.n_no_verdict += 1
                continue
            if mv.verdict == "not_mentioned" and any(
                _is_incoherent(r) for r in rows if not _is_error_row(r)
            ):
                ev.n_incoherent += 1
            ev.verdicts.append(
                PubVerdict(
                    publication_id=str(p.get("publication_id") or ""),
                    doi=p.get("doi"),
                    title=str(p.get("title") or ""),
                    citation_count=int(p.get("citation_count") or 0),
                    verdict=mv.verdict,
                    quote=mv.quote,
                    contexts=contexts,
                    categories=[c for c in (p.get("category_for") or []) if isinstance(c, str)],
                    usage_type=mv.usage_type,
                    match_type=mv.match_type,
                    names_in_references_only=_model_names_in_references_only(
                        contexts, mv.mention_location
                    ),
                    mention_location=mv.mention_location,
                    matched_name=mv.matched_name,
                    is_mentioned=mv.is_mentioned,
                    is_used=mv.is_used,
                )
            )
            continue
        v = _verdict_for(p)
        if v is None:
            continue
        overridden_from = None
        excluded = _bibliography_excluded(p)
        if v[0] != "not_mentioned" and references_only_override(
                v[1], contexts, bibliography_excluded=excluded):
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
                validator_scope=_served_scope(p),
                bibliography_excluded=excluded,
            )
        )
    return ev


def _bibliography_excluded(pub: dict[str, Any]) -> bool:
    """The pipeline cut the reference list before the judge read the paper: the served
    ``bibliography_excluded`` on the row the verdict rests on, else (a not-mentioned paper
    has no row) the publication-level value. Absent or not True (no marker, or an older
    stack) means the judge saw the reference list."""
    row = _deciding_row(pub)
    if row is not None and "bibliography_excluded" in row:
        return row.get("bibliography_excluded") is True
    return pub.get("bibliography_excluded") is True


def _served_scope(pub: dict[str, Any]) -> str | None:
    """The validator's ``mention_scope`` on the row the verdict rests on (the dataset being
    optimised), never another dataset's row."""
    row = _deciding_row(pub)
    val = row.get("mention_scope") if row else None
    return val if val in ("data", "program_only", "not_applicable") else None


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

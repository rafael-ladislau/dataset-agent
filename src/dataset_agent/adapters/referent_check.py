"""Referent check: is a genuine AI-model full-text paper about *this* model?

The full-text judge answers one broad question (mention or use) and accepts a quoted
reference title or a body passage about another model of the same name. After the
verdict collapse (``fulltext_evidence.build_evidence``), every genuine paper gets a
narrow second look:

* stage 0, no LLM call: a served reference-list name context that contains one of the
  model's admitted anchor or card titles (punctuation folded) confirms the paper;
* stage 1: one narrow question per remaining paper through ``classify_alias_mention``
  with the paper title, a passage (the first body name context, else the reference line,
  else the judge's quote) and the model profile (description, anchor title and year),
  asked ``votes`` times. A majority of False turns the paper to ``not_mentioned``
  (``overridden_from`` keeps the judge's verdict, ``referent`` = ``rejected``); a majority
  of True confirms it; otherwise (failed calls, no majority) it stays genuine as
  ``unclear``. Failed calls are counted, never decided.

No year rule and no domain-word rule. Datasets never enter the check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_agent.adapters.fp_analysis import classify_alias_mention
from dataset_agent.adapters.fulltext_evidence import Evidence, PubVerdict, _fold

_BODY_PARTS = frozenset({"title", "abstract", "body"})


def model_profile(description: str, anchors: list[tuple[str, int | None]]) -> str:
    """One profile string for the model: description plus the first anchor title and year."""
    text = " ".join((description or "").split())
    if anchors:
        title, year = anchors[0]
        text += f" Model paper: {title!r}" + (f" ({year})." if year else ".")
    return text.strip()


def _name_contexts(v: PubVerdict) -> list[dict]:
    return [c for c in v.contexts if c.get("role") != "flag" and str(c.get("text") or "").strip()]


def referent_passage(v: PubVerdict) -> str:
    """The first body name context, else the first reference line, else the judge's quote."""
    names = _name_contexts(v)
    for c in names:
        if c.get("part") in _BODY_PARTS:
            return str(c["text"])
    for c in names:
        if c.get("part") == "references":
            return str(c["text"])
    return v.quote or v.evidence_quote_text()


def matches_anchor(v: PubVerdict, anchor_titles: list[str]) -> bool:
    """A reference-list name context that holds an anchor title (punctuation folded)."""
    keys = [k for k in (_fold(t) for t in anchor_titles) if k]
    if not keys:
        return False
    for c in _name_contexts(v):
        if c.get("part") != "references":
            continue
        text = _fold(str(c.get("text") or ""))
        if any(k in text for k in keys):
            return True
    return False


@dataclass
class ReferentResult:
    n_checked: int = 0
    n_confirmed_by_anchor: int = 0
    n_confirmed: int = 0
    n_rejected: int = 0
    n_unclear: int = 0
    calls: int = 0
    failed_calls: int = 0
    #: publication id -> {"stage", "votes", "decision"}
    papers: dict[str, dict] = field(default_factory=dict)


def check_referents(
    ev: Evidence,
    *,
    agent,
    model_name: str,
    profile: str,
    anchor_titles: list[str],
    votes: int = 3,
    passage_for=referent_passage,
) -> ReferentResult:
    """Run the referent check on *ev*'s genuine papers in place; return the counts."""
    res = ReferentResult()
    if ev.entity != "ai_model":
        return res
    for v in ev.verdicts:
        if not v.genuine:
            continue
        res.n_checked += 1
        if matches_anchor(v, anchor_titles):
            v.referent = "anchor"
            res.n_confirmed_by_anchor += 1
            res.papers[v.publication_id] = {"stage": 0, "votes": [], "decision": "anchor"}
            continue
        passage = passage_for(v)
        cast: list[bool | None] = []
        for _ in range(max(1, votes)):
            res.calls += 1
            answer = classify_alias_mention(
                agent, dataset_name=model_name, alias=v.matched_name or model_name,
                snippet=passage, title=v.title, entity_context=profile,
                passage_label="Passage",
            )
            if answer is None:
                res.failed_calls += 1
            cast.append(answer)
        need = votes // 2 + 1
        if sum(1 for a in cast if a is False) >= need:
            v.overridden_from, v.verdict, v.referent = v.verdict, "not_mentioned", "rejected"
            res.n_rejected += 1
        elif sum(1 for a in cast if a is True) >= need:
            v.referent = "confirmed"
            res.n_confirmed += 1
        else:
            v.referent = "unclear"
            res.n_unclear += 1
        res.papers[v.publication_id] = {"stage": 1, "votes": cast, "decision": v.referent}
    return res

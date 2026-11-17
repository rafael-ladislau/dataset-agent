"""Referent check on genuine AI-model full-text papers (no network: a fake judge)."""

from __future__ import annotations

import pytest

from dataset_agent.adapters.fulltext_evidence import build_evidence
from dataset_agent.adapters.referent_check import (
    check_referents,
    model_profile,
    referent_passage,
)

ANCHOR = "PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method"
PROFILE = model_profile(
    "PhaseNet picks seismic P and S arrival times.", [(ANCHOR, 2018)]
)


class FakeJudge:
    """Answers the referent question from a list of votes (True / False / Exception)."""

    def __init__(self, votes=None, default=True):
        self.votes = list(votes or [])
        self.default = default
        self.prompts: list[str] = []

    def get_structured(self, prompt, _tool):
        self.prompts.append(prompt)
        vote = self.votes.pop(0) if self.votes else self.default
        if isinstance(vote, Exception):
            raise vote
        return {"is_genuine_dataset_reference": vote}


def _c(part, text, term="PhaseNet", role="alias"):
    return {"term": term, "role": role, "part": part, "text": text}


def _pub(pid, *, mentioned=True, used=False, contexts=None, quote="q", title=None):
    return {
        "publication_id": pid, "doi": None, "title": title or f"Title {pid}", "citation_count": 1,
        "fulltext": {"status": "delivered", "source": "pmc", "reason": None},
        "analyzed": True, "validation_failed": False,
        "validated_ai_models": [{"name": "PhaseNet", "is_mentioned": mentioned, "is_used": used,
                                 "text_quote": quote, "match_type": "llm_validation",
                                 "usage_type": None}],
        "category_for": [], "term_contexts": contexts or [],
    }


def _ev(*pubs, seed_type="ai_model"):
    return build_evidence({"task_id": "T", "seed_type": seed_type, "publications": list(pubs)})


def _check(ev, judge, **kw):
    return check_referents(ev, agent=judge, model_name="PhaseNet", profile=PROFILE,
                           anchor_titles=[ANCHOR], **kw)


def test_profile_holds_description_anchor_title_and_year():
    assert PROFILE == (
        f"PhaseNet picks seismic P and S arrival times. Model paper: {ANCHOR!r} (2018)."
    )
    assert model_profile("D.", []) == "D."


def test_reference_line_matching_an_anchor_title_confirms_without_a_call():
    ref = _c("references", f"[12] Zhu, W. & Beroza, G. C. {ANCHOR.lower()}. GJI 216 (2019).")
    ev = _ev(_pub("a", contexts=[ref]))
    judge = FakeJudge()
    res = _check(ev, judge)
    assert judge.prompts == []
    assert res.n_confirmed_by_anchor == 1 and res.calls == 0
    assert ev.verdicts[0].genuine and ev.verdicts[0].referent == "anchor"


def test_other_reference_title_goes_to_the_narrow_question():
    ref = _c("references", "Spoorthi, G. PhaseNet: A deep convolutional network for 2D phase unwrapping.")
    ev = _ev(_pub("a", contexts=[ref], title="Deep learning in optical metrology"))
    judge = FakeJudge([False, False, True])
    res = _check(ev, judge)
    assert res.calls == 3 and len(judge.prompts) == 3
    prompt = judge.prompts[0]
    assert "Passage:" in prompt and "Abstract snippet" not in prompt
    assert "2D phase unwrapping" in prompt and "Deep learning in optical metrology" in prompt
    assert PROFILE in prompt
    v = ev.verdicts[0]
    assert v.verdict == "not_mentioned" and v.overridden_from == "mentioned"
    assert v.referent == "rejected" and not v.cited_only
    assert ev.n_referent_rejected == 1


def test_majority_true_keeps_the_paper():
    ev = _ev(_pub("a", used=True, contexts=[_c("body", "We run PhaseNet on DAS.")]))
    res = _check(ev, FakeJudge([True, False, True]))
    assert ev.verdicts[0].verdict == "used" and ev.verdicts[0].referent == "confirmed"
    assert res.n_rejected == 0


def test_body_name_context_is_preferred_over_the_reference_line_and_the_quote():
    ctx = [_c("references", "Ref line PhaseNet."), _c("body", "Body PhaseNet use."),
           _c("body", "GraphCast body.", term="GraphCast", role="flag")]
    ev = _ev(_pub("a", contexts=ctx, quote="judge quote"))
    assert referent_passage(ev.verdicts[0]) == "Body PhaseNet use."
    ev = _ev(_pub("a", contexts=[_c("references", "Ref line PhaseNet.")], quote="judge quote"))
    assert referent_passage(ev.verdicts[0]) == "Ref line PhaseNet."
    ev = _ev(_pub("a", contexts=[], quote="judge quote"))
    assert referent_passage(ev.verdicts[0]) == "judge quote"


def test_failed_calls_are_counted_not_decided():
    boom = RuntimeError("down")
    ev = _ev(_pub("a"))
    res = _check(ev, FakeJudge([boom, boom, False]))
    assert res.failed_calls == 2 and res.calls == 3
    v = ev.verdicts[0]
    assert v.genuine and v.referent == "unclear" and ev.n_referent_unclear == 1


def test_only_genuine_papers_are_checked():
    ev = _ev(_pub("a", mentioned=False), _pub("b"))
    judge = FakeJudge()
    res = _check(ev, judge)
    assert res.n_checked == 1 and len(judge.prompts) == 3


def test_datasets_never_enter_the_check():
    body = {"task_id": "T", "publications": [{
        "publication_id": "d", "doi": None, "title": "t", "citation_count": 1,
        "fulltext": {"status": "delivered", "source": "pmc", "reason": None},
        "analyzed": True, "validated_datasets": [{"is_mentioned": True, "is_used": True}],
        "term_contexts": [],
    }]}
    ev = build_evidence(body)
    judge = FakeJudge()
    res = _check(ev, judge)
    assert res.n_checked == 0 and judge.prompts == []
    assert ev.verdicts[0].referent is None


def test_raw_and_corrected_precision():
    ev = _ev(_pub("a"), _pub("b"), _pub("c", mentioned=False), _pub("d"))
    _check(ev, FakeJudge([False, False, False]))  # the first checked paper is rejected
    assert ev.raw_precision == pytest.approx(3 / 4)
    assert ev.corrected_precision == pytest.approx(2 / 4)
    m = ev.metrics(0, "q", None)
    assert m.fulltext_precision == pytest.approx(0.5) and m.referent_rejected_count == 1
    assert m.raw_fulltext_precision == pytest.approx(0.75)

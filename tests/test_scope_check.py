"""Scope check on genuine dataset papers (no network: a fake judge)."""

from __future__ import annotations

from dataset_agent.adapters.fulltext_evidence import build_evidence
from dataset_agent.adapters.scope_check import (
    check_scope,
    program_owned_data_names,
    scope_passages,
)

PROGRAM = ["Example Program", "Example Research Program"]
DATA = ["Example Program Workbench", "Controlled Tier", "Curated Data Repository"]


class FakeJudge:
    def __init__(self, answers=None, default="data"):
        self.answers = list(answers or [])
        self.default = default
        self.prompts: list[str] = []

    def get_structured(self, prompt, _tool):
        self.prompts.append(prompt)
        a = self.answers.pop(0) if self.answers else self.default
        if isinstance(a, Exception):
            raise a
        return {"scope": a}


def _c(part, text, term="Example Program"):
    return {"term": term, "role": "alias", "part": part, "text": text}


def _pub(pid, *, verdict="mentioned", contexts=None, quote="judge quote"):
    return {
        "publication_id": pid, "doi": None, "title": f"Title {pid}", "citation_count": 1,
        "fulltext": {"status": "delivered", "source": "pmc", "reason": None},
        "analyzed": True,
        "validated_datasets": [{"is_mentioned": verdict != "not_mentioned",
                                "is_used": verdict == "used", "text_quote": quote}],
        "term_contexts": contexts if contexts is not None else [_c("body", "Example Program aims")],
    }


def _ev(*pubs, seed_type="dataset"):
    return build_evidence({"task_id": "T", "seed_type": seed_type, "publications": list(pubs)})


def _check(ev, judge, **kw):
    return check_scope(ev, agent=judge, dataset_name="Example", description="A cohort.",
                       program_names=PROGRAM, data_names=DATA, **kw)


def test_program_owned_names_carry_the_program_phrase():
    assert program_owned_data_names(PROGRAM, DATA) == ["Example Program Workbench"]


def test_owned_data_name_outside_references_decides_data_without_a_call():
    ev = _ev(_pub("a", contexts=[_c("body", "analyses ran on the Example Program Workbench")]))
    judge = FakeJudge()
    res = _check(ev, judge)
    assert judge.prompts == [] and res.n_data_by_name == 1
    assert ev.verdicts[0].scope == "data" and ev.verdicts[0].genuine


def test_owned_name_only_in_the_reference_list_goes_to_the_question():
    ev = _ev(_pub("a", contexts=[_c("references", "Example Program Workbench. 2023."),
                                 _c("body", "The Example Program will enrol people.")]))
    judge = FakeJudge(["program_only"])
    res = _check(ev, judge)
    assert res.calls == 1 and res.n_program_only == 1
    v = ev.verdicts[0]
    assert v.scope == "program_only" and v.verdict == "mentioned"
    assert v.overridden_from is None and v.genuine  # a label only
    assert "Example Program Workbench. 2023." not in judge.prompts[0]


def test_used_papers_are_data_and_never_asked():
    ev = _ev(_pub("u", verdict="used"), _pub("n", verdict="not_mentioned"))
    judge = FakeJudge()
    res = _check(ev, judge)
    assert judge.prompts == [] and res.n_checked == 0
    assert ev.verdicts[0].scope == "data" and ev.verdicts[1].scope is None


def test_a_paper_with_no_contexts_is_asked_on_the_quote_never_dropped():
    ev = _ev(_pub("a", contexts=[_c("references", "Example Program. 2019.")],
                  quote="Example Program"),
             _pub("b", contexts=[]))
    judge = FakeJudge(default="data")
    res = _check(ev, judge)
    # "a" was already turned by build_evidence's references-only rule; "b" has no context
    # at all: the question is asked on the judge quote, and the paper is not dropped.
    a, b = ev.verdicts
    assert a.verdict == "not_mentioned" and a.scope is None
    assert len(judge.prompts) == 1 and "judge quote" in judge.prompts[0]
    assert "(none)" in judge.prompts[0]  # no passages
    assert b.verdict == "mentioned" and b.scope == "data" and res.n_references_only == 0


def test_no_context_and_no_quote_keeps_the_paper_without_a_call():
    p = _pub("c", contexts=[])
    p["validated_datasets"][0]["text_quote"] = ""
    ev = _ev(p)
    judge = FakeJudge()
    res = _check(ev, judge)
    assert judge.prompts == [] and ev.verdicts[0].genuine and ev.verdicts[0].scope == "unclear"
    assert res.papers["c"]["decision"] == "no_context" and res.n_unclear == 1


def test_contexts_only_in_the_reference_list_fall_under_the_references_only_rule():
    from dataset_agent.adapters.fulltext_evidence import Evidence, PubVerdict

    ev = Evidence(entity="dataset")
    ev.verdicts.append(PubVerdict(publication_id="r", doi=None, title="", citation_count=0,
                                  verdict="mentioned", quote="",
                                  contexts=[_c("references", "Example Program. 2019.")]))
    judge = FakeJudge()
    res = _check(ev, judge)
    assert judge.prompts == [] and res.n_references_only == 1
    assert ev.verdicts[0].verdict == "not_mentioned"


def test_failed_or_unclear_answers_keep_the_paper():
    ev = _ev(_pub("a"), _pub("b"))
    res = _check(ev, FakeJudge([RuntimeError("down"), "unclear"]))
    assert res.failed_calls == 1 and res.n_unclear == 2
    assert all(v.genuine and v.scope == "unclear" for v in ev.verdicts)


def test_passages_put_data_terms_first_and_skip_references():
    ctx = [_c("body", "The Example Program goals."), _c("references", "Controlled Tier ref"),
           _c("abstract", "data from the Controlled Tier"), _c("body", "More program talk."),
           _c("body", "Fourth passage.")]
    ev = _ev(_pub("a", contexts=ctx))
    assert scope_passages(ev.verdicts[0], DATA) == [
        "data from the Controlled Tier", "The Example Program goals.", "More program talk."]


def test_prompt_carries_names_rule_and_quote():
    ev = _ev(_pub("a"))
    judge = FakeJudge(["data"])
    _check(ev, judge)
    p = judge.prompts[0]
    assert "Program names: Example Program, Example Research Program." in p
    assert "Controlled Tier" in p and "'judge quote'" in p and "PROGRAM_ONLY" in p
    # the amended rule and its noun test
    assert "biobanks, cohorts, datasets, repositories or resources" in p
    assert "initiatives, programs, efforts or projects" in p
    assert "future analyses" in p and "own plans to collect data" in p


def test_no_map_or_ai_models_never_enter():
    ev = _ev(_pub("a"))
    judge = FakeJudge()
    assert check_scope(ev, agent=judge, dataset_name="E", description="", program_names=PROGRAM,
                       data_names=[]).n_checked == 0
    ev2 = build_evidence({"task_id": "T", "seed_type": "ai_model", "publications": []})
    assert _check(ev2, judge).n_checked == 0
    assert judge.prompts == [] and ev.verdicts[0].scope is None


def test_list_rules_and_the_unsure_case_do_not_contradict():
    from dataset_agent.adapters.scope_check import scope_prompt

    p = scope_prompt(dataset_name="E", description="", program_names=["P"], data_names=["D"],
                     title="t", passages=["x"], quote="q")
    flat = " ".join(p.split())
    # Owner rule: a list of data nouns is DATA; a list of program nouns with no data noun is
    # PROGRAM_ONLY; only the genuinely unsure case leans toward DATA.
    assert ("named in a list of biobanks, cohorts, datasets, repositories or resources, "
            "or of data sources: DATA") in flat
    assert ("named in a list of initiatives, programs, efforts or projects with no data noun: "
            "PROGRAM_ONLY") in flat
    assert "Only a genuinely unsure case" in flat and "leans toward DATA" in flat
    # The old unqualified close-call sentence is gone.
    assert "a list mention" not in flat
    assert "an unsure case is DATA" not in flat


def test_availability_acknowledgement_funding_or_annex_snippets_are_not_references():
    # Any part other than the reference list is text: the paper is not references-only.
    for part in ("availability", "acknowledgement", "funding", "annex"):
        ctx = [_c(part, "Data are available through the Example Program Workbench portal."),
               _c("references", "The Example Program. 2019.")]
        ev = _ev(_pub("a", contexts=ctx))
        res = _check(ev, FakeJudge(["data"]))
        v = ev.verdicts[0]
        assert res.n_references_only == 0, part
        assert v.verdict == "mentioned" and not v.scope_references_only, part


def _ref_only_verdict(quote):
    from dataset_agent.adapters.fulltext_evidence import Evidence, PubVerdict

    ev = Evidence(entity="dataset")
    ev.verdicts.append(PubVerdict(publication_id="r", doi=None, title="", citation_count=0,
                                  verdict="mentioned", quote=quote,
                                  contexts=[_c("references", "The Example Program. 2019.")]))
    return ev


def test_references_only_rule_keeps_the_verdict_when_the_quote_is_from_the_text():
    ev = _ref_only_verdict("Participants of the EP41 cohort were enrolled.")
    judge = FakeJudge(["data"])
    res = _check(ev, judge)
    assert res.n_references_only == 0 and len(judge.prompts) == 1
    assert ev.verdicts[0].verdict == "mentioned" and ev.verdicts[0].scope == "data"


def test_references_only_rule_applies_on_an_empty_or_reference_quote():
    for quote in ("", "The Example Program"):
        ev = _ref_only_verdict(quote)
        res = _check(ev, FakeJudge())
        assert res.n_references_only == 1 and ev.verdicts[0].verdict == "not_mentioned"

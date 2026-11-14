"""Refiner: program-only papers get their own block and never create an exclude."""

from __future__ import annotations

from dataset_agent.adapters.fulltext_evidence import build_evidence
from dataset_agent.adapters.fulltext_refiner import propose_refinements, refine_candidate
from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.settings import Settings


def _pub(pid, *, verdict, text):
    return {
        "publication_id": pid, "doi": None, "title": f"T {pid}", "citation_count": 1,
        "fulltext": {"status": "delivered", "source": "pmc", "reason": None}, "analyzed": True,
        "validated_datasets": [{"is_mentioned": verdict != "not_mentioned",
                                "is_used": verdict == "used", "text_quote": text}],
        "term_contexts": [{"term": "Example Program", "role": "alias", "part": "body", "text": text}],
    }


def _evidence():
    pubs = [_pub(f"g{i}", verdict="used", text="We analysed Example Program data.") for i in range(6)]
    pubs += [_pub(f"p{i}", verdict="mentioned", text="Example Program raises consent ethics issues.")
             for i in range(4)]
    ev = build_evidence({"task_id": "T", "publications": pubs})
    for v in ev.verdicts:
        if v.publication_id.startswith("p"):
            v.overridden_from, v.verdict, v.scope = v.verdict, "not_mentioned", "program_only"
        else:
            v.scope = "data"
    return ev


class Agent:
    def __init__(self, answer=None):
        self.prompts = []
        self.answer = answer or {}

    def get_structured(self, prompt, _tool):
        self.prompts.append(prompt)
        return self.answer


def _kw(ev):
    return dict(dataset_name="Example", genuine=ev.genuine,
                false_positives=[v for v in ev.false_positives if v.scope != "program_only"],
                aliases=["Example Program"], flag_terms=["Workbench"], exclude_terms=[])


def test_prompt_is_byte_identical_without_program_only_papers():
    ev = _evidence()
    a, b = Agent(), Agent()
    propose_refinements(a, **_kw(ev))
    propose_refinements(b, **_kw(ev), program_only=[])
    assert a.prompts == b.prompts and "program only" not in a.prompts[0].lower()


def test_program_only_papers_get_their_own_block():
    ev = _evidence()
    a = Agent()
    propose_refinements(a, **_kw(ev), program_only=[v for v in ev.verdicts if v.scope == "program_only"])
    p = a.prompts[0]
    assert "only the program" in p and "never a source of exclude terms" in p
    assert p.count("consent ethics") == 8  # 4 papers: quote and snippet each


def test_program_only_papers_never_create_an_exclude():
    ev = _evidence()
    agent = Agent({"exclude_terms": ["consent ethics"], "flag_terms": [], "new_aliases": [],
                   "false_positive_groups": []})
    s = Settings(fulltext_min_evidence=5, refine_min_exclude_papers=2)
    cand = QueryCandidate(main_dataset_name="Example", safe=[], risky=["Example Program"],
                          flag_terms=["Workbench"])
    res = refine_candidate(cand, ev, settings=s, agent=agent, dataset_name="Example",
                           impact_fn=lambda t: 50)
    assert res.candidate.exclude_terms == []
    assert any("rejected exclude candidates" in n and "consent ethics" in n for n in res.notes)
    assert "program only" in agent.prompts[0].lower()

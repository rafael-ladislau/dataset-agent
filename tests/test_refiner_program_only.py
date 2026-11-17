"""Refiner: papers labelled program-only stay genuine (round 3), so they never create an
exclude and never count against a name."""

from __future__ import annotations

from dataset_agent.adapters.fulltext_evidence import build_evidence
from dataset_agent.adapters.fulltext_refiner import refine_candidate
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
        v.scope = "program_only" if v.publication_id.startswith("p") else "data"
    return ev


class Agent:
    def __init__(self, answer=None):
        self.prompts = []
        self.answer = answer or {}

    def get_structured(self, prompt, _tool):
        self.prompts.append(prompt)
        return self.answer


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
    assert "program only" not in agent.prompts[0].lower()  # no separate block any more
    assert agent.prompts[0].count("consent ethics") == 8  # listed with the matching papers

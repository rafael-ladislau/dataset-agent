"""Round 2, item 6: a refiner alias that names the program follows the phase-1 routing."""

from __future__ import annotations

from dataset_agent.adapters.fulltext_refiner import refine_candidate
from dataset_agent.adapters.scope_check import acronym_form, names_program
from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import ResourceMap
from tests.test_fulltext_refiner import S, FakeLLM, ev, pub

PROGRAMS = ["All of Example", "Example Research Program"]


def test_names_program_equals_contains_or_abbreviates():
    assert names_program("All of Example", PROGRAMS) == "All of Example"
    assert names_program("the All of Example cohort", PROGRAMS) == "All of Example"
    assert names_program("AoE cohort", PROGRAMS) == "All of Example"
    assert names_program("ERP data", PROGRAMS) == "Example Research Program"
    assert names_program("Example Workbench", PROGRAMS) is None
    assert acronym_form("AoE cohort") and not acronym_form("All of Example cohort")


def _refine(new_aliases, quote):
    pubs = [pub(f"g{i}", 40 - i, True, quote) for i in range(3)]
    llm = FakeLLM({"exclude_terms": [], "flag_terms": [], "new_aliases": new_aliases,
                   "reasoning": ""})
    cand = QueryCandidate("X", safe=["Example data"], risky=["All of Example"],
                          flag_terms=["Example Workbench"])
    return refine_candidate(cand, ev(pubs), settings=S, agent=llm, dataset_name="X",
                            count_fn=lambda t: 50, reference_count=100, program_names=PROGRAMS)


def test_refiner_alias_naming_the_program_is_gated():
    r = _refine(["AoE cohort", "Example Research Program"],
                "we used the AoE cohort and the Example Research Program data")
    assert "AoE cohort" in r.candidate.risky and "AoE cohort" not in r.candidate.safe
    assert "Example Research Program" in r.candidate.risky
    assert any("names the program: gated" in d.reason for d in r.diffs if d.term == "AoE cohort")


def test_data_noun_without_acronym_keeps_count_routing():
    r = _refine(["All of Example cohort"], "participants of the All of Example cohort were used")
    assert "All of Example cohort" in r.candidate.safe


def test_acronym_data_phrases_are_gated_in_phase1():
    rmap = ResourceMap(program_names=["All of Example", "AOE"], data_resource_names=["Example Workbench"])
    safe, risky, _ = QueryOptimizationUseCase._apply_resource_map(
        rmap, ["data from the AOE", "AOE dataset"], [], [], [])
    assert "data from the AOE" in risky and "AOE dataset" in risky
    assert "data from the AOE" not in safe and "AOE dataset" not in safe and "AOE" in risky

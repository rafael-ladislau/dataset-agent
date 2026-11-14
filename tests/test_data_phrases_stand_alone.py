"""Round 2, item 2: generated data phrases are safe candidates, never behind the gate."""

from __future__ import annotations

from dataset_agent.adapters.fulltext_recall import recall_candidates
from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import ResourceMap

MAP = ResourceMap(program_names=["Example Program", "EP"], data_resource_names=["Example Workbench"])


def test_no_recruitment_phrase_is_generated():
    from dataset_agent.adapters.scope_check import generated_data_phrases

    assert not any("participants" in p for p in generated_data_phrases(["Example Program"]))


def test_data_phrases_leave_the_gate_but_acronym_forms_do_not():
    notes: list[str] = []
    safe, risky, gate = QueryOptimizationUseCase._apply_resource_map(
        MAP, ["Example Cohort Study"],
        ["Example Program data", "data from the Example Program", "EP data", "data from the EP"],
        ["Agency"], notes)
    assert "Example Program data" in safe and "data from the Example Program" in safe
    assert "EP data" in risky and "data from the EP" in risky
    assert "Example Program" in risky and "EP" in risky and gate == ["Example Workbench"]
    assert any("data phrases stand alone" in n for n in notes)


def test_recall_check_may_promote_a_data_phrase():
    cur = QueryCandidate(main_dataset_name="Example Program", safe=["Example Cohort Study"],
                         risky=["Example Program"], flag_terms=["Example Workbench"])
    kw = dict(main_names=["Example Program"], known_aliases=["data from the Example Program"],
              flag_terms=["Example Workbench"])
    eligible, rejected = recall_candidates(cur, **kw)
    assert "data from the Example Program" not in [c.term for c in eligible]
    assert any("data from the Example Program (not a proper name)" in r for r in rejected)
    eligible, _ = recall_candidates(cur, **kw, data_phrases=["data from the Example Program"])
    assert "data from the Example Program" in [c.term for c in eligible]


def test_specific_data_names_stand_alone_and_the_gate_is_screened():
    rmap = MAP.model_copy(update={
        "data_resource_names": ["Example Workbench", "Data Browser", "DB"],
        "specific_data_names": ["Example Workbench"],
        "data_name_hits": {"Example Workbench": 100, "Data Browser": 900_000, "DB": 10}})
    notes: list[str] = []
    safe, risky, gate = QueryOptimizationUseCase._apply_resource_map(
        rmap, [], ["Example Program"], [], notes, max_hits=500_000)
    assert "Example Workbench" in safe and gate == ["Example Workbench"]
    assert any("Data Browser (900000 Dimensions hits)" in n and "DB (bare short acronym)" in n
               for n in notes)

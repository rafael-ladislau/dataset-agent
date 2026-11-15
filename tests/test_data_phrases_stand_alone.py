"""Generated data phrases are safe candidates (round 2, item 2); acronym forms stay risky."""

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
    assert "Example Program" in safe and "EP" in risky and gate == ["Agency"]


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


def test_specific_data_names_stand_alone_and_generic_ones_are_not_searched():
    rmap = MAP.model_copy(update={
        "data_resource_names": ["Example Workbench", "Data Browser", "DB"],
        "specific_data_names": ["Example Workbench"],
        "data_name_hits": {"Example Workbench": 100, "Data Browser": 900_000, "DB": 10}})
    notes: list[str] = []
    safe, risky, gate = QueryOptimizationUseCase._apply_resource_map(
        rmap, [], ["Example Program"], [], notes, max_hits=500_000)
    assert "Example Workbench" in safe and gate == []
    assert "Data Browser" not in safe + risky and "DB" not in safe + risky
    assert any("data names not searched ['Data Browser', 'DB']" in n for n in notes)

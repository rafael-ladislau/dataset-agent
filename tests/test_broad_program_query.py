"""Round 3, item 1: for a split dataset the query is broad on the program name.

Program names and specific data resource names are OR-ed; nothing is gated by data names.
A bare ordinary phrase or a short acronym never stands alone; a name phase 1 marked risky
stays risky (the probe decides); the gate is the research flags, as without a map.
"""

from __future__ import annotations

from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import ResourceMap

MAP = ResourceMap(
    program_names=["All of Example", "All of Example Research Program", "AOE"],
    data_resource_names=["Example Workbench", "Data Browser", "Controlled Tier"],
    specific_data_names=["Example Workbench"],
    data_name_hits={"Example Workbench": 900, "Data Browser": 12_000, "Controlled Tier": 1_200},
)


def _apply(rmap=MAP, safe=(), risky=(), flags=("Example Agency",)):
    notes: list[str] = []
    out = QueryOptimizationUseCase._apply_resource_map(
        rmap, list(safe), list(risky), list(flags), notes)
    return (*out, notes)


def test_program_and_specific_data_names_are_or_ed():
    safe, risky, flags, notes = _apply()
    assert "All of Example Research Program" in safe
    assert "Example Workbench" in safe
    assert flags == ["Example Agency"]  # research flags; no data name gates anything
    assert "Data Browser" not in safe + risky + flags  # generic data name: not searched
    assert "Controlled Tier" not in flags
    assert any("broad on the program name" in n for n in notes)


def test_ordinary_phrase_and_acronym_never_stand_alone():
    rmap = MAP.model_copy(update={"program_names": ["All of Us", "AOU", "All of Us Research Program"]})
    safe, risky, flags, notes = _apply(rmap)
    assert "All of Us" in risky and "All of Us" not in safe
    assert "AOU" in risky and "AOU" not in safe
    assert "All of Us Research Program" in safe
    assert any("never stand alone" in n and "All of Us" in n for n in notes)


def test_phase1_risky_names_stay_risky():
    safe, risky, _, _ = _apply(risky=["All of Example Research Program"])
    assert "All of Example Research Program" in risky
    assert "All of Example Research Program" not in safe


def test_generated_phrases_follow_the_acronym_guard():
    rmap = MAP.model_copy(update={"program_names": ["All of Us", "AOU"]})
    safe, risky, _, _ = _apply(rmap, risky=["All of Us data", "data from the All of Us",
                                            "AOU data", "data from the AOU"])
    assert "All of Us data" in safe and "data from the All of Us" in safe
    assert "AOU data" in risky and "data from the AOU" in risky


def test_flags_that_became_safe_names_leave_the_gate():
    _, _, flags, _ = _apply(flags=["All of Example Research Program", "Example Agency"])
    assert flags == ["Example Agency"]


def test_screened_specific_names_do_not_stand_alone():
    rmap = MAP.model_copy(update={"specific_data_names": ["Example Workbench", "EW"],
                                  "data_name_hits": {"Example Workbench": 900_000}})
    notes: list[str] = []
    safe, _, _ = QueryOptimizationUseCase._apply_resource_map(
        rmap, [], [], [], notes, max_hits=500_000)
    assert "Example Workbench" not in safe and "EW" not in safe

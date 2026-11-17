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


FORMER = ["Old Example Name", "Parent Example Initiative", "Huge Parent Initiative"]
FMAP = MAP.model_copy(update={
    "program_names": ["All of Example Research Program", *FORMER],
    "former_or_parent_names": list(FORMER),
})


def test_former_or_parent_names_never_stand_alone_unscreened():
    safe, risky, _, notes = _apply(FMAP)
    assert "All of Example Research Program" in safe
    for name in FORMER:
        assert name not in safe and name not in risky


def test_former_or_parent_names_go_through_the_count_and_risk_path(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from dataset_agent.settings import Settings

    hits = {"Old Example Name": 50, "Parent Example Initiative": 5_000,
            "Huge Parent Initiative": 900_000}

    class DSL:
        async def execute_dsl(self, dsl):
            n = next((c for k, c in hits.items() if k in dsl), 0)
            return SimpleNamespace(count_total=n, publications=[])

    s = Settings(tasks_db=tmp_path / "t.db", output_dir=tmp_path / "o", dimensions_api_key="k",
                 optimize_max_dimensions_calls=100, optimize_flag_max_hits=500_000)
    uc = QueryOptimizationUseCase(dsl_port=DSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=s)
    entries: list = []
    notes: list[str] = []
    safe, risky = asyncio.run(uc._screen_former_program_names(
        FMAP, [], [], entries, c_full=100, has_fn=True, notes_parts=notes))
    assert safe == ["Old Example Name"]  # within 10x the title count
    assert risky == ["Parent Example Initiative"]  # > 10x: gated by the research flags
    assert "Huge Parent Initiative" not in safe + risky  # above max_hits: not searched
    assert {e.alias: e.count for e in entries} == hits
    assert any("Huge Parent Initiative" in n and "not searched" in n for n in notes)
    # Then the broad step keeps them where the screen put them.
    out_safe, out_risky, _, _ = _apply(FMAP, safe=safe, risky=risky)
    assert "Old Example Name" in out_safe and "Parent Example Initiative" in out_risky
    assert "Huge Parent Initiative" not in out_safe + out_risky


def test_former_names_with_stray_whitespace_or_case_never_stand_alone():
    # The map keeps the LLM's former names unstripped; program names are stripped.
    rmap = FMAP.model_copy(update={"former_or_parent_names": [
        " Old Example Name ", "parent  example\tinitiative", "HUGE PARENT INITIATIVE"]})
    assert QueryOptimizationUseCase._own_program_names(rmap) == ["All of Example Research Program"]
    safe, risky, _, _ = _apply(rmap)
    for name in FORMER:
        assert name not in safe and name not in risky

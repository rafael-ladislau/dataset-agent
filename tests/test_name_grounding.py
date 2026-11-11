"""Name grounding for descriptive dataset titles (phase 1 recovery)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dataset_agent.adapters.name_grounding import (
    GROUNDING_TOOL,
    needs_grounding,
    screen_candidates,
)
from dataset_agent.application import query_optimization as qopt_mod
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import (
    DatasetRecord,
    Group,
    OptimizeRequest,
    TermsEvaluation,
    YearsRange,
)
from dataset_agent.settings import Settings

TITLE = "GNSS EarthScope UNAVCO Earthquakes"
FLAGS = ["EarthScope Consortium", "UNAVCO"]


def test_needs_grounding_only_for_dead_title_and_weak_aliases() -> None:
    assert needs_grounding(0, [0, 23, 1], 100)
    assert not needs_grounding(0, [80, 40], 100)
    assert not needs_grounding(24, [0, 0], 100)


def test_screen_rejects_rearranged_title_generic_acronym_and_duplicates() -> None:
    kept, rejected = screen_candidates(
        [
            "Plate Boundary Observatory",
            "PBO",
            "UNAVCO GNSS",
            "GNSS Earthquake Data",
            "gnss time series",
            "UNAVCO",
            "Network of the Americas",
            "plate boundary observatory",
            "GNSS Earthquakes",
        ],
        dataset_name=TITLE,
        known_aliases=["GNSS Earthquakes"],
        flag_terms=FLAGS,
        limit=8,
    )
    assert kept == ["Plate Boundary Observatory", "Network of the Americas"]
    assert "PBO (bare short acronym)" in rejected
    assert "UNAVCO GNSS (rearranges the title)" in rejected
    assert "GNSS Earthquake Data (rearranges the title)" in rejected
    assert "gnss time series (not a proper name)" in rejected
    assert not any(r.startswith("UNAVCO (") or r.startswith("GNSS Earthquakes (") for r in rejected)


def test_screen_respects_limit() -> None:
    kept, rejected = screen_candidates(
        ["Alpha Network", "Beta Archive Name", "Gamma Observatory"],
        dataset_name=TITLE, known_aliases=[], flag_terms=[], limit=2,
    )
    assert kept == ["Alpha Network", "Beta Archive Name"]
    assert rejected == ["Gamma Observatory (over the candidate limit)"]


class _KeyedDSL:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts
        self.calls: list[str] = []

    async def execute_dsl(self, dsl: str) -> SimpleNamespace:
        self.calls.append(dsl)
        for phrase, n in self._counts.items():
            if f'\\"{phrase}\\"' in dsl:
                return SimpleNamespace(count_total=n, publications=[])
        return SimpleNamespace(count_total=0, publications=[])


def _record(aliases: list[str], suggested: list[str]) -> DatasetRecord:
    return DatasetRecord(
        engine="dimensions",
        group=Group(name="default"),
        main_dataset_name=TITLE,
        description="GNSS stations of the Network of the Americas run by EarthScope Consortium.",
        dataset_names=aliases,
        flag_terms=FLAGS,
        years_range=YearsRange(start_year=2015, end_year=2025),
        terms_evaluation=TermsEvaluation(
            is_effective=False, dataset_names_score=3, flag_terms_score=6,
            suggested_dataset_names=suggested,
        ),
    )


def _run(tmp_path, monkeypatch, counts, aliases, suggested, llm_names, **overrides):
    monkeypatch.setattr(qopt_mod, "web_search", lambda q: "web results")
    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync",
        lambda *a, **k: {"fp_rate_pct": 0.0, "fp_hits": [], "publications_total": 10,
                         "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                         "total_pubs": 10, "coverage_pct": 100.0, "query": "",
                         "for_clause": k.get("for_clause", "")},
    )
    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=60,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0,
        optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
        optimize_suffix_check_max_aliases=0,
        **overrides,
    )
    research = MagicMock()
    research.execute.return_value = (_record(aliases, suggested), tmp_path / "r.json")
    agent = MagicMock()
    agent.get_structured.return_value = (
        {"names": llm_names} if isinstance(llm_names, list) else llm_names
    )
    uc = QueryOptimizationUseCase(
        dsl_port=_KeyedDSL(counts), research=research, agent=agent, settings=settings
    )
    out = asyncio.run(uc.execute(OptimizeRequest(dataset_name=TITLE)))
    return out, agent


def _grounding_calls(agent) -> list:
    return [c for c in agent.get_structured.call_args_list if c.args[1] is GROUNDING_TOOL]


def test_descriptive_title_is_grounded_in_cited_names(tmp_path, monkeypatch) -> None:
    counts = {"GNSS Earthquakes": 23, "Plate Boundary Observatory": 1045,
              "Network of the Americas": 308, "GAGE Facility": 2}
    out, agent = _run(
        tmp_path, monkeypatch, counts,
        aliases=[TITLE, "GNSS Earthquakes", "GNSS Earthquake Data"],
        suggested=["Plate Boundary Observatory", "PBO"],
        llm_names=["Network of the Americas", "GAGE Facility", "UNAVCO GNSS"],
    )
    assert len(_grounding_calls(agent)) == 1
    assert out.failed_phase != "phase1_alias_counting"
    pool = out.aliases.safe + out.aliases.risky
    assert "Plate Boundary Observatory" in pool
    assert "Network of the Americas" in pool
    assert "GAGE Facility" not in pool and "PBO" not in pool and "UNAVCO GNSS" not in pool
    assert "grounding: title has no Dimensions hits; added 'Plate Boundary Observatory'" in out.notes
    assert "GAGE Facility (2 Dimensions hits)" in out.notes


def test_grounding_rescues_run_where_every_alias_has_zero_hits(tmp_path, monkeypatch) -> None:
    out, _agent = _run(
        tmp_path, monkeypatch, {"Plate Boundary Observatory": 1045},
        aliases=[TITLE, "EarthScope GNSS Earthquake Catalog"], suggested=[],
        llm_names=["Plate Boundary Observatory"],
    )
    assert out.failed_phase != "phase1_alias_counting"
    assert "Plate Boundary Observatory" in out.aliases.safe + out.aliases.risky


def test_no_grounding_when_title_has_hits(tmp_path, monkeypatch) -> None:
    counts = {TITLE: 40, "GNSS Earthquakes": 23}
    _out, agent = _run(
        tmp_path, monkeypatch, counts, aliases=[TITLE, "GNSS Earthquakes"],
        suggested=["Plate Boundary Observatory"], llm_names=["Plate Boundary Observatory"],
    )
    assert _grounding_calls(agent) == []


def test_grounding_can_be_disabled(tmp_path, monkeypatch) -> None:
    out, agent = _run(
        tmp_path, monkeypatch, {"Plate Boundary Observatory": 1045},
        aliases=[TITLE], suggested=["Plate Boundary Observatory"], llm_names=[],
        optimize_grounding_enabled=False,
    )
    assert _grounding_calls(agent) == []
    assert out.failed_phase == "phase1_alias_counting"


@pytest.mark.parametrize("llm_reply", [{}, {"names": "oops"}, None])
def test_grounding_survives_malformed_llm_reply(tmp_path, monkeypatch, llm_reply) -> None:
    out, _agent = _run(
        tmp_path, monkeypatch, {"Plate Boundary Observatory": 1045},
        aliases=[TITLE], suggested=["Plate Boundary Observatory"], llm_names=llm_reply,
    )
    assert "Plate Boundary Observatory" in out.aliases.safe + out.aliases.risky

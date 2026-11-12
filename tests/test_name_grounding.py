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


def _run(tmp_path, monkeypatch, counts, aliases, suggested, llm_names, record=None,
         web="web results", llm=None, **overrides):
    monkeypatch.setattr(qopt_mod, "web_search", lambda q: web)
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
    research.execute.return_value = (record or _record(aliases, suggested), tmp_path / "r.json")
    agent = MagicMock()
    agent.get_structured.return_value = (
        {"names": llm_names} if isinstance(llm_names, list) else llm_names
    )
    if llm is not None:
        agent.get_structured.side_effect = llm
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


def test_generic_alias_is_dropped_and_does_not_block_grounding(tmp_path, monkeypatch) -> None:
    # "dataset" alone has millions of hits; it must neither reach the query nor hide
    # that the real aliases find nothing.
    counts = {"dataset": 2_000_000, "GNSS Earthquakes": 23, "Plate Boundary Observatory": 1045}
    out, agent = _run(
        tmp_path, monkeypatch, counts,
        aliases=[TITLE, "GNSS Earthquakes", "dataset"], suggested=["Plate Boundary Observatory"],
        llm_names=[],
    )
    pool = out.aliases.safe + out.aliases.risky
    assert "dataset" not in pool
    assert all(e.alias != "dataset" for e in out.alias_counts)
    assert "dropped generic alias(es) ['dataset']" in out.notes
    assert len(_grounding_calls(agent)) == 1
    assert "Plate Boundary Observatory" in pool


@pytest.mark.parametrize(
    ("alias", "value", "expected"),
    [
        (TITLE, "dataset", f"{TITLE} dataset"),
        ("WR", "WR dataset", "WR dataset"),
        ("WR", "wr  dataset", "wr dataset"),
        (TITLE, "GNSS dataset", ""),
        ("WR", "WR", ""),
        ("WR", "WRDS", ""),
    ],
)
def test_llm_suffix_form_must_extend_the_alias(alias, value, expected) -> None:
    from dataset_agent.application.research import _suffixed_form_of

    assert _suffixed_form_of(alias, value) == expected


def test_defined_names_need_matching_initials() -> None:
    from dataset_agent.adapters.name_grounding import defined_names

    text = (
        "The Network of the Americas (NOTA) spans 20 countries. Plate Boundary Observatory "
        "(PBO) stations. Data Portal (beta). Global Navigation Satellite System (GNSS)."
    )
    assert defined_names(text) == [
        ("Network of the Americas", "NOTA"),
        ("Plate Boundary Observatory", "PBO"),
        ("Global Navigation Satellite System", "GNSS"),
    ]


def test_web_defined_name_grounds_when_llm_returns_nothing(tmp_path, monkeypatch) -> None:
    # The LLM emits no names, but the web results define the network with its acronym.
    web = (
        "- Network of the Americas | EarthScope Consortium\n  The Network of the Americas "
        "(NOTA) is a sensor network of GNSS stations. Global Navigation Satellite System "
        "(GNSS) receivers. Tiny Name Archive (TNA)."
    )
    counts = {"GNSS Earthquakes": 23, "Network of the Americas": 1011,
              "Global Navigation Satellite System": 90000, "Tiny Name Archive": 40}
    out, _agent = _run(
        tmp_path, monkeypatch, counts, aliases=[TITLE, "GNSS Earthquakes"], suggested=[],
        llm_names=[], web=web,
    )
    pool = out.aliases.safe + out.aliases.risky
    assert "Network of the Americas" in pool
    assert "Global Navigation Satellite System" not in pool
    assert "Tiny Name Archive" not in pool  # unvouched: needs optimize_grounding_min_hits
    assert "Global Navigation Satellite System (spells out a title acronym)" in out.notes
    assert "Tiny Name Archive (40 Dimensions hits)" in out.notes


def test_unvouched_name_too_common_is_rejected(tmp_path, monkeypatch) -> None:
    out, _agent = _run(
        tmp_path, monkeypatch, {"Synthetic Aperture Radar": 50000},
        aliases=[TITLE, "GNSS Earthquakes"], suggested=[], llm_names=[],
        web="Synthetic Aperture Radar (SAR) data.",
    )
    assert "too common for an unvouched name" in out.notes
    assert "Synthetic Aperture Radar" not in out.aliases.safe + out.aliases.risky


def test_broad_funder_flags_dropped_next_to_specific_ones(tmp_path, monkeypatch) -> None:
    rec = _record([TITLE, "Plate Boundary Observatory"], [])
    rec.flag_terms = ["EarthScope Consortium", "NSF", "National Science Foundation"]
    out, _agent = _run(
        tmp_path, monkeypatch, {"Plate Boundary Observatory": 1045}, aliases=[], suggested=[],
        llm_names=[], record=rec,
    )
    assert "dropped broad funder flag(s) ['NSF', 'National Science Foundation']" in out.notes


def test_lone_funder_flag_is_kept(tmp_path, monkeypatch) -> None:
    rec = _record([TITLE, "Plate Boundary Observatory"], [])
    rec.flag_terms = ["NSF"]
    out, _agent = _run(
        tmp_path, monkeypatch, {"Plate Boundary Observatory": 1045}, aliases=[], suggested=[],
        llm_names=[], record=rec,
    )
    assert "dropped broad funder flag" not in out.notes


def test_is_organization_alias() -> None:
    from dataset_agent.adapters.name_grounding import is_organization_alias

    flags = ["EarthScope Consortium", "UNAVCO"]
    assert is_organization_alias("Earthscope", flags, TITLE)
    assert is_organization_alias("Earthscope dataset", flags, TITLE)
    assert not is_organization_alias("GNSS Earthquakes", flags, TITLE)
    assert not is_organization_alias("UK Biobank", ["UK Biobank Limited"], "UK Biobank")


def test_organization_alias_does_not_block_grounding_and_is_dropped(tmp_path, monkeypatch) -> None:
    counts = {"Earthscope": 5000, "Earthscope dataset": 120, "Network of the Americas": 1011}
    out, agent = _run(
        tmp_path, monkeypatch, counts, aliases=[TITLE, "Earthscope", "Earthscope dataset"],
        suggested=[], llm_names=["Network of the Americas"],
    )
    assert len(_grounding_calls(agent)) == 1
    pool = out.aliases.safe + out.aliases.risky
    assert "Network of the Americas" in pool
    assert "Earthscope" not in pool and "Earthscope dataset" not in pool
    assert "dropped organization alias(es) ['Earthscope', 'Earthscope dataset']" in out.notes


def test_organization_alias_kept_when_nothing_better_is_found(tmp_path, monkeypatch) -> None:
    out, _agent = _run(
        tmp_path, monkeypatch, {"Earthscope": 5000}, aliases=[TITLE, "Earthscope"],
        suggested=[], llm_names=[], web="",
    )
    assert "Earthscope" in out.aliases.safe + out.aliases.risky

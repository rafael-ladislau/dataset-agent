"""Risk of grounded and history names, research flag screening, and the refiner's
demote-on-overlapping-exclude backstop (GNSS: "BINEX", "GAGE", "COCONet")."""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dataset_agent.adapters.fulltext_evidence import build_evidence
from dataset_agent.adapters.fulltext_refiner import refine_candidate
from dataset_agent.adapters.name_grounding import GROUNDING_TOOL
from dataset_agent.adapters.query_heuristics import is_acronym_shaped_name
from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.application import query_optimization as qopt_mod
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import (
    DatasetRecord,
    Group,
    OptimizeRequest,
    TermsEvaluation,
    YearsRange,
)
from dataset_agent.domain.ports import AgentPort
from dataset_agent.settings import Settings

TITLE = "GNSS EarthScope UNAVCO Earthquakes"
_PHRASE = re.compile(r'\\"([^"\\]+)\\"')


class _ShareDSL:
    """Counts per phrase; ``"<name>" AND (<flags>)`` reads ``with_flags[name]``."""

    def __init__(self, counts, with_flags=None):
        self.counts = counts
        self.with_flags = with_flags or {}
        self.calls: list[str] = []

    async def execute_dsl(self, dsl: str):
        self.calls.append(dsl)
        phrases = _PHRASE.findall(dsl)
        first = phrases[0] if phrases else ""
        if " AND (" in dsl and first in self.with_flags:
            return SimpleNamespace(count_total=self.with_flags[first], publications=[])
        return SimpleNamespace(count_total=self.counts.get(first, 0), publications=[])


def _record(flags, suggested):
    return DatasetRecord(
        engine="dimensions",
        group=Group(name="default"),
        main_dataset_name=TITLE,
        description="GNSS stations run by EarthScope Consortium.",
        dataset_names=[TITLE, "GNSS Earthquakes"],
        flag_terms=flags,
        years_range=YearsRange(start_year=2015, end_year=2025),
        terms_evaluation=TermsEvaluation(
            is_effective=False, dataset_names_score=3, flag_terms_score=6,
            suggested_dataset_names=suggested,
        ),
    )


def _run(tmp_path, monkeypatch, dsl, flags, suggested, **overrides):
    monkeypatch.setattr(qopt_mod, "web_search", lambda q: "web results")
    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync",
        lambda *a, **k: {"fp_rate_pct": 0.0, "fp_hits": [], "publications_total": 10,
                         "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                         "total_pubs": 10, "coverage_pct": 100.0, "query": "",
                         "for_clause": k.get("for_clause", "")},
    )
    base = dict(
        tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out", dimensions_api_key="k",
        optimize_max_dimensions_calls=80, optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0, optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=0, optimize_web_disambig_max_aliases=0,
        optimize_name_history_enabled=False,
    )
    settings = Settings(**{**base, **overrides})
    research = MagicMock()
    research.execute.return_value = (_record(flags, suggested), tmp_path / "r.json")
    agent = MagicMock()
    agent.get_structured.side_effect = lambda prompt, tool: (
        {"names": []} if tool is GROUNDING_TOOL else {}
    )
    uc = QueryOptimizationUseCase(dsl_port=dsl, research=research, agent=agent, settings=settings)
    return asyncio.run(uc.execute(OptimizeRequest(dataset_name=TITLE)))


@pytest.mark.parametrize(
    ("name", "expected"),
    [("BINEX", True), ("COCONet", True), ("TLALOCNet", True), ("Prithvi", False),
     ("Network of the Americas", False), ("EarthScope PBO", False), ("ABCDEFGHIJK", False)],
)
def test_acronym_shaped_name(name, expected):
    assert is_acronym_shaped_name(name) is expected


def test_gage_flag_dropped_and_names_classified_by_flag_share(tmp_path, monkeypatch):
    dsl = _ShareDSL(
        counts={"GNSS Earthquakes": 3, "BINEX": 308, "Network of the Americas": 1011,
                "EarthScope PBO": 64, "GAGE": 5_291_583, "EarthScope Consortium": 2000,
                "UNAVCO": 9000},
        with_flags={"BINEX": 35, "Network of the Americas": 74, "EarthScope PBO": 33},
    )
    out = _run(
        tmp_path, monkeypatch, dsl, flags=["EarthScope Consortium", "GAGE", "UNAVCO"],
        suggested=["BINEX", "Network of the Americas", "EarthScope PBO"],
    )
    assert "GAGE" not in out.flag_terms
    assert "dropped research flag(s): GAGE (bare short acronym)" in out.notes
    assert "'BINEX' flag-gated: acronym-shaped name" in out.notes
    assert "BINEX" in out.aliases.risky
    # 7% of its hits carry a flag: kept, but only with flags (never rejected).
    assert "Network of the Americas" in out.aliases.risky
    assert "'Network of the Americas': 74/1011 hits (7%) contain a flag term; risky" in out.notes
    # 52% carry a flag: searched alone.
    assert "EarthScope PBO" in out.aliases.safe
    # The share query ORs only the screened flags.
    share_q = next(c for c in dsl.calls if '\\"BINEX\\" AND (' in c or '\\"Network of the Americas\\" AND (' in c)
    assert "GAGE" not in share_q


def test_too_broad_flag_dropped_by_count(tmp_path, monkeypatch):
    dsl = _ShareDSL(
        counts={"GNSS Earthquakes": 3, "Plate Boundary Observatory": 2115,
                "EarthScope Consortium": 2000, "Geophysics Laboratory": 900_000},
        with_flags={"Plate Boundary Observatory": 823},
    )
    out = _run(
        tmp_path, monkeypatch, dsl, flags=["EarthScope Consortium", "Geophysics Laboratory"],
        suggested=["Plate Boundary Observatory"],
    )
    assert out.flag_terms == ["EarthScope Consortium"]
    assert "Geophysics Laboratory (900000 Dimensions hits)" in out.notes


def test_lone_short_flag_is_kept(tmp_path, monkeypatch):
    dsl = _ShareDSL(counts={"GNSS Earthquakes": 3, "Plate Boundary Observatory": 2115, "GAGE": 5_000_000})
    out = _run(tmp_path, monkeypatch, dsl, flags=["GAGE"], suggested=["Plate Boundary Observatory"])
    assert out.flag_terms == ["GAGE"]
    assert "dropped research flag(s)" not in out.notes


def test_flag_screening_can_be_disabled(tmp_path, monkeypatch):
    dsl = _ShareDSL(counts={"GNSS Earthquakes": 3, "Plate Boundary Observatory": 2115,
                            "UNAVCO": 9000, "GAGE": 5_000_000})
    out = _run(
        tmp_path, monkeypatch, dsl, flags=["UNAVCO", "GAGE"], suggested=["Plate Boundary Observatory"],
        optimize_drop_short_acronym_flags=False, optimize_flag_max_hits=0,
    )
    assert "GAGE" in out.flag_terms


# --- refiner backstop -----------------------------------------------------------------


def _pub(pid, used, quote):
    return {"publication_id": pid, "doi": None, "title": "", "citation_count": 10,
            "fulltext": {"status": "delivered"}, "analyzed": True,
            "validated_datasets": [{"name": "X", "is_used": used, "is_mentioned": used,
                                    "text_quote": quote}]}


class _LLM(AgentPort):
    def __init__(self, out):
        self.out = out

    def get_information(self, prompt, *, tools=None):
        return ""

    def get_structured(self, prompt, result_tool):
        return self.out


_S = Settings(refine_min_alias_papers=3, fulltext_min_evidence=3, refine_min_exclude_papers=2)


def _refine(excludes, n_fp=2, **settings):
    pubs = (
        [_pub(f"g{i}", True, "GNSS data from the Network of the Americas") for i in range(4)]
        + [_pub(f"f{i}", False, "BINEX tablets from Binex Co were dispensed") for i in range(n_fp)]
    )
    cand = QueryCandidate("X", safe=["Network of the Americas", "BINEX"],
                          flag_terms=["EarthScope Consortium"])
    out = {"exclude_terms": excludes, "flag_terms": [], "new_aliases": [], "reasoning": ""}
    s = _S.model_copy(update=settings) if settings else _S
    return refine_candidate(cand, build_evidence({"publications": pubs}), settings=s,
                            agent=_LLM(out), dataset_name="X", count_fn=lambda t: 5)


def test_overlapping_exclude_demotes_safe_alias_with_little_evidence():
    r = _refine(["Binex Co"])
    d = next(d for d in r.diffs if d.kind == "alias" and d.term == "BINEX")
    assert d.action == "demote"
    assert "'Binex Co'" in d.reason
    assert "BINEX" in r.candidate.risky and "BINEX" not in r.candidate.safe
    assert "Network of the Americas" in r.candidate.safe


def test_overlapping_exclude_in_one_false_positive_does_not_demote():
    r = _refine(["Binex Co"], n_fp=1)
    assert "BINEX" in r.candidate.safe
    assert any("Binex Co (overlaps an alias or flag)" in n for n in r.notes)


def test_overlapping_exclude_in_genuine_paper_does_not_demote():
    # "Network of the Americas" occurs in the genuine quotes: no other meaning shown.
    r = _refine(["Network of the Americas"])
    assert "Network of the Americas" in r.candidate.safe


def test_flag_inside_name_is_no_evidence(tmp_path, monkeypatch):
    # "UNAVCO GAGE" always contains UNAVCO: only the other flags measure its share.
    dsl = _ShareDSL(
        counts={"GNSS Earthquakes": 3, "UNAVCO GAGE": 24, "EarthScope GAGE": 40,
                "EarthScope Consortium": 2000, "UNAVCO": 9000},
        with_flags={"UNAVCO GAGE": 5, "EarthScope GAGE": 25},
    )
    out = _run(
        tmp_path, monkeypatch, dsl, flags=["EarthScope Consortium", "UNAVCO"],
        suggested=["UNAVCO GAGE", "EarthScope GAGE"],
    )
    assert "'UNAVCO GAGE': 5/24 hits (21%) contain a flag term; risky" in out.notes
    assert "UNAVCO GAGE" in out.aliases.risky
    share_q = next(c for c in dsl.calls if '\\"UNAVCO GAGE\\" AND (' in c)
    assert '\\"UNAVCO\\"' not in share_q.split(" AND (", 1)[1]
    assert '\\"EarthScope Consortium\\"' in share_q


def test_all_flags_overlapping_name_keeps_default_risk(tmp_path, monkeypatch):
    dsl = _ShareDSL(
        counts={"GNSS Earthquakes": 3, "UNAVCO GAGE": 24, "UNAVCO": 9000, "UNAVCO Inc": 50},
        with_flags={"UNAVCO GAGE": 24},
    )
    out = _run(tmp_path, monkeypatch, dsl, flags=["UNAVCO", "UNAVCO Inc"], suggested=["UNAVCO GAGE"])
    assert "'UNAVCO GAGE': every flag shares a word with it; flag share not measured" in out.notes
    assert "UNAVCO GAGE" in out.aliases.risky  # title has no hits: default is risky
    assert not any('\\"UNAVCO GAGE\\" AND (' in c for c in dsl.calls)


def test_budget_defaults_leave_room_for_refinement():
    s = Settings(dimensions_api_key="k")
    assert s.optimize_max_dimensions_calls == 96


def test_phase1_checks_keep_the_refiner_reserve(tmp_path, monkeypatch):
    names = [f"Station Network {c}" for c in "ABCDEFGH"]
    counts = {"GNSS Earthquakes": 3, "EarthScope Consortium": 2000, "UNAVCO": 9000}
    counts.update({n: 500 for n in names})
    dsl = _ShareDSL(counts=counts, with_flags={n: 400 for n in names})
    cap, reserve = 30, 14
    out = _run(
        tmp_path, monkeypatch, dsl, flags=["EarthScope Consortium", "UNAVCO"], suggested=names,
        optimize_max_dimensions_calls=cap, refine_impact_budget_reserve=reserve,
        optimize_grounding_max_candidates=8,
    )
    assert "flag share not measured (Dimensions budget kept for refinement)" in out.notes
    share_calls = [c for c in dsl.calls if '\\" AND (' in c and "Station Network" in c]
    assert 0 < len(share_calls) < len(names)
    # Every share query ran while the refiner's reserve (plus the next gate) was intact.
    last_share = max(i for i, c in enumerate(dsl.calls) if '\\" AND (' in c and "Station Network" in c)
    assert last_share + 1 <= cap - reserve - 2


def _screen(flags, short_guard=False):
    settings = Settings(
        tasks_db="t.db", output_dir="out", dimensions_api_key="k",
        optimize_flag_max_hits=0, optimize_drop_short_acronym_flags=short_guard,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=MagicMock(), research=MagicMock(), agent=MagicMock(), settings=settings
    )
    notes: list[str] = []
    return asyncio.run(uc._screen_research_flags(flags, notes)), notes


def test_research_flag_that_is_a_broader_form_of_another_is_dropped():
    kept, notes = _screen(["EarthScope", "EarthScope Consortium", "UNAVCO", "IRIS Consortium"])
    assert kept == ["EarthScope Consortium", "UNAVCO", "IRIS Consortium"]
    assert notes == ["dropped research flag(s): EarthScope (broader form of EarthScope Consortium)"]


def test_flags_differing_only_in_case_keep_the_first_and_are_not_dropped_as_broader():
    kept, notes = _screen(["EarthScope", "earthscope"])
    assert kept == ["EarthScope"]
    assert notes == ["dropped research flag(s): earthscope (same words as EarthScope)"]


def test_flags_with_equal_tokens_keep_one():
    kept, notes = _screen(["O*NET", "O-NET"])
    assert kept == ["O*NET"]
    assert notes == ["dropped research flag(s): O-NET (same words as O*NET)"]


def test_equal_token_flags_then_short_acronym_guard():
    kept, notes = _screen(["UNAVCO Inc.", "UNAVCO, Inc", "IRIS"], short_guard=True)
    assert kept == ["UNAVCO Inc."]
    assert notes == [
        "dropped research flag(s): UNAVCO, Inc (same words as UNAVCO Inc.); IRIS (bare short acronym)"
    ]


"""A dataset's program/data decision is stored on its research record and reused on later
runs; it is drawn again only on an explicit refresh or when nothing decided is stored."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from dataset_agent.adapters.resource_map import ResourceMapUndecided
from dataset_agent.adapters.storage_json import JsonDatasetRepository
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.application.research import DatasetResearchUseCase
from dataset_agent.domain.models import (
    DatasetRecord,
    Group,
    OptimizeRequest,
    ResearchRequest,
    ResourceMap,
    YearsRange,
)
from dataset_agent.settings import Settings
from tests.test_fulltext_loop import CountDSL, FakeClient, make_settings

NAME = "Example Program"


def _rec(**kw):
    return DatasetRecord(engine="dimensions", group=Group(name="default"),
                         main_dataset_name=NAME, description="A cohort.",
                         dataset_names=[NAME, "Example Cohort Study"], flag_terms=[],
                         years_range=YearsRange(start_year=2015, end_year=2026), **kw)


def _map(tag: str = "Researcher Workbench") -> ResourceMap:
    return ResourceMap(program_names=[NAME], data_resource_names=[tag],
                       name_sources={tag: "pages"},
                       found_names=[{"name": tag, "source": "pages", "where": "https://x/data"}])


def _research(tmp_path, draw, **settings):
    s = Settings(output_dir=str(tmp_path), resource_map_enabled=True, **settings)
    uc = DatasetResearchUseCase(MagicMock(), MagicMock(), JsonDatasetRepository(tmp_path), s,
                                dsl_port=MagicMock())
    uc._resource_map = draw
    return uc


def _research_run(uc, refresh: bool = False) -> DatasetRecord:
    """What ``execute`` does around the map: begin the decision, decide, save."""
    uc.begin_map_decision(ResearchRequest(dataset_name=NAME, refresh_resource_map=refresh))
    rec = _rec()
    uc._map_once(rec)
    uc._repository.save(rec)
    return rec


# -- the store -------------------------------------------------------------------------

def test_repository_loads_what_it_saved(tmp_path):
    repo = JsonDatasetRepository(tmp_path)
    assert repo.load(NAME) is None
    repo.save(_rec(resource_map=_map(), resource_map_checked=True))
    got = repo.load(NAME)
    assert got is not None and got.resource_map.data_resource_names == ["Researcher Workbench"]
    (tmp_path / "example_program_research.json").write_text("{not json", encoding="utf-8")
    assert repo.load(NAME) is None


# -- research --------------------------------------------------------------------------

def test_first_run_draws_and_stores_then_later_runs_reuse(tmp_path):
    draws = []
    first = _research_run(_research(tmp_path, lambda r: draws.append(1) or _map()))
    assert draws == [1]
    d1 = first.resource_map_decision
    assert d1.origin == "drawn" and d1.split and d1.decided_at and not d1.refreshed
    # A later run (new use case, same store) reuses it: no draw, same date, same names.
    second = _research_run(_research(tmp_path, lambda r: draws.append(2) or _map("Other")))
    assert draws == [1]
    d2 = second.resource_map_decision
    assert d2.origin == "reused" and d2.decided_at == d1.decided_at
    assert d2.stored_in == "example_program_research.json"
    assert second.resource_map.data_resource_names == ["Researcher Workbench"]
    assert second.resource_map.found_names[0]["where"] == "https://x/data"
    assert second.program_datasets is not None


def test_refresh_draws_again_and_replaces_the_stored_map(tmp_path):
    _research_run(_research(tmp_path, lambda r: _map()))
    fresh = _research_run(_research(tmp_path, lambda r: _map("Data Browser")), refresh=True)
    assert fresh.resource_map_decision.origin == "drawn"
    assert fresh.resource_map_decision.refreshed
    later = _research_run(_research(tmp_path, lambda r: _map("never drawn")))
    assert later.resource_map.data_resource_names == ["Data Browser"]


def test_a_firm_no_split_is_a_decision_and_is_reused(tmp_path):
    draws = []
    _research_run(_research(tmp_path, lambda r: draws.append(1) or None))
    again = _research_run(_research(tmp_path, lambda r: draws.append(2) or _map()))
    assert draws == [1]
    assert again.resource_map is None and again.resource_map_checked
    assert again.resource_map_decision.origin == "reused"
    assert not again.resource_map_decision.split


def test_an_undecided_draw_is_never_stored_and_is_retried(tmp_path):
    def undecided(_rec):
        raise ResourceMapUndecided("no pages could be read")

    failed = _research_run(_research(tmp_path, undecided))
    assert failed.resource_map_error and not failed.resource_map_checked
    assert failed.resource_map_decision is None
    draws = []
    retried = _research_run(_research(tmp_path, lambda r: draws.append(1) or _map()))
    assert draws == [1] and retried.resource_map_decision.origin == "drawn"


def test_a_stored_record_without_a_decision_is_not_reused(tmp_path):
    JsonDatasetRepository(tmp_path).save(_rec())  # research ran with the map off
    draws = []
    _research_run(_research(tmp_path, lambda r: draws.append(1) or _map()))
    assert draws == [1]


# -- optimize --------------------------------------------------------------------------

def _optimize(tmp_path, research):
    s = make_settings(tmp_path)
    s.resource_map_enabled = True
    return QueryOptimizationUseCase(dsl_port=CountDSL(), research=research, agent=MagicMock(),
                                    settings=s, extraction_client=FakeClient([{}]))


def test_optimize_reuses_the_stored_map_for_an_undecided_research_record(tmp_path):
    store = tmp_path / "store"
    research = _research(store, lambda r: pytest.fail("research must not draw"))
    research._repository.save(_rec(resource_map=_map(), resource_map_checked=True,
                                   resource_map_decision={"origin": "drawn", "split": True,
                                                          "decided_at": "2026-11-15T10:00:00Z"}))
    uc = _optimize(tmp_path, research)

    async def draw(_rec):
        raise AssertionError("optimize must not draw a stored map")

    uc._research_resource_map = draw

    async def replayed(_req):  # a stored research output from before the map was stored
        return _rec()

    uc._run_research = replayed
    out = asyncio.run(uc.execute(OptimizeRequest(dataset_name=NAME)))
    assert out.resource_map_decision.origin == "reused"
    assert out.resource_map_decision.decided_at == "2026-11-15T10:00:00Z"
    assert out.resource_map.data_resource_names == ["Researcher Workbench"]
    assert "reused the stored decision" in out.notes


def test_optimize_draws_once_stores_and_the_next_run_reuses(tmp_path):
    store = tmp_path / "store"
    research = _research(store, lambda r: pytest.fail("research must not draw"))
    uc = _optimize(tmp_path, research)
    draws = []

    async def draw(_rec):
        draws.append(1)
        return _map()

    uc._research_resource_map = draw

    async def replayed(_req):
        return _rec()

    uc._run_research = replayed
    first = asyncio.run(uc.execute(OptimizeRequest(dataset_name=NAME)))
    second = asyncio.run(uc.execute(OptimizeRequest(dataset_name=NAME)))
    assert draws == [1]
    assert first.resource_map_decision.origin == "drawn"
    assert second.resource_map_decision.origin == "reused"
    assert second.resource_map_decision.decided_at == first.resource_map_decision.decided_at
    third = asyncio.run(uc.execute(OptimizeRequest(dataset_name=NAME, refresh_resource_map=True)))
    assert draws == [1, 1] and third.resource_map_decision.refreshed


def test_a_map_passed_in_the_request_wins_over_the_stored_one(tmp_path):
    store = tmp_path / "store"
    research = _research(store, lambda r: pytest.fail("no draw"))
    research._repository.save(_rec(resource_map=_map(), resource_map_checked=True))
    uc = _optimize(tmp_path, research)
    req = OptimizeRequest(dataset_name=NAME, dataset_names=[NAME, "Example Cohort Study"],
                          flag_terms=["NIH"], resource_map=_map("Given Tier"),
                          resource_map_checked=True)
    out = asyncio.run(uc.execute(req))
    assert out.resource_map_decision.origin == "request"
    assert out.resource_map.data_resource_names == ["Given Tier"]


def test_optimize_never_changes_the_callers_map(tmp_path):
    uc = _optimize(tmp_path, MagicMock())
    given = _map()
    req = OptimizeRequest(dataset_name=NAME, dataset_names=[NAME, "Example Cohort Study"],
                          flag_terms=["NIH"], resource_map=given, resource_map_checked=True)
    asyncio.run(uc.execute(req))
    assert uc._resource_map is not req.resource_map


# -- switches --------------------------------------------------------------------------

def test_refresh_is_one_switch_on_the_requests_and_the_cli():
    assert ResearchRequest.model_fields["refresh_resource_map"].default is False
    assert OptimizeRequest.model_fields["refresh_resource_map"].default is False
    from dataset_agent.interfaces.cli import app

    for cmd in ("run", "optimize"):
        r = CliRunner().invoke(app, [cmd, "--help"])
        assert "--refresh-map" in r.stdout


def test_cli_run_passes_the_refresh_switch(monkeypatch):
    import dataset_agent.bootstrap as boot
    from dataset_agent.interfaces.cli import app

    seen = {}

    async def fake(req, settings):
        seen["refresh"] = req.refresh_resource_map
        raise RuntimeError("stop here")

    monkeypatch.setattr(boot, "run_full_pipeline", fake)
    CliRunner().invoke(app, ["run", NAME, "--refresh-map"])
    assert seen == {"refresh": True}


def test_full_pipeline_keeps_a_map_optimize_drew_after_research_failed(tmp_path, monkeypatch):
    """Research could not decide; optimize drew and stored the map. The pipeline's final
    save of the research record must not overwrite that decision."""
    import dataset_agent.bootstrap as boot

    def undecided(_rec):
        raise ResourceMapUndecided("the map LLM call failed")

    research = _research(tmp_path, undecided)

    def execute(req):
        rec = _research_run(research)
        return rec, research._repository.path_for(NAME)

    research.execute = execute

    class Opt:
        async def execute(self, req):
            assert req.resource_map_error
            decision = research.drawn_decision(NAME, _map(), refreshed=False)
            research.remember_map(_rec(), _map(), decision)
            from dataset_agent.domain.models import QueryOptimizationRecord

            return QueryOptimizationRecord(resource_map_decision=decision)

    monkeypatch.setattr(boot, "build_request_agent", lambda s, r: (MagicMock(), None))
    monkeypatch.setattr(boot, "build_use_case", lambda *a, **k: research)
    monkeypatch.setattr(boot, "build_optimize_use_case", lambda *a, **k: Opt())
    s = Settings(output_dir=str(tmp_path), resource_map_enabled=True, dimensions_api_key="k")
    record, _path = asyncio.run(boot.run_full_pipeline(ResearchRequest(dataset_name=NAME), s))
    stored = JsonDatasetRepository(tmp_path).load(NAME)
    assert stored.resource_map_checked and stored.resource_map_decision.origin == "drawn"
    assert record.resource_map_decision.origin == "drawn"


def test_a_map_passed_without_dataset_names_wins_over_research(tmp_path):
    """Research runs (no dataset_names) and decides its own map; the request's map wins."""
    store = tmp_path / "store"
    research = _research(store, lambda r: pytest.fail("no draw"))
    uc = _optimize(tmp_path, research)

    async def researched(_req):
        return _rec(resource_map=_map(), resource_map_checked=True,
                    resource_map_decision={"origin": "reused", "split": True})

    uc._run_research = researched
    out = asyncio.run(uc.execute(OptimizeRequest(dataset_name=NAME,
                                                 resource_map=_map("Given Tier"))))
    assert out.resource_map_decision.origin == "request"
    assert out.resource_map.data_resource_names == ["Given Tier"]
    assert "passed in the request" in out.notes


def test_a_request_no_split_is_recorded_as_a_request_decision(tmp_path):
    store = tmp_path / "store"
    research = _research(store, lambda r: pytest.fail("no draw"))
    research._repository.save(_rec(resource_map=_map(), resource_map_checked=True))
    uc = _optimize(tmp_path, research)
    req = OptimizeRequest(dataset_name=NAME, dataset_names=[NAME, "Example Cohort Study"],
                          flag_terms=["NIH"], resource_map=None, resource_map_checked=True)
    out = asyncio.run(uc.execute(req))
    assert out.resource_map is None
    assert out.resource_map_decision.origin == "request"
    assert not out.resource_map_decision.split
    assert "passed in the request (no split)" in out.notes

    async def researched(_req):
        return _rec(resource_map=_map(), resource_map_checked=True)

    uc._run_research = researched
    out = asyncio.run(uc.execute(OptimizeRequest(dataset_name=NAME, resource_map_checked=True)))
    assert out.resource_map is None and out.resource_map_decision.origin == "request"


@pytest.mark.parametrize(("enabled", "dsl"), [(False, True), (True, False)])
def test_a_run_that_does_not_draw_keeps_the_stored_decision(tmp_path, enabled, dsl):
    """Map step off (or no DSL port): the run saves its record, but the stored map fields
    are carried forward, so the next run with the step on reuses them without a draw."""
    _research_run(_research(tmp_path, lambda r: _map()))
    s = Settings(output_dir=str(tmp_path), resource_map_enabled=enabled, dimensions_api_key="")
    off = DatasetResearchUseCase(MagicMock(), MagicMock(), JsonDatasetRepository(tmp_path), s,
                                 dsl_port=MagicMock() if dsl else None)
    off._run_pipeline_once = lambda req: _rec()
    record, _path = off.execute(ResearchRequest(dataset_name=NAME))
    assert record.resource_map is None  # this run used no map
    stored = JsonDatasetRepository(tmp_path).load(NAME)
    assert stored.resource_map_checked and stored.resource_map is not None
    draws = []
    later = _research_run(_research(tmp_path, lambda r: draws.append(1) or _map("Other")))
    assert draws == []
    assert later.resource_map_decision.origin == "reused"
    assert later.resource_map.data_resource_names == ["Researcher Workbench"]

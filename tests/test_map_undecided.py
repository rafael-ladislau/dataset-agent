"""Round 4, item 5: a failed map LLM call, or no pages, is recorded as "could not be
decided", never as a firm "no split"; the record is not marked checked, so the optimize
path draws the map once more (no network: fakes)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dataset_agent.adapters.resource_map import ResourceMapUndecided, map_for_record
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.application.research import DatasetResearchUseCase
from dataset_agent.domain.models import OptimizeRequest, ResourceMap
from dataset_agent.settings import Settings
from tests.test_fulltext_loop import CountDSL, FakeClient, make_settings
from tests.test_one_map_decision import _rec


class Down:
    def __init__(self):
        self.calls = 0

    def get_structured(self, _p, _t):
        self.calls += 1
        raise RuntimeError("LLM timeout")


def test_a_failed_map_call_is_undecided_not_no_split():
    with pytest.raises(ResourceMapUndecided, match="LLM"):
        map_for_record(Down(), _rec(), count=lambda c: 10, pages="Some page text.")


def test_no_pages_is_undecided_without_an_llm_call():
    agent = Down()
    with pytest.raises(ResourceMapUndecided, match="pages"):
        map_for_record(agent, _rec(), count=lambda c: 10, pages="  ")
    assert agent.calls == 0


def test_research_records_the_failure_and_leaves_the_map_unchecked(monkeypatch, tmp_path):
    uc = DatasetResearchUseCase(MagicMock(), MagicMock(), MagicMock(),
                                Settings(output_dir=str(tmp_path), resource_map_enabled=True),
                                dsl_port=MagicMock())

    def fail(_rec):
        raise ResourceMapUndecided("the map LLM call failed or returned no answer")

    monkeypatch.setattr(uc, "_resource_map", fail)
    rec = _rec()
    uc._map_once(rec)
    assert rec.resource_map is None and rec.resource_map_checked is False
    assert "LLM" in (rec.resource_map_error or "")


def _optimize(tmp_path, stored, draw):
    s = make_settings(tmp_path)
    s.resource_map_enabled = True
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=s, extraction_client=FakeClient([{}]))
    uc._research_resource_map = draw

    async def research(_req):
        return stored

    uc._run_research = research
    return asyncio.run(uc.execute(OptimizeRequest(dataset_name="Example Program")))


def test_optimize_retries_once_and_reports_a_second_failure(tmp_path):
    draws = []

    async def draw(_rec):
        draws.append(1)
        raise ResourceMapUndecided("the map LLM call failed or returned no answer")

    stored = _rec(resource_map_error="the map LLM call failed or returned no answer")
    out = _optimize(tmp_path, stored, draw)
    assert draws == [1]
    assert out.resource_map is None
    assert out.resource_map_error and "LLM" in out.resource_map_error
    assert "resource map: could not be decided" in out.notes
    assert stored.resource_map_checked is False


def test_optimize_retry_that_succeeds_clears_the_failure(tmp_path):
    rmap = ResourceMap(program_names=["Example Program"], data_resource_names=["Example Workbench"])

    async def draw(_rec):
        return rmap

    stored = _rec(resource_map_error="no pages could be read")
    out = _optimize(tmp_path, stored, draw)
    assert out.resource_map_error is None
    assert stored.resource_map_checked is True and stored.resource_map_error is None


def test_full_pipeline_request_retries_a_failed_research_map(tmp_path):
    s = make_settings(tmp_path)
    s.resource_map_enabled = True
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=s, extraction_client=FakeClient([{}]))
    draws = []

    async def draw(rec):
        draws.append(rec.main_dataset_name)
        raise ResourceMapUndecided("no pages could be read")

    uc._research_resource_map = draw
    out = asyncio.run(uc.execute(OptimizeRequest(
        dataset_name="Example Program", dataset_names=["Example Program", "Example Cohort Study"],
        flag_terms=["NIH"], resource_map_error="the map LLM call failed or returned no answer")))
    assert draws == ["Example Program"]
    assert out.resource_map_error == "no pages could be read"
    # Without a recorded failure, a request with names never draws the map.
    draws.clear()
    asyncio.run(uc.execute(OptimizeRequest(
        dataset_name="Example Program", dataset_names=["Example Program"], flag_terms=["NIH"])))
    assert draws == []

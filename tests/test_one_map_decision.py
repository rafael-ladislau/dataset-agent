"""Round 2, item 9: one resource-map decision per run."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.application.research import DatasetResearchUseCase
from dataset_agent.domain.models import DatasetRecord, Group, OptimizeRequest, YearsRange
from dataset_agent.settings import Settings
from tests.test_fulltext_loop import CountDSL, FakeClient, make_settings


def _rec(**kw):
    return DatasetRecord(engine="dimensions", group=Group(name="default"),
                         main_dataset_name="Example Program", description="A cohort.",
                         dataset_names=["Example Program", "Example Cohort Study"], flag_terms=[],
                         years_range=YearsRange(start_year=2015, end_year=2026), **kw)


def test_research_draws_the_map_once_across_literature_attempts(monkeypatch, tmp_path):
    uc = DatasetResearchUseCase(MagicMock(), MagicMock(), MagicMock(),
                                Settings(output_dir=str(tmp_path), resource_map_enabled=True),
                                dsl_port=MagicMock())
    draws = []
    monkeypatch.setattr(uc, "_resource_map", lambda rec: draws.append(1) or None)
    recs = [_rec() for _ in range(3)]  # one per literature attempt
    for rec in recs:
        uc._map_once(rec)
    assert draws == [1] and all(r.resource_map_checked for r in recs)


def test_optimize_does_not_draw_again_after_research_checked(monkeypatch, tmp_path):
    s = make_settings(tmp_path)
    s.resource_map_enabled = True
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=s, extraction_client=FakeClient([{}]))
    draws = []

    async def draw(_rec):
        draws.append(1)
        return None

    uc._research_resource_map = draw

    async def stored(_req):
        return _rec(resource_map_checked=True)

    uc._run_research = stored
    asyncio.run(uc.execute(OptimizeRequest(dataset_name="Example Program")))
    assert draws == []

    async def fresh(_req):
        return _rec()

    uc._run_research = fresh
    asyncio.run(uc.execute(OptimizeRequest(dataset_name="Example Program")))
    assert draws == [1]

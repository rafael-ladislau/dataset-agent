"""Round 2, item 8: the full pipeline draws the map with working counts and passes it on."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from dataset_agent import bootstrap
from dataset_agent.application.research import DatasetResearchUseCase
from dataset_agent.domain.models import DatasetRecord, Group, ResourceMap, YearsRange
from dataset_agent.settings import Settings

PAGES = "The Example Program enrolls people. Researchers use the Example Workbench."


class Dsl:
    def __init__(self):
        self.calls = []

    async def execute_dsl(self, dsl):
        self.calls.append(dsl)
        return type("R", (), {"count_total": 40, "errors_string": None})()


class Agent:
    def get_structured(self, _prompt, _tool):
        return {"names": [{"name": "Example Program", "kind": "program"},
                          {"name": "Example Workbench", "kind": "data_resource"}]}


def _rec():
    return DatasetRecord(engine="dimensions", group=Group(name="default"),
                         main_dataset_name="Example Program", description="A cohort.",
                         dataset_names=["Example Program"], flag_terms=[],
                         years_range=YearsRange(start_year=2015, end_year=2026))


def test_map_counts_work_inside_a_running_event_loop(monkeypatch, tmp_path):
    from dataset_agent.adapters import resource_map as rm

    monkeypatch.setattr(rm, "gather_pages_with_sources", lambda *a, **k: (PAGES, ["p"]))
    dsl = Dsl()
    uc = DatasetResearchUseCase(Agent(), MagicMock(), MagicMock(),
                                Settings(output_dir=str(tmp_path), resource_map_enabled=True),
                                dsl_port=dsl)

    async def inside_loop():
        return uc._resource_map(_rec())

    m = asyncio.run(inside_loop())
    assert dsl.calls and m is not None and m.data_resource_names == ["Example Workbench"]


def test_run_full_pipeline_passes_the_map_to_optimize(monkeypatch, tmp_path):
    rec = _rec()
    rec.resource_map = ResourceMap(program_names=["Example Program"],
                                   data_resource_names=["Example Workbench"])
    seen = {}

    class Research:
        def execute(self, _req):
            return rec, tmp_path / "r.json"

    class Optimize:
        async def execute(self, req):
            seen["req"] = req
            return MagicMock()

    monkeypatch.setattr(bootstrap, "_build_agent", lambda s: MagicMock())
    monkeypatch.setattr(bootstrap, "build_use_case", lambda *a, **k: Research())
    monkeypatch.setattr(bootstrap, "build_optimize_use_case", lambda *a, **k: Optimize())
    monkeypatch.setattr(bootstrap, "JsonDatasetRepository",
                        lambda d: MagicMock(save=lambda r: tmp_path / "x.json"))
    s = Settings(output_dir=str(tmp_path), resource_map_enabled=True,
                 dimensions_api_key="k")
    from dataset_agent.domain.models import ResearchRequest

    asyncio.run(bootstrap.run_full_pipeline(ResearchRequest(dataset_name="Example Program"), s))
    req = seen["req"]
    assert req.resource_map is rec.resource_map and req.dataset_description == "A cohort."

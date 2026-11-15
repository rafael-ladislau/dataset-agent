"""Round 4, items 6 and 7: the scope check and the full-text referent judge report the
configuration they ran with, and whether it is the measured one (budget 4096, temperature
0.6, top_p 0.95, top_k 20; one ask for the scope check)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import OptimizeRequest
from dataset_agent.settings import Settings
from tests.test_fulltext_loop import CountDSL, FakeClient, make_settings


def _anthropic(tmp_path, **kw):
    return make_settings(tmp_path, llm_provider="ollama", referent_thinking_budget=4096,
                         referent_temperature=0.6, referent_top_p=0.95, referent_top_k=20, **kw)


def _uc(settings, referent=True):
    return QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                    settings=settings, extraction_client=FakeClient([{}]),
                                    referent_agent=MagicMock() if referent else None)


def test_default_scope_check_runs_on_the_measured_configuration(tmp_path):
    cfg = _uc(_anthropic(tmp_path))._scope_check_config()
    assert cfg == {"agent": "referent", "api_style": "anthropic", "fallback": False,
                   "thinking_budget": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                   "votes": 1, "measured_configuration": True}


def test_scope_check_on_the_main_agent_is_not_the_measured_configuration(tmp_path):
    cfg = _uc(make_settings(tmp_path, referent_thinking_budget=0), referent=False)._scope_check_config()
    assert cfg["agent"] == "main" and cfg["thinking_budget"] == 0
    assert cfg["measured_configuration"] is False


def test_more_scope_votes_are_not_the_measured_configuration(tmp_path):
    cfg = _uc(_anthropic(tmp_path, scope_votes=3))._scope_check_config()
    assert cfg["votes"] == 3 and cfg["measured_configuration"] is False


def test_the_record_and_notes_carry_the_scope_check_configuration(tmp_path, monkeypatch):
    from dataset_agent.domain.models import DatasetRecord, Group, ResourceMap, YearsRange

    rec = DatasetRecord(
        engine="dimensions", group=Group(name="default"), main_dataset_name="Example Program NIH",
        description="A cohort.", dataset_names=["Example Program NIH"], flag_terms=["National Agency"],
        years_range=YearsRange(start_year=2015, end_year=2026),
        resource_map=ResourceMap(program_names=["Example Program NIH"],
                                 data_resource_names=["Example Workbench"]),
    )
    uc = _uc(_anthropic(tmp_path))

    async def stored(_req):
        return rec.model_copy(deep=True)

    uc._run_research = stored
    out = asyncio.run(uc.execute(OptimizeRequest(dataset_name="Example Program NIH")))
    assert out.scope_check_config is not None
    assert out.scope_check_config["measured_configuration"] is True
    assert "scope check configuration: referent agent, thinking budget 4096" in out.notes
    assert "(the measured configuration)" in out.notes


def test_settings_default_votes_are_unchanged():
    assert Settings.model_fields["scope_votes"].default == 1

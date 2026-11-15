"""Per-request sampling overrides on the research API (ported from main's ResearchRequest
model / temperature / top_k / top_p): bounds, per-request scope, what is recorded, and
that the referent judge and scope check keep their own configuration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dataset_agent import bootstrap
from dataset_agent.adapters.agent_anthropic import AnthropicAgent
from dataset_agent.adapters.agent_openai import OpenAICompatibleAgent
from dataset_agent.domain.models import DatasetRecord, Group, ResearchRequest, YearsRange
from dataset_agent.interfaces import api
from dataset_agent.settings import Settings

_PREFIXED = ("URL", "LLM_PROVIDER", "LLM_API_STYLE", "LLM_MODEL", "LLM_API_KEY",
             "LMSTUDIO_BASE_URL", "LMSTUDIO_MODEL", "REFERENT_THINKING_BUDGET")


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LMSTUDIO_BASE_URL", "LMSTUDIO_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name in _PREFIXED:
        monkeypatch.delenv(f"DATASET_AGENT_{name}", raising=False)


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def _anthropic() -> Settings:
    return _settings(llm_provider="lmstudio", llm_api_style="anthropic")


def _openai() -> Settings:
    return _settings(llm_provider="lmstudio", llm_api_style="openai")


# --- request fields and bounds ---------------------------------------------------------


def test_overrides_default_to_none():
    req = ResearchRequest(dataset_name="X")
    assert (req.model, req.temperature, req.top_p, req.top_k) == (None, None, None, None)


@pytest.mark.parametrize("field,value", [
    ("temperature", -0.1), ("temperature", 2.01), ("top_p", -0.01), ("top_p", 0.0),
    ("top_p", 1.01), ("top_k", 0), ("model", "   "),
])
def test_out_of_bounds_overrides_are_rejected(field, value):
    with pytest.raises(ValidationError):
        ResearchRequest(dataset_name="X", **{field: value})


@pytest.mark.parametrize("field,value", [
    ("temperature", 0.0), ("temperature", 2.0), ("top_p", 0.01), ("top_p", 1.0), ("top_k", 1),
])
def test_bounds_are_inclusive(field, value):
    assert getattr(ResearchRequest(dataset_name="X", **{field: value}), field) == value


def test_model_override_is_stripped():
    assert ResearchRequest(dataset_name="X", model="  qwen3.6  ").model == "qwen3.6"


def test_post_tasks_rejects_top_p_zero_at_request_time(tmp_path: Path, monkeypatch):
    """vLLM rejects top_p 0 only after the task is queued; the request is refused."""
    api.app.dependency_overrides[api.get_settings] = lambda: Settings(
        tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out").model_copy(
        update={"api_keys": ""})
    try:
        r = TestClient(api.app).post("/tasks", json={"dataset_name": "X", "top_p": 0})
    finally:
        api.app.dependency_overrides.clear()
    assert r.status_code == 422


def test_post_tasks_rejects_an_out_of_bounds_override(tmp_path: Path, monkeypatch):
    api.app.dependency_overrides[api.get_settings] = lambda: Settings(
        tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out").model_copy(
        update={"api_keys": ""})
    try:
        r = TestClient(api.app).post("/tasks", json={"dataset_name": "X", "temperature": 3})
    finally:
        api.app.dependency_overrides.clear()
    assert r.status_code == 422


# --- the agent a request runs with ------------------------------------------------------


def test_no_override_keeps_the_default_agent_and_records_it():
    s = _anthropic()
    agent, sampling = bootstrap.build_request_agent(s, ResearchRequest(dataset_name="X"))
    default = bootstrap._build_agent(s)
    assert isinstance(agent, AnthropicAgent)
    for attr in ("_model_name", "_temperature", "_top_p", "_top_k"):
        assert getattr(agent, attr) == getattr(default, attr)
    assert sampling.model == s.llm_endpoint().model
    assert sampling.temperature == default._temperature == 0.6
    assert (sampling.top_p, sampling.top_k) == (None, None)
    assert sampling.overridden == [] and sampling.not_applied == []
    assert sampling.api_style == "anthropic"


def test_anthropic_style_applies_every_override():
    req = ResearchRequest(dataset_name="X", model="m-1", temperature=0.2, top_p=0.8, top_k=30)
    agent, sampling = bootstrap.build_request_agent(_anthropic(), req)
    assert (agent._model_name, agent._temperature, agent._top_p, agent._top_k) == (
        "m-1", 0.2, 0.8, 30)
    assert (sampling.model, sampling.temperature, sampling.top_p, sampling.top_k) == (
        "m-1", 0.2, 0.8, 30)
    assert sampling.overridden == ["model", "temperature", "top_p", "top_k"]
    assert sampling.not_applied == []


def test_openai_style_records_top_k_as_not_applied():
    req = ResearchRequest(dataset_name="X", temperature=0.3, top_p=0.9, top_k=30)
    agent, sampling = bootstrap.build_request_agent(_openai(), req)
    assert isinstance(agent, OpenAICompatibleAgent)
    assert (agent._temperature, agent._top_p) == (0.3, 0.9)
    assert sampling.top_k is None
    assert sampling.not_applied == ["top_k"]
    assert sampling.overridden == ["temperature", "top_p"]


def test_openai_adapter_sends_top_p_only_when_set():
    sent = []

    def fake_create(**kw):
        sent.append(kw)
        return MagicMock()

    for top_p in (None, 0.9):
        agent = OpenAICompatibleAgent(model_name="m", base_url="http://127.0.0.1:9/v1",
                                      api_key="k", top_p=top_p)
        agent._client = MagicMock()
        agent._client.chat.completions.create = fake_create
        agent._create([{"role": "user", "content": "hi"}])
    assert "top_p" not in sent[0]
    assert sent[1]["top_p"] == 0.9


def test_an_override_never_leaks_into_the_next_request():
    s = _anthropic()
    before = s.model_dump()
    bootstrap.build_request_agent(
        s, ResearchRequest(dataset_name="X", model="m-1", temperature=1.5, top_p=0.5, top_k=5))
    agent, sampling = bootstrap.build_request_agent(s, ResearchRequest(dataset_name="Y"))
    assert s.model_dump() == before
    assert agent._model_name == s.llm_endpoint().model
    assert (agent._temperature, agent._top_p, agent._top_k) == (0.6, None, None)
    assert sampling.overridden == []


# --- the full pipeline ------------------------------------------------------------------


def _run(monkeypatch, tmp_path, settings, request):
    seen: dict = {}
    rec = DatasetRecord(engine="dimensions", group=Group(name="default"),
                        main_dataset_name="X", description="d",
                        years_range=YearsRange(start_year=2015, end_year=2026))

    class Research:
        def execute(self, _req):
            return rec, tmp_path / "r.json"

    class Optimize:
        async def execute(self, _req):
            return MagicMock()

    def fake_build_use_case(_s, agent=None, dsl_port=None):
        seen["research_agent"] = agent
        return Research()

    def fake_build_optimize(_s, agent=None, dsl_port=None, referent_agent=None):
        seen["optimize_agent"] = agent
        seen["referent_agent"] = referent_agent
        return Optimize()

    monkeypatch.setattr(bootstrap, "build_use_case", fake_build_use_case)
    monkeypatch.setattr(bootstrap, "build_optimize_use_case", fake_build_optimize)
    monkeypatch.setattr(bootstrap, "JsonDatasetRepository",
                        lambda d: MagicMock(save=lambda r: tmp_path / "x.json"))
    record, _ = asyncio.run(bootstrap.run_full_pipeline(request, settings))
    return record, seen


@pytest.mark.parametrize("budget", [0, 4096])
def test_overrides_reach_research_and_optimize_but_not_the_referent_or_scope(
        monkeypatch, tmp_path, budget):
    s = _settings(llm_provider="lmstudio", llm_api_style="anthropic", output_dir=str(tmp_path),
                  dimensions_api_key="k", referent_thinking_budget=budget)
    req = ResearchRequest(dataset_name="X", model="m-1", temperature=1.2, top_p=0.7, top_k=9)
    record, seen = _run(monkeypatch, tmp_path, s, req)
    assert seen["research_agent"] is seen["optimize_agent"]
    assert seen["research_agent"]._temperature == 1.2
    ref = seen["referent_agent"]
    assert ref is not None and ref is not seen["research_agent"]
    assert ref._model_name == s.llm_endpoint().model
    expected = bootstrap.build_referent_agent(s)
    assert (ref._temperature, ref._top_p, ref._top_k) == (
        expected._temperature, expected._top_p, expected._top_k)
    assert record.llm_sampling is not None
    assert (record.llm_sampling.model, record.llm_sampling.temperature) == ("m-1", 1.2)
    assert "llm_sampling" in record.model_dump(mode="json")


def test_without_overrides_the_pipeline_is_wired_as_before(monkeypatch, tmp_path):
    s = _settings(llm_provider="lmstudio", llm_api_style="anthropic", output_dir=str(tmp_path),
                  dimensions_api_key="k", referent_thinking_budget=0)
    record, seen = _run(monkeypatch, tmp_path, s, ResearchRequest(dataset_name="X"))
    assert seen["referent_agent"] is None
    assert record.llm_sampling.overridden == []
    assert record.llm_sampling.temperature == 0.6

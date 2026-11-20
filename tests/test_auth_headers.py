"""Which auth header each API style sends with the configured key.

Sandstone (LiteLLM) takes ``x-api-key`` on /v1/messages and Bearer on /v1/chat/completions;
a keyed vLLM (``--api-key``) takes only ``Authorization: Bearer`` on both.
"""

from __future__ import annotations

import pytest

from dataset_agent.adapters.agent_anthropic import AnthropicAgent
from dataset_agent.adapters.agent_openai import OpenAICompatibleAgent
from dataset_agent.bootstrap import _build_agent, build_referent_agent, build_request_agent
from dataset_agent.domain.models import ResearchRequest
from dataset_agent.settings import Settings
from tests.llm_fakes import ANTHROPIC, OPENAI, FakeLLMServer

KEY = "sk-test-secret"
TOOL = {"name": "emit", "description": "d",
        "input_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                         "required": ["ok"]}}


def _anthropic(server, **kw):
    return AnthropicAgent(model_name="m", base_url="http://x", api_key=KEY, max_tokens=64,
                          http_client=server.http_client(), max_retries=0, **kw)


def _openai(server):
    return OpenAICompatibleAgent(model_name="m", base_url="http://x/v1", api_key=KEY,
                                 max_tokens=64, http_client=server.http_client(), max_retries=0)


def _sent(server, call):
    server.push(server.text("hi"))
    call()
    return server.headers[0]


def test_anthropic_style_sends_x_api_key_not_bearer_by_default():
    server = FakeLLMServer(ANTHROPIC)
    h = _sent(server, lambda: _anthropic(server).get_information("q", tools=[]))
    assert h["x-api-key"] == KEY
    assert "authorization" not in h
    assert server.paths[0] == "/v1/messages"


def test_openai_style_sends_bearer_only():
    server = FakeLLMServer(OPENAI)
    h = _sent(server, lambda: _openai(server).get_information("q", tools=[]))
    assert h["authorization"] == f"Bearer {KEY}"
    assert "x-api-key" not in h
    assert server.paths[0] == "/v1/chat/completions"


def test_anthropic_bearer_mode_sends_bearer_not_x_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    server = FakeLLMServer(ANTHROPIC)
    h = _sent(server, lambda: _anthropic(server, auth_mode="bearer").get_information("q", tools=[]))
    assert h["authorization"] == f"Bearer {KEY}"
    assert "x-api-key" not in h


def test_anthropic_bearer_mode_forced_tool_call_still_works(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    server = FakeLLMServer(ANTHROPIC)
    server.push(server.structured("emit", {"ok": True}))
    agent = _anthropic(server, auth_mode="bearer")
    assert agent.get_structured("q", TOOL) == {"ok": True}
    assert server.headers[0]["authorization"] == f"Bearer {KEY}"
    assert server.requests[0]["tool_choice"] == {"type": "tool", "name": "emit"}


def test_unknown_auth_mode_is_rejected():
    with pytest.raises(ValueError):
        AnthropicAgent(model_name="m", base_url="http://x", api_key=KEY, auth_mode="basic")


def _settings(tmp_path, **over):
    return Settings(tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out", _env_file=None,
                    llm_provider="openai_compatible", openai_compatible_base_url="http://x",
                    openai_compatible_model="m", openai_compatible_api_key=KEY, **over)


def test_auth_mode_defaults_to_x_api_key(tmp_path):
    assert _settings(tmp_path).llm_auth_mode == "x-api-key"


def test_auth_mode_reads_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_AGENT_LLM_AUTH_MODE", "bearer")
    assert _settings(tmp_path).llm_auth_mode == "bearer"


def _auth_of(agent):
    return agent._auth_mode


def test_every_anthropic_construction_gets_the_auth_mode(tmp_path):
    s = _settings(tmp_path, llm_api_style="anthropic", llm_auth_mode="bearer",
                  referent_thinking_budget=1024)
    assert _auth_of(_build_agent(s)) == "bearer"
    assert _auth_of(build_referent_agent(s)) == "bearer"
    request = ResearchRequest(dataset_name="x", temperature=0.2)
    agent, _ = build_request_agent(s, request)
    assert _auth_of(agent) == "bearer"
    # no override: falls back to the main agent
    agent, _ = build_request_agent(s, ResearchRequest(dataset_name="x"))
    assert _auth_of(agent) == "bearer"


def test_openai_style_ignores_the_auth_mode(tmp_path):
    s = _settings(tmp_path, llm_api_style="openai", llm_auth_mode="bearer")
    assert isinstance(_build_agent(s), OpenAICompatibleAgent)
    agent, _ = build_request_agent(s, ResearchRequest(dataset_name="x", temperature=0.2))
    assert isinstance(agent, OpenAICompatibleAgent)

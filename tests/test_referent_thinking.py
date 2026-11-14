"""Thinking for the referent question only (a scoped setting)."""

from __future__ import annotations

import pytest

from dataset_agent.adapters.agent_anthropic import AnthropicAgent
from dataset_agent.bootstrap import _build_agent, build_referent_agent
from dataset_agent.settings import Settings
from tests.llm_fakes import ANTHROPIC, FakeLLMServer, hx

TOOL = {"name": "emit", "description": "d",
        "input_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                         "required": ["ok"]}}


def _agent(server, **kw):
    return AnthropicAgent(model_name="m", base_url="http://x", api_key="k", max_tokens=512,
                          http_client=server.http_client(), max_retries=0, **kw)


def _thinking_then_tool(obj):
    from tests.llm_fakes import _anthropic_message
    blocks = [{"type": "thinking", "thinking": "hmm", "signature": "s"},
              {"type": "tool_use", "id": "t1", "name": "emit", "input": obj}]
    return hx.Response(200, json=_anthropic_message(blocks, "tool_use"))


def test_structured_call_with_thinking_sends_the_recipe_and_reads_past_the_thinking_block():
    server = FakeLLMServer(ANTHROPIC)
    server.push(_thinking_then_tool({"ok": False}))
    agent = _agent(server, temperature=0.6, thinking_budget=4096, top_p=0.95, top_k=20)
    assert agent.get_structured("q", TOOL) == {"ok": False}
    body = server.requests[0]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert body["max_tokens"] == 512 + 4096
    # A forced tool is not allowed together with thinking.
    assert body["tool_choice"] == {"type": "auto"}
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.6, 0.95, 20)


def test_structured_call_without_thinking_is_unchanged():
    server = FakeLLMServer(ANTHROPIC)
    server.push(server.structured("emit", {"ok": True}))
    agent = _agent(server)
    assert agent.get_structured("q", TOOL) == {"ok": True}
    body = server.requests[0]
    assert "thinking" not in body and "top_p" not in body and "top_k" not in body
    assert body["tool_choice"] == {"type": "tool", "name": "emit"}
    assert body["max_tokens"] == 512


def _settings(tmp_path, **over):
    return Settings(tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out", _env_file=None,
                    llm_provider="openai_compatible", llm_api_style="anthropic",
                    openai_compatible_base_url="http://x", openai_compatible_model="m",
                    openai_compatible_api_key="k", **over)


def test_referent_agent_is_the_main_agent_when_thinking_is_off(tmp_path):
    s = _settings(tmp_path)
    a = build_referent_agent(s)
    assert type(a) is type(_build_agent(s)) and a._thinking_budget == 0


def test_referent_agent_thinks_only_when_its_setting_says_so(tmp_path):
    s = _settings(tmp_path, referent_thinking_budget=4096, referent_temperature=0.6,
                  referent_top_p=0.95, referent_top_k=20)
    a = build_referent_agent(s)
    assert (a._thinking_budget, a._temperature, a._top_p, a._top_k) == (4096, 0.6, 0.95, 20)
    assert _build_agent(s)._thinking_budget == 0  # every other LLM step stays thinking off


def test_referent_thinking_needs_the_anthropic_style(tmp_path):
    s = _settings(tmp_path, referent_thinking_budget=4096)
    s = s.model_copy(update={"llm_api_style": "openai"})
    with pytest.raises(ValueError, match="anthropic"):
        build_referent_agent(s)

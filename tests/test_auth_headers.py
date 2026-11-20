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



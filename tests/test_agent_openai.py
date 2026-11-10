"""OpenAI-compatible adapter specifics: structured tiers, tier cache, reasoning output."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from llm_fakes import OPENAI, FakeLLMServer, make_agent

from dataset_agent.adapters import tools
from dataset_agent.adapters.agent_openai import (
    _TIER_CACHE,
    TIER_JSON_SCHEMA,
    TIER_TEXT_JSON,
    TIER_TOOL_CALL,
    reset_tier_cache,
)

RESULT_TOOL = {
    "name": "emit_names",
    "description": "Return the names.",
    "input_schema": {
        "type": "object",
        "properties": {"names": {"type": "array", "items": {"type": "string"}}},
        "required": ["names"],
        "additionalProperties": False,
    },
}
VALID = {"names": ["ACS"]}
CACHE_KEY = ("http://fake-llm/v1", "fake-model")


@pytest.fixture(autouse=True)
def _clean_tier_cache():
    reset_tier_cache()
    yield
    reset_tier_cache()


@pytest.fixture
def server() -> FakeLLMServer:
    return FakeLLMServer(OPENAI)


def _tier_of(body: dict[str, Any]) -> str:
    if "response_format" in body:
        return TIER_JSON_SCHEMA
    if "tool_choice" in body:
        return TIER_TOOL_CALL
    return TIER_TEXT_JSON


def test_json_schema_tier_request_shape(server):
    agent = make_agent(OPENAI, server)
    server.push(server.text(json.dumps(VALID)))

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID

    body = server.requests[0]
    assert server.paths == ["/v1/chat/completions"]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "emit_names",
            "schema": RESULT_TOOL["input_schema"],
            "strict": False,
        },
    }
    assert body["messages"][0]["role"] == "system"
    assert _TIER_CACHE[CACHE_KEY] == TIER_JSON_SCHEMA


def test_400_on_json_schema_falls_through_to_tool_call(server):
    agent = make_agent(OPENAI, server)
    server.push(server.error(400), server.tool_call("emit_names", VALID))

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID

    assert [_tier_of(b) for b in server.requests] == [TIER_JSON_SCHEMA, TIER_TOOL_CALL]
    assert server.requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "emit_names"},
    }
    assert server.requests[1]["tools"][0]["function"]["parameters"] == RESULT_TOOL["input_schema"]
    assert _TIER_CACHE[CACHE_KEY] == TIER_TOOL_CALL


def test_text_tier_parses_fenced_json(server):
    agent = make_agent(OPENAI, server)
    server.push(
        server.error(400),
        server.error(422),
        server.text('Here you go:\n```json\n{"names": ["ACS"]}\n```'),
    )

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID

    assert [_tier_of(b) for b in server.requests] == [
        TIER_JSON_SCHEMA,
        TIER_TOOL_CALL,
        TIER_TEXT_JSON,
    ]
    assert "tools" not in server.requests[2]
    assert "JSON Schema" in server.user_texts(2)[-1]
    assert _TIER_CACHE[CACHE_KEY] == TIER_TEXT_JSON


def test_tier_retries_once_then_falls_to_next_tier(server):
    agent = make_agent(OPENAI, server)
    server.push(
        server.text("not json"),
        server.text('{"names": 3}'),
        server.tool_call("emit_names", VALID),
    )

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID
    assert [_tier_of(b) for b in server.requests] == [
        TIER_JSON_SCHEMA,
        TIER_JSON_SCHEMA,
        TIER_TOOL_CALL,
    ]
    assert "did not contain a JSON object" in server.user_texts(1)[-1]


def test_invalid_tool_arguments_retry_in_tool_tier(server):
    agent = make_agent(OPENAI, server)
    bad = server.tool_call("emit_names", VALID)
    payload = bad.json()
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    server.push(
        server.error(400),
        type(bad)(200, json=payload),
        server.tool_call("emit_names", VALID),
    )

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID
    assert "not valid JSON" in server.user_texts(2)[-1]


def test_cached_tier_is_used_first(server):
    agent = make_agent(OPENAI, server)
    server.push(server.error(400), server.tool_call("emit_names", VALID))
    agent.get_structured("Names?", RESULT_TOOL)

    server.requests.clear()
    server.push(server.tool_call("emit_names", VALID))
    assert agent.get_structured("Again?", RESULT_TOOL) == VALID
    assert [_tier_of(b) for b in server.requests] == [TIER_TOOL_CALL]

    # The cache is per (base_url, model): another model starts from the top tier.
    other = make_agent(OPENAI, server, model="other-model")
    server.requests.clear()
    server.push(server.text(json.dumps(VALID)))
    assert other.get_structured("Names?", RESULT_TOOL) == VALID
    assert [_tier_of(b) for b in server.requests] == [TIER_JSON_SCHEMA]


def test_server_error_aborts_without_fallthrough(server, caplog):
    agent = make_agent(OPENAI, server)
    server.push(server.error(500, "boom"))

    with caplog.at_level(logging.ERROR):
        assert agent.get_structured("Names?", RESULT_TOOL) == {}
    assert len(server.requests) == 1
    assert agent.structured_failures == 1


def test_structured_strips_think_before_parsing(server):
    agent = make_agent(OPENAI, server)
    server.push(server.text('<think>{"names": "draft"}</think>{"names": ["ACS"]}'))

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID


def test_reasoning_fields_are_ignored(server):
    agent = make_agent(OPENAI, server)
    server.push(
        server.text("The answer.", reasoning_content="secret chain", reasoning="more secrets")
    )

    assert agent.get_information("Q?", tools=[]) == "The answer."


def test_empty_content_returns_placeholder(server):
    agent = make_agent(OPENAI, server)
    server.push(server.text("<think>only reasoning, never finished"))

    assert agent.get_information("Q?", tools=[]) == "No output from agent."


def test_tool_call_written_as_text_is_not_executed(server, monkeypatch):
    calls: list[dict[str, Any]] = []
    monkeypatch.setitem(tools.TOOL_DISPATCH, "web_search", lambda a: calls.append(a) or "x")
    agent = make_agent(OPENAI, server)
    text_call = '{"name": "web_search", "arguments": {"query": "acs"}}'
    server.push(server.text(text_call))

    assert agent.get_information("Search.") == text_call
    assert calls == []
    assert len(server.requests) == 1


def test_tool_loop_strips_think_from_assistant_history(server, monkeypatch):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "web_search", lambda a: "results")
    agent = make_agent(OPENAI, server)
    first = server.tool_call("web_search", {"query": "acs"})
    payload = first.json()
    payload["choices"][0]["message"]["content"] = "<think>plan</think>"
    server.push(type(first)(200, json=payload), server.text("done"))

    assert agent.get_information("Search.") == "done"
    assistant = server.requests[1]["messages"][2]
    assert assistant["role"] == "assistant"
    assert assistant.get("content") is None
    assert assistant["tool_calls"][0]["function"]["name"] == "web_search"

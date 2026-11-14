"""OpenAI-compatible adapter specifics: structured tiers, last-tier tracking, reasoning output."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from llm_fakes import OPENAI, FakeLLMServer, make_agent

from dataset_agent.adapters import tools
from dataset_agent.adapters.agent_openai import (
    TIER_JSON_SCHEMA,
    TIER_TEXT_JSON,
    TIER_TOOL_CALL,
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


def test_400_on_json_schema_falls_through_to_tool_call(server):
    agent = make_agent(OPENAI, server)
    server.push(server.error(400), server.tool_call("emit_names", VALID))

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID

    assert [_tier_of(b) for b in server.requests] == [TIER_JSON_SCHEMA, TIER_TOOL_CALL]
    assert server.requests[1]["tool_choice"] == "required"
    assert server.requests[1]["tools"][0]["function"]["parameters"] == RESULT_TOOL["input_schema"]


def test_text_tier_parses_fenced_json(server):
    agent = make_agent(OPENAI, server)
    server.push(
        server.error(400),
        server.error(422),
        server.error(422),
        server.text('Here you go:\n```json\n{"names": ["ACS"]}\n```'),
    )

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID

    assert [_tier_of(b) for b in server.requests] == [
        TIER_JSON_SCHEMA,
        TIER_TOOL_CALL,
        TIER_TOOL_CALL,
        TIER_TEXT_JSON,
    ]
    assert "tools" not in server.requests[3]
    assert "JSON Schema" in server.user_texts(3)[-1]


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


def test_empty_content_returns_empty_string(server):
    agent = make_agent(OPENAI, server)
    server.push(server.text("<think>only reasoning, never finished"))

    assert agent.get_information("Q?", tools=[]) == ""


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


OTHER_TOOL = {
    "name": "emit_terms",
    "description": "Return terms.",
    "input_schema": {
        "type": "object",
        "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
        "required": ["terms"],
    },
}


def _with_dict_arguments(reply, args):
    payload = reply.json()
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = args
    return type(reply)(200, json=payload)


def test_tool_call_arguments_as_dict_are_accepted(server, monkeypatch):
    calls: list[dict[str, Any]] = []
    monkeypatch.setitem(tools.TOOL_DISPATCH, "web_search", lambda a: calls.append(a) or "r")
    agent = make_agent(OPENAI, server)
    server.push(
        _with_dict_arguments(server.tool_call("web_search", {}), {"query": "acs"}),
        server.text("done"),
        server.error(400),
        _with_dict_arguments(server.tool_call("emit_names", {}), VALID),
    )

    assert agent.get_information("Search.") == "done"
    assert calls == [{"query": "acs"}]
    assert agent.get_structured("Names?", RESULT_TOOL) == VALID


def test_every_tier_rejected_returns_empty(server):
    agent = make_agent(OPENAI, server)
    server.push(*[server.error(400) for _ in range(4)])

    assert agent.get_structured("Names?", RESULT_TOOL) == {}
    assert agent.last_structured_tier is None


def test_last_structured_tier_tracks_the_returned_object(server):
    agent = make_agent(OPENAI, server)
    assert agent.last_structured_tier is None
    server.push(server.text(json.dumps(VALID)))
    agent.get_structured("Names?", RESULT_TOOL)
    assert agent.last_structured_tier == TIER_JSON_SCHEMA

    server.push(server.error(500))
    assert agent.get_structured("Names?", RESULT_TOOL) == {}
    assert agent.last_structured_tier is None


def test_json_schema_reply_in_reasoning_content_is_used(server):
    agent = make_agent(OPENAI, server)
    server.push(server.text("", reasoning_content=json.dumps(VALID)))

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID
    assert len(server.requests) == 1
    assert agent.last_structured_tier == TIER_JSON_SCHEMA
    assert agent.structured_failures == 0


def test_json_schema_reply_in_reasoning_field_is_used(server):
    agent = make_agent(OPENAI, server)
    server.push(server.text("", reasoning=json.dumps(VALID)))

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID
    assert len(server.requests) == 1


def test_reasoning_fallback_is_schema_validated(server):
    agent = make_agent(OPENAI, server)
    server.push(
        server.text("", reasoning_content='{"names": "draft"}'),
        server.text("", reasoning_content='{"names": "draft"}'),
        server.tool_call("emit_names", VALID),
    )

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID
    assert "schema validation failed" in server.user_texts(1)[-1]
    assert agent.last_structured_tier == TIER_TOOL_CALL


def test_reasoning_with_prose_is_not_parsed(server):
    agent = make_agent(OPENAI, server)
    server.push(
        server.text("", reasoning_content='Thinking... {"names": ["ACS"]} maybe'),
        server.text(json.dumps(VALID)),
    )

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID
    assert "did not contain a JSON object" in server.user_texts(1)[-1]


def test_content_json_wins_over_reasoning(server):
    agent = make_agent(OPENAI, server)
    server.push(server.text(json.dumps(VALID), reasoning_content='{"names": ["OTHER"]}'))

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID


def test_text_json_tier_ignores_reasoning_content(server):
    agent = make_agent(OPENAI, server)
    server.push(
        server.error(400),
        server.error(400),
        server.error(400),
        server.text("", reasoning_content=json.dumps(VALID)),
        server.text("", reasoning_content=json.dumps(VALID)),
    )

    assert agent.get_structured("Names?", RESULT_TOOL) == {}
    assert agent.last_structured_tier is None


def test_tool_tier_falls_back_to_object_tool_choice_on_400(server):
    agent = make_agent(OPENAI, server)
    server.push(
        server.error(400),
        server.error(400, "tool_choice must be an object"),
        server.tool_call("emit_names", VALID),
    )

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID
    assert server.requests[1]["tool_choice"] == "required"
    assert server.requests[2]["tool_choice"] == {
        "type": "function",
        "function": {"name": "emit_names"},
    }
    assert agent.last_structured_tier == TIER_TOOL_CALL


# ---------------------------------------------------------------------------
# Harmony tool calls leaked into message content (gpt-oss on LM Studio)
# ---------------------------------------------------------------------------

LEAKED = (
    '<|start|>assistant<|channel|>commentary to=functions.web_search '
    '<|constrain|>json<|message|>{"query":"earthquake gnss unavco dataset"}'
)


def test_leaked_harmony_tool_call_is_executed(server, monkeypatch):
    calls: list[dict[str, Any]] = []
    monkeypatch.setitem(tools.TOOL_DISPATCH, "web_search", lambda a: calls.append(a) or "results")
    agent = make_agent(OPENAI, server)
    server.push(server.text(LEAKED), server.text("GNSS data from UNAVCO."))

    assert agent.get_information("Search.") == "GNSS data from UNAVCO."
    assert calls == [{"query": "earthquake gnss unavco dataset"}]
    history = server.requests[1]["messages"]
    assistant, tool = history[2], history[3]
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["function"]["name"] == "web_search"
    assert tool["role"] == "tool" and tool["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert tool["content"] == "results"
    assert "<|" not in json.dumps(history)


def test_multiple_leaked_harmony_calls_are_all_executed(server, monkeypatch):
    calls: list[dict[str, Any]] = []
    monkeypatch.setitem(tools.TOOL_DISPATCH, "web_search", lambda a: calls.append(a) or "r")
    agent = make_agent(OPENAI, server)
    two = (
        LEAKED
        + '<|call|><|start|>assistant<|channel|>commentary to=functions.web_search '
        '<|constrain|>json<|message|>{"query":"second"}<|call|>'
    )
    server.push(server.text(two), server.text("done"))

    assert agent.get_information("Search.") == "done"
    assert [c["query"] for c in calls] == ["earthquake gnss unavco dataset", "second"]
    ids = [t["id"] for t in server.requests[1]["messages"][2]["tool_calls"]]
    assert len(set(ids)) == 2


def test_unparseable_harmony_retries_once_then_recovers(server, monkeypatch):
    calls: list[dict[str, Any]] = []
    monkeypatch.setitem(tools.TOOL_DISPATCH, "web_search", lambda a: calls.append(a) or "r")
    agent = make_agent(OPENAI, server)
    broken = "<|channel|>commentary to=functions.web_search <|message|>{not json"
    server.push(server.text(broken), server.text("A plain answer."))

    assert agent.get_information("Search.") == "A plain answer."
    assert calls == []
    assert "control markup" in server.requests[1]["messages"][-1]["content"]
    assert "<|" not in json.dumps(server.requests[1]["messages"][2:])


def test_persistent_harmony_leak_is_never_returned(server):
    agent = make_agent(OPENAI, server)
    broken = "<|channel|>commentary to=functions.web_search <|message|>{not json"
    server.push(server.text(broken), server.text(broken))

    assert agent.get_information("Search.") == ""
    assert len(server.requests) == 2


def test_harmony_call_for_unoffered_tool_is_not_executed(server, monkeypatch):
    calls: list[dict[str, Any]] = []
    monkeypatch.setitem(tools.TOOL_DISPATCH, "web_search", lambda a: calls.append(a) or "r")
    agent = make_agent(OPENAI, server)
    server.push(server.text(LEAKED), server.text(LEAKED))

    assert agent.get_information("Q?", tools=[]) == ""
    assert calls == []


def test_final_text_with_harmony_final_channel_is_stripped(server):
    agent = make_agent(OPENAI, server)
    server.push(
        server.text(
            "<|channel|>analysis<|message|>thinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>The answer.<|return|>"
        )
    )

    assert agent.get_information("Q?", tools=[]) == "The answer."


def test_summary_after_iteration_cap_is_sanitized(server, monkeypatch):
    monkeypatch.setitem(tools.TOOL_DISPATCH, "web_search", lambda a: "r")
    agent = make_agent(OPENAI, server, max_iterations=1)
    server.push(server.tool_call("web_search", {"query": "x"}), server.text(LEAKED))

    assert agent.get_information("Search.") == ""


def test_structured_text_json_with_harmony_markup_is_rejected_and_retried(server):
    agent = make_agent(OPENAI, server)
    server.push(
        server.error(400),
        server.error(400),
        server.error(400),
        server.text(
            '<|channel|>final<|message|>{"names": ["<|start|>bad"]}<|return|>'
        ),
        server.text('<|channel|>final<|message|>{"names": ["ACS"]}<|return|>'),
    )

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID

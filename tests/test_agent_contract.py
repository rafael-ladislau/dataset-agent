"""AgentPort contract tests run against both LLM adapters with fake servers."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from llm_fakes import ANTHROPIC, OPENAI, STYLES, FakeLLMServer, make_agent

from dataset_agent.adapters import tools
from dataset_agent.adapters.agent_openai import reset_tier_cache

LOOKUP_TOOL = {
    "name": "lookup",
    "description": "Look something up.",
    "input_schema": {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    },
}

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
VALID = {"names": ["ACS", "American Community Survey"]}
INVALID = {"names": "ACS"}


@pytest.fixture(autouse=True)
def _clean_tier_cache():
    reset_tier_cache()
    yield
    reset_tier_cache()


@pytest.fixture
def lookup_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _lookup(args: dict[str, Any]) -> str:
        calls.append(args)
        return "LOOKUP-RESULT"

    monkeypatch.setitem(tools.TOOL_DISPATCH, "lookup", _lookup)
    return calls


@pytest.fixture(params=STYLES)
def style(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def server(style: str) -> FakeLLMServer:
    return FakeLLMServer(style)


def _tool_names_sent(server: FakeLLMServer, index: int) -> list[str]:
    sent = server.requests[index].get("tools") or []
    if server.style == ANTHROPIC:
        return [t["name"] for t in sent]
    return [t["function"]["name"] for t in sent]


# ---------------------------------------------------------------------------
# get_information
# ---------------------------------------------------------------------------


def test_tool_loop_runs_tool_and_returns_final_text(server, style, lookup_calls):
    agent = make_agent(style, server)
    server.push(server.tool_call("lookup", {"q": "acs"}), server.text("Final answer."))

    out = agent.get_information("Find ACS.", tools=[LOOKUP_TOOL])

    assert out == "Final answer."
    assert lookup_calls == [{"q": "acs"}]
    assert len(server.requests) == 2
    assert _tool_names_sent(server, 0) == ["lookup"]
    last = server.requests[1]["messages"][-1]
    if style == OPENAI:
        assert server.requests[0]["tools"][0] == {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up.",
                "parameters": LOOKUP_TOOL["input_schema"],
            },
        }
        assert last == {"role": "tool", "tool_call_id": "call_1", "content": "LOOKUP-RESULT"}
    else:
        assert last["content"][0]["type"] == "tool_result"
        assert last["content"][0]["content"] == "LOOKUP-RESULT"


def test_iteration_cap_requests_final_summary_without_tools(server, style, lookup_calls):
    agent = make_agent(style, server, max_iterations=2)
    server.push(
        server.tool_call("lookup", {"q": "one"}, call_id="c1"),
        server.tool_call("lookup", {"q": "two"}, call_id="c2"),
        server.text("Summary after cap."),
    )

    out = agent.get_information("Loop forever.", tools=[LOOKUP_TOOL])

    assert out == "Summary after cap."
    assert [c["q"] for c in lookup_calls] == ["one", "two"]
    assert len(server.requests) == 3
    assert "tools" not in server.requests[2]


def test_empty_tool_list_sends_no_tools(server, style):
    agent = make_agent(style, server)
    server.push(server.text("Just reasoning."))

    assert agent.get_information("Think.", tools=[]) == "Just reasoning."
    assert "tools" not in server.requests[0]


def test_default_tools_are_web_search_and_make_request(server, style):
    agent = make_agent(style, server)
    server.push(server.text("ok"))

    agent.get_information("Hi.")

    assert _tool_names_sent(server, 0) == ["web_search", "make_request"]


def test_think_blocks_are_stripped_from_text(server, style):
    agent = make_agent(style, server)
    server.push(server.text("<think>private reasoning</think>\nThe answer."))

    assert agent.get_information("Q?", tools=[]) == "The answer."


def test_timeout_returns_error_text(server, style):
    agent = make_agent(style, server, timeout_seconds=1)
    server.push(server.timeout())

    out = agent.get_information("Slow.", tools=[])

    assert out.startswith("Error in agent execution:")


# ---------------------------------------------------------------------------
# get_structured
# ---------------------------------------------------------------------------


def test_structured_happy_path(server, style):
    agent = make_agent(style, server)
    server.push(server.structured("emit_names", VALID))

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID
    assert len(server.requests) == 1
    assert agent.structured_failures == 0


def test_structured_invalid_then_retry_succeeds(server, style):
    agent = make_agent(style, server)
    server.push(
        server.structured("emit_names", INVALID),
        server.structured("emit_names", VALID),
    )

    assert agent.get_structured("Names?", RESULT_TOOL) == VALID
    assert len(server.requests) == 2
    retry_text = server.user_texts(1)[-1]
    assert "rejected" in retry_text
    assert "schema validation failed" in retry_text
    assert agent.structured_failures == 0


def test_structured_total_failure_returns_empty_and_warns(server, style, caplog):
    agent = make_agent(style, server)
    attempts = 2 if style == ANTHROPIC else 6  # one retry per tier
    server.push(*[server.text("I cannot do that.") for _ in range(attempts)])

    with caplog.at_level(logging.WARNING):
        assert agent.get_structured("Names?", RESULT_TOOL) == {}

    assert len(server.requests) == attempts
    assert agent.structured_failures == 1
    assert any(
        r.levelno == logging.WARNING and "emit_names" in r.getMessage() for r in caplog.records
    )


def test_structured_timeout_returns_empty(server, style):
    agent = make_agent(style, server, timeout_seconds=1)
    server.push(server.timeout())

    assert agent.get_structured("Names?", RESULT_TOOL) == {}
    assert len(server.requests) == 1
    assert agent.structured_failures == 1


def test_structured_requires_named_result_tool(server, style):
    agent = make_agent(style, server)

    with pytest.raises(KeyError):
        agent.get_structured("Suffix?", {"type": "string"})


NEAR_MISS = {"names": "ACS", "note": "extra key the schema forbids"}
COERCED = {"names": ["ACS"]}
HOPELESS = {"names": 5}


def _near_miss_replies(server: FakeLLMServer, obj: dict[str, Any]) -> list[Any]:
    """Replies where every attempt yields *obj* or nothing parseable."""
    if server.style == ANTHROPIC:
        return [server.tool_call("emit_names", obj) for _ in range(2)]
    return [
        server.text(json.dumps(obj)),
        server.text(json.dumps(obj)),
        server.text("no tool call"),
        server.text("no tool call"),
        server.text("still nothing parseable"),
        server.text("still nothing parseable"),
    ]


def test_structured_near_miss_is_coerced_to_schema(server, style, caplog):
    agent = make_agent(style, server)
    server.push(*_near_miss_replies(server, NEAR_MISS))

    with caplog.at_level(logging.WARNING):
        assert agent.get_structured("Names?", RESULT_TOOL) == COERCED

    assert agent.structured_failures == 1
    assert any(r.levelno == logging.WARNING and "coerced" in r.getMessage() for r in caplog.records)


def test_structured_near_miss_still_invalid_returns_empty(server, style, caplog):
    agent = make_agent(style, server)
    server.push(*_near_miss_replies(server, HOPELESS))

    with caplog.at_level(logging.WARNING):
        assert agent.get_structured("Names?", RESULT_TOOL) == {}

    assert agent.structured_failures == 1
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_temperature_reaches_request_body(server, style):
    agent = make_agent(style, server, temperature=0.2)
    server.push(server.text("ok"), server.structured("emit_names", VALID))

    agent.get_information("Q?", tools=[])
    agent.get_structured("Names?", RESULT_TOOL)

    assert [r["temperature"] for r in server.requests] == [0.2, 0.2]

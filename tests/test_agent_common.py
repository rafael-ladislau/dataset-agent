"""Shared agent helpers: think stripping, JSON extraction, schema validation."""

from __future__ import annotations

import pytest

from dataset_agent.adapters import tools
from dataset_agent.adapters.agent_common import (
    extract_json_object,
    run_tool,
    strip_think,
    to_openai_tool,
    validate_against_schema,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain answer", "plain answer"),
        ("<think>a</think>answer", "answer"),
        ("<THINK>a\nb</THINK>\n answer \n", "answer"),
        ("one <think>x</think> two <think>y</think> three", "one  two  three"),
        ("reasoning without opening tag</think>answer", "answer"),
        ("answer<think>unfinished reasoning", "answer"),
        ("<think>only reasoning", ""),
        (None, ""),
    ],
)
def test_strip_think(raw, expected):
    assert strip_think(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure!\n```\n{"a": [1, 2]}\n```\nDone.', {"a": [1, 2]}),
        ('Result: {"a": "x}y", "b": {"c": 2}} trailing', {"a": "x}y", "b": {"c": 2}}),
        ('<think>{"draft": true}</think>{"final": true}', {"final": True}),
        ('{broken {"a": 1}', {"a": 1}),
        ('for {dataset}: {"names": ["a"]}', {"names": ["a"]}),
        ("[1, 2]", None),
        ("no json here", None),
        ("", None),
    ],
)
def test_extract_json_object(raw, expected):
    assert extract_json_object(raw) == expected


SCHEMA = {
    "type": "object",
    "properties": {"names": {"type": "array", "items": {"type": "string"}}},
    "required": ["names"],
    "additionalProperties": False,
}


def test_validate_against_schema():
    assert validate_against_schema({"names": ["a"]}, SCHEMA) is None
    assert "required" in validate_against_schema({}, SCHEMA)
    assert "names/0" in validate_against_schema({"names": [1]}, SCHEMA)
    assert validate_against_schema({"x": 1}, None) is None
    # An invalid schema is not enforced.
    assert validate_against_schema({"x": 1}, {"type": "not-a-type"}) is None


def test_to_openai_tool():
    tool = {"name": "t", "description": "d", "input_schema": SCHEMA}
    assert to_openai_tool(tool) == {
        "type": "function",
        "function": {"name": "t", "description": "d", "parameters": SCHEMA},
    }


def test_run_tool_reports_unknown_and_failing_tools(monkeypatch):
    def _boom(args):
        raise RuntimeError("down")

    monkeypatch.setitem(tools.TOOL_DISPATCH, "boom", _boom)
    assert run_tool("missing", {}) == "Unknown tool: missing"
    assert run_tool("boom", {}) == "Tool error (boom): down"

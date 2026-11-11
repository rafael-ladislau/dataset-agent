"""Shared agent helpers: think stripping, JSON extraction, schema validation."""

from __future__ import annotations

import pytest

from dataset_agent.adapters import tools
from dataset_agent.adapters.agent_common import (
    coerce_to_schema,
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


COERCE_SCHEMA = {
    "type": "object",
    "properties": {
        "used": {"type": "boolean"},
        "score": {"type": "integer"},
        "ratio": {"type": "number"},
        "names": {"type": "array", "items": {"type": "string"}},
        "nested": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"used": "false"}, {"used": False}),
        ({"used": "TRUE"}, {"used": True}),
        ({"score": "7"}, {"score": 7}),
        ({"score": " -3 "}, {"score": -3}),
        ({"ratio": "0.25"}, {"ratio": 0.25}),
        ({"ratio": "2"}, {"ratio": 2}),
        ({"names": "ACS"}, {"names": ["ACS"]}),
        ({"names": ["ACS"], "extra": 1}, {"names": ["ACS"]}),
        ({"nested": {"ok": "true", "junk": 1}}, {"nested": {"ok": True}}),
        # Not coercible: left as-is for validation to reject.
        ({"used": "maybe"}, {"used": "maybe"}),
        ({"score": "7.5"}, {"score": "7.5"}),
        ({"names": 5}, {"names": 5}),
    ],
)
def test_coerce_to_schema(raw, expected):
    assert coerce_to_schema(raw, COERCE_SCHEMA) == expected


def test_coerce_keeps_extra_keys_when_additional_properties_allowed():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert coerce_to_schema({"a": "1", "b": "x"}, schema) == {"a": 1, "b": "x"}


def test_coerce_drops_optional_null_but_keeps_required_null():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
        "required": ["b"],
        "additionalProperties": False,
    }
    assert coerce_to_schema({"a": None, "b": "x"}, schema) == {"b": "x"}
    assert coerce_to_schema({"a": 1, "b": None}, schema) == {"a": 1, "b": None}


@pytest.mark.parametrize("field,value", [
    ("papers", None), ("papers", "~6"), ("cause_term", None), ("exclude_terms", None)])
def test_refinement_schema_tolerates_null_informational_group_fields(field, value):
    from dataset_agent.adapters.agent_common import repair_near_miss, validate_against_schema
    schema = tools.EMIT_REFINEMENT_CANDIDATES["input_schema"]
    group = {"label": "UI = user interface", "cause_term": "UI", "papers": 3, "exclude_terms": ["x y"]}
    group[field] = value
    reply = {"false_positive_groups": [group], "exclude_terms": ["graphical interface"],
             "flag_terms": ["Census Bureau"], "new_aliases": []}
    assert validate_against_schema(reply, schema) is None
    fixed, err = repair_near_miss(reply, schema)
    assert err == "" and fixed["exclude_terms"] == ["graphical interface"]

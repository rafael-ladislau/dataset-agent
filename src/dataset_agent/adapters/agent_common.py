"""Helpers shared by the LLM agent adapters (Anthropic and OpenAI-compatible)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from jsonschema import SchemaError
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for

from dataset_agent.adapters import tools as _tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a dataset research assistant. Your job is to find accurate, factual "
    "information about academic and government datasets: their descriptions, official "
    "URLs, publisher organizations, alternative names, and access types. "
    "Use web_search to look up current information and make_request to validate URLs. "
    "Always prefer official sources (government agencies, academic repositories, "
    "publisher websites). Be concise and factual."
)

NO_OUTPUT_TEXT = "No output from agent."

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def prompt_head(prompt: str, limit: int) -> str:
    """One-line prompt preview for logs."""
    head = prompt.strip().replace("\n", " ")[:limit]
    if len(prompt) > limit:
        head += "…"
    return head


def run_tool(name: str, input_args: dict[str, Any]) -> str:
    """Execute an action tool from ``TOOL_DISPATCH`` and return its string result."""
    handler = _tools.TOOL_DISPATCH.get(name)
    if handler is None:
        msg = f"Unknown tool: {name}"
        logger.warning(msg)
        return msg
    try:
        return handler(input_args)
    except Exception as exc:
        logger.exception("Tool %s raised an error", name)
        return f"Tool error ({name}): {exc}"


def to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate the canonical ``{name, description, input_schema}`` tool to OpenAI shape."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [to_openai_tool(t) for t in tools]


def strip_think(text: str | None) -> str:
    """Remove ``<think>…</think>`` reasoning blocks from model text.

    Also handles a reply whose opening ``<think>`` tag was swallowed by the
    server (text up to the first ``</think>`` is dropped) and a reply that opens
    ``<think>`` but never closes it (everything from the tag on is dropped).
    """
    if not text:
        return ""
    out = _THINK_BLOCK_RE.sub("", text)
    close = _THINK_CLOSE_RE.search(out)
    if close and not _THINK_OPEN_RE.search(out[: close.start()]):
        out = out[close.end() :]
    open_tag = _THINK_OPEN_RE.search(out)
    if open_tag:
        out = out[: open_tag.start()]
    return out.strip()


def _balanced_objects(text: str):
    """Yield each balanced ``{...}`` substring in order, honoring JSON strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break
        start = text.find("{", start + 1)


def _first_object_in(text: str) -> dict[str, Any] | None:
    """First balanced ``{...}`` in *text* that parses as a JSON object."""
    for candidate in _balanced_objects(text):
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed
    return None


def _loads_object(candidate: str) -> dict[str, Any] | None:
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    """Parse a JSON object from model text (think blocks stripped).

    Tries, in order: the whole text, each fenced code block, then each
    balanced ``{...}`` substring until one parses as a JSON object.
    """
    cleaned = strip_think(text)
    if not cleaned:
        return None
    whole = _loads_object(cleaned)
    if whole is not None:
        return whole
    for block in _JSON_FENCE_RE.findall(cleaned):
        parsed = _loads_object(block.strip())
        if parsed is not None:
            return parsed
        inner = _first_object_in(block)
        if inner is not None:
            return inner
    return _first_object_in(cleaned)


def validate_against_schema(value: Any, schema: dict[str, Any] | None) -> str | None:
    """Validate *value* against a JSON Schema; return an error message or ``None``.

    A missing or invalid schema is not enforced (the value is accepted).
    """
    if not schema:
        return None
    try:
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
    except SchemaError as exc:
        logger.debug("Result schema is not a valid JSON Schema; skipping validation: %s", exc)
        return None
    error = best_match(validator_cls(schema).iter_errors(value))
    if error is None:
        return None
    path = "/".join(str(p) for p in error.absolute_path)
    return f"{error.message} (at '{path}')" if path else error.message


_INT_RE = re.compile(r"[+-]?\d+")
_NUMBER_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def _schema_types(schema: dict[str, Any]) -> list[str]:
    declared = schema.get("type")
    if isinstance(declared, list):
        return [t for t in declared if isinstance(t, str)]
    return [declared] if isinstance(declared, str) else []


def _coerce_scalar_string(value: str, types: list[str]) -> Any:
    text = value.strip()
    if "string" in types:
        return value
    if "boolean" in types and text.lower() in ("true", "false"):
        return text.lower() == "true"
    if "integer" in types and _INT_RE.fullmatch(text):
        return int(text)
    if "number" in types and _NUMBER_RE.fullmatch(text):
        return int(text) if _INT_RE.fullmatch(text) else float(text)
    return value


def coerce_to_schema(value: Any, schema: dict[str, Any] | None) -> Any:
    """Lenient, schema-guided repair of a near-miss structured result.

    Applies only safe conversions: ``"true"``/``"false"`` (any case) to booleans,
    numeric strings to integers/numbers, optional keys sent as null dropped, a lone string to a one-item array of
    strings, and dropping keys outside ``properties`` when
    ``additionalProperties`` is false.  Anything else is returned unchanged so
    that schema validation still rejects it.
    """
    if not isinstance(schema, dict):
        return value
    types = _schema_types(schema)
    if isinstance(value, dict) and (not types or "object" in types):
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        out: dict[str, Any] = {}
        for key, item in value.items():
            if item is None and key in properties and key not in required \
                    and "null" not in _schema_types(properties[key]):
                continue  # an optional key sent as null means "absent"
            if key in properties:
                out[key] = coerce_to_schema(item, properties[key])
            elif schema.get("additionalProperties") is not False:
                out[key] = item
        return out
    if isinstance(value, list) and "array" in types:
        items = schema.get("items")
        return [coerce_to_schema(v, items) for v in value] if isinstance(items, dict) else value
    if isinstance(value, str):
        if "array" in types and "string" not in types:
            items = schema.get("items")
            if isinstance(items, dict) and "string" in _schema_types(items):
                return [value]
            return value
        return _coerce_scalar_string(value, types)
    return value


def repair_near_miss(
    value: dict[str, Any] | None, schema: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, str]:
    """Coerce a near-miss and re-validate; return ``(object, "")`` or ``(None, error)``."""
    if not value:
        return None, "nothing parseable was obtained"
    coerced = coerce_to_schema(value, schema)
    error = validate_against_schema(coerced, schema)
    if error or not isinstance(coerced, dict):
        return None, error or "not a JSON object"
    return coerced, ""


def retry_prompt(prompt: str, tool_name: str, error: str) -> str:
    """Original prompt plus a note that names the previous structured-output error."""
    return (
        f"{prompt}\n\n"
        f"Your previous answer for '{tool_name}' was rejected: {error}. "
        "Answer again with a single JSON object that matches the required schema exactly."
    )


def json_only_prompt(prompt: str, result_tool: dict[str, Any]) -> str:
    """Prompt suffix asking for plain JSON, used when no structured API feature works."""
    schema = json.dumps(result_tool.get("input_schema") or {}, ensure_ascii=False)
    description = result_tool.get("description", "")
    return (
        f"{prompt}\n\n"
        f"Return your answer ({result_tool['name']}: {description}) as a single JSON object "
        f"that matches this JSON Schema, with no other text:\n{schema}"
    )

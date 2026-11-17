"""Live smoke tests against real OpenAI-compatible servers (skipped by default).

Enable with ``DATASET_AGENT_LIVE_LLM=1`` and list targets in
``DATASET_AGENT_LIVE_LLM_TARGETS`` as comma-separated ``<base_url>|<model>``
pairs, for example::

    DATASET_AGENT_LIVE_LLM=1 \\
    DATASET_AGENT_LIVE_LLM_TARGETS="http://127.0.0.1:1234/v1|qwen3.6-35b-a3b,\\
    http://openllm:3000/v1|gpt-oss-120b" \\
    pytest tests/test_llm_live.py -s

``DATASET_AGENT_LIVE_LLM_API_KEY`` is sent as the bearer key (default ``local``).
Each run prints the structured-output tier that worked for the target.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from dataset_agent.adapters import tools
from dataset_agent.adapters.agent_openai import OpenAICompatibleAgent
from dataset_agent.adapters.tools import EMIT_DATASET_NAMES

LIVE = os.environ.get("DATASET_AGENT_LIVE_LLM") == "1"


def _targets() -> list[tuple[str, str]]:
    raw = os.environ.get("DATASET_AGENT_LIVE_LLM_TARGETS", "")
    out = []
    for item in raw.split(","):
        base_url, sep, model = item.strip().partition("|")
        if sep and base_url and model:
            out.append((base_url.strip(), model.strip()))
    return out


requires_live = pytest.mark.skipif(
    not LIVE, reason="set DATASET_AGENT_LIVE_LLM=1 to run live LLM smoke"
)

TARGETS = _targets() if LIVE else []
if LIVE and not TARGETS:
    TARGETS = [pytest.param("", "", marks=pytest.mark.skip(reason="no live targets configured"))]

SECRET_TOOL = {
    "name": "get_dataset_code",
    "description": "Return the internal catalog code for a dataset name.",
    "input_schema": {
        "type": "object",
        "properties": {"dataset": {"type": "string", "description": "Dataset name."}},
        "required": ["dataset"],
    },
}


def _agent(base_url: str, model: str) -> OpenAICompatibleAgent:
    return OpenAICompatibleAgent(
        model_name=model,
        base_url=base_url,
        api_key=os.environ.get("DATASET_AGENT_LIVE_LLM_API_KEY", "local"),
        temperature=0.0,
        max_iterations=3,
        timeout_seconds=300,
    )


@requires_live
@pytest.mark.parametrize(("base_url", "model"), TARGETS)
def test_live_tool_call(base_url: str, model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _code(args: dict[str, Any]) -> str:
        calls.append(args)
        return "CODE-7731"

    monkeypatch.setitem(tools.TOOL_DISPATCH, "get_dataset_code", _code)
    answer = _agent(base_url, model).get_information(
        "Use the get_dataset_code tool to look up the catalog code for the "
        "'American Community Survey', then reply with that code only.",
        tools=[SECRET_TOOL],
    )
    print(f"\n[{model} @ {base_url}] tool calls={calls} answer={answer!r}")
    assert calls, f"model did not emit an OpenAI tool call; answer was {answer!r}"
    assert "CODE-7731" in answer


@requires_live
@pytest.mark.parametrize(("base_url", "model"), TARGETS)
def test_live_structured_call(base_url: str, model: str) -> None:
    agent = _agent(base_url, model)
    result = agent.get_structured(
        "List the dataset name aliases for the American Community Survey "
        "(official title and acronym).",
        EMIT_DATASET_NAMES,
    )
    tier = agent.last_structured_tier
    print(f"\n[{model} @ {base_url}] structured tier={tier} result={result}")
    assert agent.structured_failures == 0
    assert tier is not None
    assert result.get("dataset_names")


# --- gpt-oss structured-output smokes (separate gate) -----------------------

GPT_OSS_ENV = "DATASET_AGENT_LIVE_GPT_OSS_ENDPOINTS"


def _gpt_oss_targets() -> list[Any]:
    """Parse ``<host_root>|<model>`` pairs from ``DATASET_AGENT_LIVE_GPT_OSS_ENDPOINTS``.

    Example: ``http://127.0.0.1:1234|gpt-oss-120b,http://127.0.0.1:11434|gpt-oss:120b``.
    ``/v1`` is appended when missing.  Unset or empty yields one skipped param.
    """
    out: list[Any] = []
    for item in os.environ.get(GPT_OSS_ENV, "").split(","):
        host, sep, model = item.strip().partition("|")
        host, model = host.strip().rstrip("/"), model.strip()
        if sep and host and model:
            base = host if host.endswith("/v1") else f"{host}/v1"
            out.append(pytest.param(base, model, id=f"{model}@{host}"))
    if not out:
        out.append(
            pytest.param(
                "", "", marks=pytest.mark.skip(reason=f"set {GPT_OSS_ENV} to run gpt-oss smokes")
            )
        )
    return out


SMALL_TOOL = {
    "name": "emit_country",
    "description": "Report the capital city of a country.",
    "input_schema": {
        "type": "object",
        "properties": {
            "country": {"type": "string"},
            "capital": {"type": "string"},
        },
        "required": ["country", "capital"],
    },
}


@pytest.mark.parametrize(("base_url", "model"), _gpt_oss_targets())
def test_live_gpt_oss_get_structured(base_url: str, model: str) -> None:
    """gpt-oss reasoning must not exhaust the budget before the structured answer."""
    agent = OpenAICompatibleAgent(
        model_name=model,
        base_url=base_url,
        api_key=os.environ.get("DATASET_AGENT_LIVE_LLM_API_KEY", "local"),
        temperature=0.0,
        max_tokens=8192,
        max_iterations=3,
        timeout_seconds=600,
    )
    result = agent.get_structured("What is the capital of France?", SMALL_TOOL)
    print(f"\n[{model} @ {base_url}] tier={agent.last_structured_tier} result={result}")
    assert agent.structured_failures == 0
    assert str(result.get("capital", "")).strip().lower() == "paris"

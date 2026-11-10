"""OpenAI-compatible chat-completions agent (LM Studio, OpenLLM, Ollama, vLLM, OpenRouter)."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from openai import APIStatusError, OpenAI

from dataset_agent.adapters.agent_common import (
    NO_OUTPUT_TEXT,
    SYSTEM_PROMPT,
    extract_json_object,
    json_only_prompt,
    prompt_head,
    retry_prompt,
    run_tool,
    strip_think,
    to_openai_tool,
    to_openai_tools,
    validate_against_schema,
)
from dataset_agent.adapters.tools import TOOL_DEFINITIONS
from dataset_agent.domain.ports import AgentPort

logger = logging.getLogger(__name__)

#: Structured-output strategies, most to least reliable.
TIER_JSON_SCHEMA = "json_schema"
TIER_TOOL_CALL = "tool_call"
TIER_TEXT_JSON = "text_json"
STRUCTURED_TIERS: tuple[str, ...] = (TIER_JSON_SCHEMA, TIER_TOOL_CALL, TIER_TEXT_JSON)

#: HTTP statuses treated as "the server does not support this request shape".
_UNSUPPORTED_STATUSES = frozenset({400, 404, 405, 415, 422, 501})

#: (base_url, model) → first structured tier known to work, shared in-process.
_TIER_CACHE: dict[tuple[str, str], str] = {}
_TIER_CACHE_LOCK = threading.Lock()


def reset_tier_cache() -> None:
    """Forget which structured-output tier works for each (base_url, model)."""
    with _TIER_CACHE_LOCK:
        _TIER_CACHE.clear()


class _TierUnsupportedError(Exception):
    """The server rejected the request shape used by a tier."""


class OpenAICompatibleAgent(AgentPort):
    """AgentPort backed by the official ``openai`` SDK (``/v1/chat/completions``).

    Tool definitions keep the canonical Anthropic shape
    (``{name, description, input_schema}``) and are translated per request.
    ``base_url`` is the OpenAI API root, usually ending in ``/v1``.
    """

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        max_iterations: int = 5,
        timeout_seconds: int = 120,
        http_client: Any | None = None,
        max_retries: int | None = None,
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or "local"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client
        self._max_retries = max_retries
        self._client: OpenAI | None = None
        #: Number of ``get_structured`` calls that ended in the ``{}`` fallback.
        self.structured_failures = 0

    def _get_client(self) -> OpenAI:
        if self._client is None:
            logger.info(
                "Initializing OpenAI-compatible client: base_url=%s model=%s max_iter=%s "
                "timeout_s=%s",
                self._base_url,
                self._model_name,
                self._max_iterations,
                self._timeout_seconds,
            )
            extra: dict[str, Any] = {}
            if self._http_client is not None:
                extra["http_client"] = self._http_client
            if self._max_retries is not None:
                extra["max_retries"] = self._max_retries
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=float(self._timeout_seconds),
                **extra,
            )
        return self._client

    def _create(self, messages: list[dict[str, Any]], **extra: Any) -> Any:
        return self._get_client().chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **extra,
        )

    @staticmethod
    def _base_messages(prompt: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    # ------------------------------------------------------------------
    # Tool-calling agentic loop
    # ------------------------------------------------------------------

    def get_information(self, prompt: str, *, tools: list | None = None) -> str:
        """Run the chat-completions tool loop.

        ``tools=None``  → use the default TOOL_DEFINITIONS (web_search + make_request).
        ``tools=[]``    → disable all tools (pure reasoning call).
        ``tools=[...]`` → use only the supplied tool definitions.

        Tool calls written into the message text (instead of ``tool_calls``)
        are not executed; that text is returned as the answer.
        """
        active_tools: list[Any] = TOOL_DEFINITIONS if tools is None else tools
        logger.info(
            "LLM agent call (prompt %s chars, tools=%s): %s",
            len(prompt),
            [t["name"] for t in active_tools] if active_tools else "none",
            prompt_head(prompt, 160),
        )

        messages = self._base_messages(prompt)
        openai_tools = to_openai_tools(active_tools) if active_tools else None
        start = time.time()

        try:
            for iteration in range(1, self._max_iterations + 1):
                extra: dict[str, Any] = {"tools": openai_tools} if openai_tools else {}
                response = self._create(messages, **extra)
                choice = response.choices[0]
                message = choice.message
                tool_calls = [
                    tc for tc in (message.tool_calls or []) if getattr(tc, "type", "") == "function"
                ]

                if not tool_calls:
                    logger.info(
                        "LLM agent finished in %.2fs (finish_reason=%s)",
                        time.time() - start,
                        choice.finish_reason,
                    )
                    return self._extract_text(message)

                logger.info(
                    "Iteration %s/%s: %s tool call(s)",
                    iteration,
                    self._max_iterations,
                    len(tool_calls),
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": strip_think(message.content) or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments or "{}",
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    args = self._parse_arguments(tc.function.arguments)
                    logger.info(
                        "Executing tool %s (id=%s) args=%s",
                        tc.function.name,
                        tc.id,
                        json.dumps(args, ensure_ascii=False)[:200],
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": run_tool(tc.function.name, args),
                        }
                    )

            logger.warning(
                "Max iterations (%s) reached; requesting final summary",
                self._max_iterations,
            )
            final = self._create(messages)
            logger.info("LLM agent finished in %.2fs (hit iteration cap)", time.time() - start)
            return self._extract_text(final.choices[0].message)

        except Exception as exc:
            logger.exception("Agent failed")
            return f"Error in agent execution: {exc}"

    @staticmethod
    def _parse_arguments(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return extract_json_object(raw) or {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _extract_text(message: Any) -> str:
        """Visible answer text; reasoning fields and ``<think>`` blocks are dropped."""
        return strip_think(getattr(message, "content", None)) or NO_OUTPUT_TEXT

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        """Return a dict that validates against ``result_tool["input_schema"]``.

        Tiers, starting at the one cached for this (base_url, model):
        ``json_schema`` response format → forced tool call → JSON parsed from
        text.  Each tier retries once on a parse/validation error; a tier the
        server rejects as unsupported is skipped without retry.  Returns ``{}``
        (WARNING log, ``structured_failures`` incremented) when every tier fails.
        """
        tool_name = result_tool["name"]
        schema = result_tool.get("input_schema")
        cache_key = (self._base_url, self._model_name)
        with _TIER_CACHE_LOCK:
            cached = _TIER_CACHE.get(cache_key)
        first = STRUCTURED_TIERS.index(cached) if cached in STRUCTURED_TIERS else 0

        logger.info(
            "LLM structured call (tool=%s, prompt %s chars, start_tier=%s): %s",
            tool_name,
            len(prompt),
            STRUCTURED_TIERS[first],
            prompt_head(prompt, 120),
        )
        start = time.time()
        errors: list[str] = []
        for tier in STRUCTURED_TIERS[first:]:
            attempt_prompt = prompt
            for attempt in (1, 2):
                try:
                    result, error = self._run_tier(tier, attempt_prompt, result_tool)
                except _TierUnsupportedError as exc:
                    logger.info("Structured tier %s unsupported by server: %s", tier, exc)
                    errors.append(f"{tier}: unsupported ({exc})")
                    break
                except Exception:
                    logger.exception("get_structured failed (tool=%s tier=%s)", tool_name, tier)
                    self.structured_failures += 1
                    return {}
                if result is not None and not error:
                    error = validate_against_schema(result, schema) or ""
                    if error:
                        error = f"schema validation failed: {error}"
                if result is not None and not error:
                    with _TIER_CACHE_LOCK:
                        _TIER_CACHE[cache_key] = tier
                    logger.info(
                        "LLM structured call finished in %.2fs (tool=%s tier=%s attempt=%s "
                        "keys=%s)",
                        time.time() - start,
                        tool_name,
                        tier,
                        attempt,
                        list(result.keys()),
                    )
                    return result
                logger.info(
                    "Structured tier %s attempt %s rejected (tool=%s): %s",
                    tier,
                    attempt,
                    tool_name,
                    error,
                )
                errors.append(f"{tier}#{attempt}: {error}")
                attempt_prompt = retry_prompt(prompt, tool_name, error)

        self.structured_failures += 1
        logger.warning(
            "LLM structured call failed in every tier (tool=%s model=%s): %s; returning {}",
            tool_name,
            self._model_name,
            "; ".join(errors),
        )
        return {}

    def _run_tier(
        self, tier: str, prompt: str, result_tool: dict
    ) -> tuple[dict[str, Any] | None, str]:
        """Run one structured request; return ``(object, "")`` or ``(None, error)``."""
        tool_name = result_tool["name"]
        try:
            if tier == TIER_JSON_SCHEMA:
                response = self._create(
                    self._base_messages(prompt),
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": tool_name,
                            "schema": result_tool.get("input_schema") or {"type": "object"},
                            "strict": False,
                        },
                    },
                )
                return self._object_from_text(response.choices[0].message)
            if tier == TIER_TOOL_CALL:
                response = self._create(
                    self._base_messages(prompt),
                    tools=[to_openai_tool(result_tool)],
                    tool_choice={"type": "function", "function": {"name": tool_name}},
                )
                return self._object_from_tool_call(response.choices[0].message, tool_name)
            response = self._create(self._base_messages(json_only_prompt(prompt, result_tool)))
            return self._object_from_text(response.choices[0].message)
        except APIStatusError as exc:
            if exc.status_code in _UNSUPPORTED_STATUSES:
                raise _TierUnsupportedError(f"HTTP {exc.status_code}") from exc
            raise

    @staticmethod
    def _object_from_text(message: Any) -> tuple[dict[str, Any] | None, str]:
        parsed = extract_json_object(getattr(message, "content", None))
        if parsed is None:
            return None, "the reply did not contain a JSON object"
        return parsed, ""

    @staticmethod
    def _object_from_tool_call(message: Any, tool_name: str) -> tuple[dict[str, Any] | None, str]:
        for tc in message.tool_calls or []:
            if getattr(tc, "type", "") == "function" and tc.function.name == tool_name:
                try:
                    value = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, ValueError) as exc:
                    return None, f"the tool arguments were not valid JSON ({exc.msg})"
                if not isinstance(value, dict):
                    return None, "the tool arguments were not a JSON object"
                return value, ""
        return None, f"the tool '{tool_name}' was not called"

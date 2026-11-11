"""OpenAI-compatible chat-completions agent (LM Studio, OpenLLM, Ollama, vLLM, OpenRouter)."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

from openai import APIStatusError, OpenAI

from dataset_agent.adapters.agent_common import (
    NO_OUTPUT_TEXT,
    SYSTEM_PROMPT,
    contains_harmony_markup,
    extract_json_object,
    has_harmony_markup,
    parse_harmony_tool_calls,
    json_only_prompt,
    prompt_head,
    repair_near_miss,
    retry_prompt,
    run_tool,
    strip_harmony,
    strip_think,
    to_openai_tool,
    to_openai_tools,
    validate_against_schema,
)
from dataset_agent.adapters.tools import TOOL_DEFINITIONS
from dataset_agent.domain.ports import AgentPort

logger = logging.getLogger(__name__)

_HARMONY_RETRY_MESSAGE = (
    "Your last reply contained raw control markup (angle-bracket pipe tokens) instead of an "
    "answer or a proper tool call. Reply again with either a real tool call or your plain-text "
    "answer, without any such tokens."
)

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


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _whole_json_object(text: str) -> dict[str, Any] | None:
    """*text* as a JSON object only when the whole string (or one fenced block) is one."""
    t = text.strip()
    m = _FENCE_RE.match(t)
    if m:
        t = m.group(1).strip()
    try:
        value = json.loads(t)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


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
        max_tokens: int = 8192,
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
        #: Number of ``get_structured`` calls that got no schema-valid reply (the
        #: result was a coerced near-miss or ``{}``).
        self.structured_failures = 0
        #: Tier that produced the last ``get_structured`` result (``None`` for ``{}``).
        self.last_structured_tier: str | None = None

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

        Plain JSON tool calls written into the message text are not executed. Harmony
        tool calls leaked into the text (gpt-oss) are recovered and executed; Harmony
        markup never appears in the returned answer.
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
        leak_retried = False

        try:
            for iteration in range(1, self._max_iterations + 1):
                extra: dict[str, Any] = {"tools": openai_tools} if openai_tools else {}
                response = self._create(messages, **extra)
                choice = response.choices[0]
                message = choice.message
                tool_calls = [
                    tc for tc in (message.tool_calls or []) if getattr(tc, "type", "") == "function"
                ]

                # (id, name, raw argument text, parsed arguments)
                calls: list[tuple[str, str, str, dict[str, Any]]] = [
                    (
                        tc.id,
                        tc.function.name,
                        self._arguments_text(tc.function.arguments),
                        self._parse_arguments(tc.function.arguments),
                    )
                    for tc in tool_calls
                ]

                if not calls and has_harmony_markup(message.content):
                    leaked = self._leaked_calls(message.content, active_tools, iteration)
                    if leaked:
                        logger.warning(
                            "Recovered %s tool call(s) leaked as Harmony text in message content",
                            len(leaked),
                        )
                        calls = leaked
                    elif strip_harmony(strip_think(message.content)):
                        pass  # a plain final-channel answer wrapped in markup; stripped below
                    elif not leak_retried:
                        leak_retried = True
                        logger.warning(
                            "Unparseable Harmony markup in assistant content; retrying the turn"
                        )
                        messages.append({"role": "user", "content": _HARMONY_RETRY_MESSAGE})
                        continue
                    else:
                        logger.warning("Harmony markup persisted after retry; treating as no answer")
                        return strip_harmony(strip_think(message.content)) or NO_OUTPUT_TEXT

                if not calls:
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
                    len(calls),
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            None
                            if has_harmony_markup(message.content)
                            else strip_think(message.content) or None
                        ),
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": raw},
                            }
                            for call_id, name, raw, _ in calls
                        ],
                    }
                )
                for call_id, name, _, args in calls:
                    logger.info(
                        "Executing tool %s (id=%s) args=%s",
                        name,
                        call_id,
                        json.dumps(args, ensure_ascii=False)[:200],
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": run_tool(name, args),
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
    def _leaked_calls(
        content: Any, active_tools: list[Any], iteration: int
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Harmony tool calls in *content* for offered tools; ``[]`` if any is unusable."""
        parsed = parse_harmony_tool_calls(content)
        offered = {t["name"] for t in active_tools}
        if not parsed or any(name not in offered for name, _ in parsed):
            return []
        return [
            (f"harmony_{iteration}_{i}", name, json.dumps(args, ensure_ascii=False), args)
            for i, (name, args) in enumerate(parsed, start=1)
        ]

    @staticmethod
    def _parse_arguments(raw: Any) -> dict[str, Any]:
        """Tool-call arguments: a JSON string (the spec) or a dict (some local servers)."""
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            return extract_json_object(raw) if isinstance(raw, str) else {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _arguments_text(raw: Any) -> str:
        if isinstance(raw, dict):
            return json.dumps(raw, ensure_ascii=False)
        return raw or "{}"

    @staticmethod
    def _extract_text(message: Any) -> str:
        """Visible answer text; reasoning fields and ``<think>`` blocks are dropped."""
        text = strip_think(getattr(message, "content", None))
        if has_harmony_markup(text):
            text = strip_harmony(text)
        return text or NO_OUTPUT_TEXT

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        """Return a dict that validates against ``result_tool["input_schema"]``.

        Tiers, starting at the one cached for this (base_url, model):
        ``json_schema`` response format → forced tool call → JSON parsed from
        text.  Each tier retries once on a parse/validation error.  A tier the
        server rejects with a 4xx/501 is skipped without retry; the cache is
        lowered past it only once a lower tier gets a 2xx reply, so a
        request-level error that every tier hits leaves the cache unchanged.

        When no reply validates, ``structured_failures`` is incremented, a
        WARNING is logged, and the last parsed non-empty object is repaired with
        :func:`coerce_to_schema`; it is returned if it then validates, otherwise
        ``{}``.  ``last_structured_tier`` names the tier that produced the
        result (``None`` for ``{}``).
        """
        tool_name = result_tool["name"]
        schema = result_tool.get("input_schema")
        cache_key = (self._base_url, self._model_name)
        self.last_structured_tier = None
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
        rejected = False
        answered = False  # a higher tier already answered 2xx (even with bad content)
        near_miss: dict[str, Any] | None = None
        near_miss_tier: str | None = None
        for tier in STRUCTURED_TIERS[first:]:
            attempt_prompt = prompt
            for attempt in (1, 2):
                try:
                    result, error = self._run_tier(tier, attempt_prompt, result_tool)
                except _TierUnsupportedError as exc:
                    logger.info("Structured tier %s rejected by server: %s", tier, exc)
                    errors.append(f"{tier}: rejected ({exc})")
                    rejected = True
                    break
                except Exception:
                    logger.exception("get_structured failed (tool=%s tier=%s)", tool_name, tier)
                    return self._structured_fallback(tool_name, schema, near_miss, near_miss_tier)
                if result is not None and contains_harmony_markup(result):
                    result, error = None, "the reply contained Harmony control markup"
                if rejected and not answered:
                    # A lower tier answered 2xx and no higher tier did: the higher ones
                    # are really unsupported. If a higher tier answered 2xx with bad
                    # content, it is supported, so the cache stays put.
                    self._demote_cached_tier(cache_key, tier)
                rejected = False
                answered = True
                if result is not None and not error:
                    error = validate_against_schema(result, schema) or ""
                    if error:
                        error = f"schema validation failed: {error}"
                        if result:
                            near_miss, near_miss_tier = result, tier
                if result is not None and not error:
                    logger.info(
                        "LLM structured call finished in %.2fs (tool=%s tier=%s attempt=%s "
                        "keys=%s)",
                        time.time() - start,
                        tool_name,
                        tier,
                        attempt,
                        list(result.keys()),
                    )
                    self.last_structured_tier = tier
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

        logger.info(
            "LLM structured call exhausted every tier (tool=%s model=%s): %s",
            tool_name,
            self._model_name,
            "; ".join(errors),
        )
        return self._structured_fallback(tool_name, schema, near_miss, near_miss_tier)

    def _structured_fallback(
        self,
        tool_name: str,
        schema: dict | None,
        near_miss: dict[str, Any] | None,
        near_miss_tier: str | None,
    ) -> dict[str, Any]:
        """Count the failure; return the coerced near-miss if it validates, else ``{}``."""
        self.structured_failures += 1
        repaired, error = repair_near_miss(near_miss, schema)
        if repaired is not None:
            logger.warning(
                "LLM structured call (tool=%s model=%s) returned no valid result; using the "
                "last parsed object coerced to the schema",
                tool_name,
                self._model_name,
            )
            self.last_structured_tier = near_miss_tier
            return repaired
        logger.warning(
            "LLM structured call (tool=%s model=%s) returned no usable result (%s); returning {}",
            tool_name,
            self._model_name,
            error,
        )
        return {}

    @staticmethod
    def _demote_cached_tier(cache_key: tuple[str, str], working_tier: str) -> None:
        """Start later calls at *working_tier*, below tiers the server rejected."""
        index = STRUCTURED_TIERS.index(working_tier)
        with _TIER_CACHE_LOCK:
            current = _TIER_CACHE.get(cache_key)
            current_index = STRUCTURED_TIERS.index(current) if current in STRUCTURED_TIERS else 0
            if index > current_index:
                _TIER_CACHE[cache_key] = working_tier

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
                # LM Studio's reasoning parser can leave content empty and put the
                # schema-constrained JSON in reasoning_content.
                return self._object_from_text(response.choices[0].message, allow_reasoning=True)
            if tier == TIER_TOOL_CALL:
                # Exactly one tool is offered, so "required" forces it; the object form is
                # rejected by LM Studio ("Invalid tool_choice type"). Endpoints that reject
                # "required" get the object form once.
                tools = [to_openai_tool(result_tool)]
                try:
                    response = self._create(
                        self._base_messages(prompt), tools=tools, tool_choice="required"
                    )
                except APIStatusError as exc:
                    if exc.status_code not in _UNSUPPORTED_STATUSES:
                        raise
                    response = self._create(
                        self._base_messages(prompt),
                        tools=tools,
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
    def _object_from_text(
        message: Any, *, allow_reasoning: bool = False
    ) -> tuple[dict[str, Any] | None, str]:
        content = getattr(message, "content", None)
        if has_harmony_markup(content):
            content = strip_harmony(strip_think(content))
        parsed = extract_json_object(content)
        if parsed is None and allow_reasoning:
            for attr in ("reasoning_content", "reasoning"):
                value = getattr(message, attr, None)
                obj = _whole_json_object(value) if isinstance(value, str) else None
                if obj is not None:
                    logger.info("structured reply taken from %s (content was empty)", attr)
                    return obj, ""
        if parsed is None:
            return None, "the reply did not contain a JSON object"
        return parsed, ""

    @staticmethod
    def _object_from_tool_call(message: Any, tool_name: str) -> tuple[dict[str, Any] | None, str]:
        for tc in message.tool_calls or []:
            if getattr(tc, "type", "") == "function" and tc.function.name == tool_name:
                raw = tc.function.arguments
                if isinstance(raw, dict):
                    return raw, ""
                try:
                    value = json.loads(raw or "{}")
                except json.JSONDecodeError as exc:
                    return None, f"the tool arguments were not valid JSON ({exc.msg})"
                except (TypeError, ValueError) as exc:
                    return None, f"the tool arguments were not valid JSON ({exc})"
                if not isinstance(value, dict):
                    return None, "the tool arguments were not a JSON object"
                return value, ""
        return None, f"the tool '{tool_name}' was not called"

"""Anthropic SDK agent (LM Studio or Ollama Anthropic-compatible endpoints)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from anthropic import Anthropic

from dataset_agent.adapters.agent_common import (
    FINAL_SUMMARY_REQUEST,
    ToolRunCache,
    SYSTEM_PROMPT,
    prompt_head,
    repair_near_miss,
    retry_prompt,
    run_tool,
    strip_think,
    validate_against_schema,
)
from dataset_agent.adapters.tools import TOOL_DEFINITIONS
from dataset_agent.domain.ports import AgentPort

logger = logging.getLogger(__name__)

# Stop reasons that indicate a completed (non-tool) turn across local providers
_TERMINAL_STOP_REASONS = frozenset({"end_turn", "stop"})


class AnthropicAgent(AgentPort):
    """AgentPort backed by the Anthropic Python SDK with tool-use loop.

    Works with any server that exposes Anthropic-compatible ``/v1/messages``
    (LM Studio, Ollama, etc.).  A dummy API key is used by default since
    local servers do not require authentication.
    """

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        temperature: float = 0.6,
        api_key: str,
        max_tokens: int = 8192,
        max_iterations: int = 5,
        timeout_seconds: int = 120,
        http_client: Any | None = None,
        max_retries: int | None = None,
        thinking_budget: int = 0,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client
        self._max_retries = max_retries
        #: > 0 asks for Anthropic ``thinking`` on structured calls (forced tools are
        #: then replaced by ``auto``, which the API requires with thinking).
        self._thinking_budget = max(0, int(thinking_budget))
        self._top_p = top_p
        self._top_k = top_k
        self._client: Anthropic | None = None
        #: Number of ``get_structured`` calls that got no schema-valid tool call
        #: (the result was a coerced near-miss or ``{}``).
        self.structured_failures = 0

    def _get_client(self) -> Anthropic:
        if self._client is None:
            logger.info(
                "Initializing Anthropic client: base_url=%s model=%s max_iter=%s timeout_s=%s",
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
            self._client = Anthropic(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=float(self._timeout_seconds),
                **extra,
            )
        return self._client

    def _sampling_body(self) -> dict[str, Any]:
        """Sampling params sent as raw body fields.

        ``temperature`` is not a typed ``messages.create`` argument in every
        ``anthropic`` SDK release, so it goes through ``extra_body``; the
        request payload is the same.
        """
        body: dict[str, Any] = {"temperature": self._temperature}
        if self._top_p is not None:
            body["top_p"] = self._top_p
        if self._top_k is not None:
            body["top_k"] = self._top_k
        return body

    # ------------------------------------------------------------------
    # Tool-calling agentic loop
    # ------------------------------------------------------------------

    def _run_tool(self, name: str, input_args: dict[str, Any]) -> str:
        """Execute a tool by name and return the string result."""
        return run_tool(name, input_args)

    def get_information(self, prompt: str, *, tools: list | None = None) -> str:
        """Run the agentic loop.

        ``tools=None``  → use the default TOOL_DEFINITIONS (web_search + make_request).
        ``tools=[]``    → disable all tools (pure reasoning call).
        ``tools=[...]`` → use only the supplied tool definitions.
        """
        client = self._get_client()
        active_tools: list[Any] = TOOL_DEFINITIONS if tools is None else tools

        head = prompt_head(prompt, 160)
        logger.info(
            "LLM agent call (prompt %s chars, tools=%s): %s",
            len(prompt),
            [t["name"] for t in active_tools] if active_tools else "none",
            head,
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        start = time.time()
        cache = ToolRunCache(self._run_tool)

        try:
            for iteration in range(1, self._max_iterations + 1):
                kwargs: dict[str, Any] = {
                    "model": self._model_name,
                    "max_tokens": self._max_tokens,
                    "extra_body": self._sampling_body(),
                    "system": SYSTEM_PROMPT,
                    "messages": messages,
                }
                if active_tools:
                    kwargs["tools"] = active_tools

                response = client.messages.create(**kwargs)

                # Terminal stop reasons differ between providers (end_turn vs stop)
                if response.stop_reason in _TERMINAL_STOP_REASONS:
                    elapsed = time.time() - start
                    logger.info(
                        "LLM agent finished in %.2fs (stop_reason=%s)",
                        elapsed,
                        response.stop_reason,
                    )
                    return self._extract_text(response)

                # Collect tool-use blocks
                tool_use_blocks = [block for block in response.content if block.type == "tool_use"]

                if not tool_use_blocks:
                    # No tool calls and not a recognised terminal reason — return text
                    elapsed = time.time() - start
                    logger.info(
                        "LLM agent finished in %.2fs (no tool_use, stop_reason=%s)",
                        elapsed,
                        response.stop_reason,
                    )
                    return self._extract_text(response)

                logger.info(
                    "Iteration %s/%s: %s tool call(s)",
                    iteration,
                    self._max_iterations,
                    len(tool_use_blocks),
                )

                # Append the assistant's response (with tool_use blocks) to messages
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool and build the tool_result message
                tool_results: list[dict[str, Any]] = []
                all_repeats = True
                for block in tool_use_blocks:
                    tool_input = block.input if isinstance(block.input, dict) else {}
                    logger.info(
                        "Executing tool %s (id=%s) args=%s",
                        block.name,
                        block.id,
                        json.dumps(tool_input, ensure_ascii=False)[:200],
                    )
                    result_text, was_repeat = cache.run(block.name, tool_input)
                    all_repeats = all_repeats and was_repeat
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        }
                    )

                if all_repeats:
                    logger.warning("Only repeated tool calls in iteration %s; finalizing", iteration)
                    messages.append({"role": "user", "content": tool_results})
                    break
                messages.append({"role": "user", "content": tool_results})

            # Exhausted iterations — one final call without tools to get a summary
            logger.warning(
                "Max iterations (%s) reached; requesting final summary",
                self._max_iterations,
            )
            if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
                if not any(
                    b.get("text") == FINAL_SUMMARY_REQUEST for b in messages[-1]["content"]
                ):
                    messages[-1]["content"].append(
                        {"type": "text", "text": FINAL_SUMMARY_REQUEST}
                    )
            final = client.messages.create(
                model=self._model_name,
                max_tokens=self._max_tokens,
                extra_body=self._sampling_body(),
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            elapsed = time.time() - start
            logger.info("LLM agent finished in %.2fs (hit iteration cap)", elapsed)
            return self._extract_text(final)

        except Exception as exc:
            logger.exception("Agent failed")
            return f"Error in agent execution: {exc}"

    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        """Force the model to return structured data via a single tool_use call.

        The model is given only *result_tool* and is instructed via
        ``tool_choice`` to call it exactly once.  The tool's ``input`` dict
        (which matches the tool's ``input_schema``) is returned directly,
        so callers get a typed dict without any regex parsing.

        The input is validated against ``input_schema``; a missing or invalid
        tool call is retried once with a message naming the error.  If both
        attempts fail, ``structured_failures`` is incremented, a WARNING is
        logged, and the last non-empty tool input is repaired with
        :func:`coerce_to_schema`; it is returned if it then validates, otherwise
        ``{}`` so callers can apply their own defaults.
        """
        client = self._get_client()
        tool_name = result_tool["name"]
        schema = result_tool.get("input_schema")

        logger.info(
            "LLM structured call (tool=%s, prompt %s chars): %s",
            tool_name,
            len(prompt),
            prompt_head(prompt, 120),
        )

        start = time.time()
        attempt_prompt = prompt
        error = ""
        near_miss: dict | None = None
        for attempt in (1, 2):
            think: dict[str, Any] = {}
            tool_choice: dict[str, Any] = {"type": "tool", "name": tool_name}
            max_tokens = self._max_tokens
            if self._thinking_budget:
                think["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}
                tool_choice = {"type": "auto"}
                max_tokens += self._thinking_budget
            try:
                response = client.messages.create(
                    model=self._model_name,
                    max_tokens=max_tokens,
                    extra_body=self._sampling_body(),
                    system=SYSTEM_PROMPT,
                    tools=[result_tool],
                    tool_choice=tool_choice,
                    messages=[{"role": "user", "content": attempt_prompt}],
                    **think,
                )
            except Exception:
                logger.exception("get_structured failed (tool=%s)", tool_name)
                return self._structured_fallback(tool_name, schema, near_miss)
            result, error = self._structured_from_response(response, tool_name, schema)
            if result is not None and not error:
                logger.info(
                    "LLM structured call finished in %.2fs (tool=%s keys=%s attempt=%s)",
                    time.time() - start,
                    tool_name,
                    list(result.keys()),
                    attempt,
                )
                return result
            if result:
                near_miss = result
            logger.info(
                "LLM structured call attempt %s rejected (tool=%s): %s",
                attempt,
                tool_name,
                error,
            )
            attempt_prompt = retry_prompt(prompt, tool_name, error)

        logger.info("LLM structured call failed after retry (tool=%s): %s", tool_name, error)
        return self._structured_fallback(tool_name, schema, near_miss)

    def _structured_fallback(
        self, tool_name: str, schema: dict | None, near_miss: dict | None
    ) -> dict:
        """Count the failure; return the coerced near-miss if it validates, else ``{}``."""
        self.structured_failures += 1
        repaired, error = repair_near_miss(near_miss, schema)
        if repaired is not None:
            logger.warning(
                "LLM structured call (tool=%s) returned no valid result; using the last tool "
                "input coerced to the schema",
                tool_name,
            )
            return repaired
        logger.warning(
            "LLM structured call (tool=%s) returned no usable result (%s); returning {}",
            tool_name,
            error,
        )
        return {}

    @staticmethod
    def _structured_from_response(
        response: Any, tool_name: str, schema: dict | None
    ) -> tuple[dict | None, str]:
        """Return ``(input, "")`` for a valid call, ``(input, error)`` when the input
        fails validation, or ``(None, error)`` when there is no usable input."""
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                if not isinstance(block.input, dict):
                    return None, "the tool input was not a JSON object"
                error = validate_against_schema(block.input, schema)
                if error:
                    return block.input, f"schema validation failed: {error}"
                return block.input, ""
        return None, (f"the tool '{tool_name}' was not called (stop_reason={response.stop_reason})")

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull all text blocks from an Anthropic response, minus ``<think>`` blocks."""
        parts = [block.text for block in response.content if hasattr(block, "text") and block.text]
        return strip_think("\n".join(parts)) 

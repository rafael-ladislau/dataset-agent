"""Anthropic SDK agent (LM Studio or Ollama Anthropic-compatible endpoints)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from anthropic import Anthropic

from dataset_agent.adapters.tools import TOOL_DEFINITIONS, TOOL_DISPATCH
from dataset_agent.domain.ports import AgentPort

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a dataset research assistant. Your job is to find accurate, factual "
    "information about academic and government datasets: their descriptions, official "
    "URLs, publisher organizations, alternative names, and access types. "
    "Use web_search to look up current information and make_request to validate URLs. "
    "Always prefer official sources (government agencies, academic repositories, "
    "publisher websites). Be concise and factual."
)

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
        max_tokens: int = 4096,
        max_iterations: int = 5,
        timeout_seconds: int = 120,
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations
        self._timeout_seconds = timeout_seconds
        self._client: Anthropic | None = None

    def _get_client(self) -> Anthropic:
        if self._client is None:
            logger.info(
                "Initializing Anthropic client: base_url=%s model=%s max_iter=%s timeout_s=%s",
                self._base_url,
                self._model_name,
                self._max_iterations,
                self._timeout_seconds,
            )
            self._client = Anthropic(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=float(self._timeout_seconds),
            )
        return self._client

    # ------------------------------------------------------------------
    # Tool-calling agentic loop
    # ------------------------------------------------------------------

    def _run_tool(self, name: str, input_args: dict[str, Any]) -> str:
        """Execute a tool by name and return the string result."""
        handler = TOOL_DISPATCH.get(name)
        if handler is None:
            msg = f"Unknown tool: {name}"
            logger.warning(msg)
            return msg
        try:
            return handler(input_args)
        except Exception as exc:
            logger.exception("Tool %s raised an error", name)
            return f"Tool error ({name}): {exc}"

    def get_information(self, prompt: str, *, tools: list | None = None) -> str:
        """Run the agentic loop.

        ``tools=None``  → use the default TOOL_DEFINITIONS (web_search + make_request).
        ``tools=[]``    → disable all tools (pure reasoning call).
        ``tools=[...]`` → use only the supplied tool definitions.
        """
        client = self._get_client()
        active_tools: list[Any] = TOOL_DEFINITIONS if tools is None else tools

        head = prompt.strip().replace("\n", " ")[:160]
        if len(prompt) > 160:
            head += "…"
        logger.info(
            "LLM agent call (prompt %s chars, tools=%s): %s",
            len(prompt),
            [t["name"] for t in active_tools] if active_tools else "none",
            head,
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        start = time.time()

        try:
            for iteration in range(1, self._max_iterations + 1):
                kwargs: dict[str, Any] = dict(
                    model=self._model_name,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
                if active_tools:
                    kwargs["tools"] = active_tools

                response = client.messages.create(**kwargs)

                # Terminal stop reasons differ between providers (end_turn vs stop)
                if response.stop_reason in _TERMINAL_STOP_REASONS:
                    elapsed = time.time() - start
                    logger.info("LLM agent finished in %.2fs (stop_reason=%s)", elapsed, response.stop_reason)
                    return self._extract_text(response)

                # Collect tool-use blocks
                tool_use_blocks = [
                    block for block in response.content if block.type == "tool_use"
                ]

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
                for block in tool_use_blocks:
                    tool_input = block.input if isinstance(block.input, dict) else {}
                    logger.info(
                        "Executing tool %s (id=%s) args=%s",
                        block.name,
                        block.id,
                        json.dumps(tool_input, ensure_ascii=False)[:200],
                    )
                    result_text = self._run_tool(block.name, tool_input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        }
                    )

                messages.append({"role": "user", "content": tool_results})

            # Exhausted iterations — one final call without tools to get a summary
            logger.warning(
                "Max iterations (%s) reached; requesting final summary",
                self._max_iterations,
            )
            final = client.messages.create(
                model=self._model_name,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
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

        Falls back to ``{}`` on failure so callers can apply their own defaults.
        """
        client = self._get_client()
        tool_name = result_tool["name"]

        head = prompt.strip().replace("\n", " ")[:120]
        if len(prompt) > 120:
            head += "…"
        logger.info("LLM structured call (tool=%s, prompt %s chars): %s", tool_name, len(prompt), head)

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        start = time.time()
        try:
            response = client.messages.create(
                model=self._model_name,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=SYSTEM_PROMPT,
                tools=[result_tool],
                tool_choice={"type": "tool", "name": tool_name},
                messages=messages,
            )
            elapsed = time.time() - start
            for block in response.content:
                if block.type == "tool_use" and block.name == tool_name:
                    result = block.input if isinstance(block.input, dict) else {}
                    logger.info(
                        "LLM structured call finished in %.2fs (tool=%s keys=%s)",
                        elapsed,
                        tool_name,
                        list(result.keys()),
                    )
                    return result
            logger.warning(
                "LLM structured call: tool=%s not found in response content (stop_reason=%s)",
                tool_name,
                response.stop_reason,
            )
            return {}
        except Exception as exc:
            logger.exception("get_structured failed (tool=%s)", tool_name)
            return {}

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull all text blocks from an Anthropic response."""
        parts = [
            block.text
            for block in response.content
            if hasattr(block, "text") and block.text
        ]
        return "\n".join(parts).strip() or "No output from agent."

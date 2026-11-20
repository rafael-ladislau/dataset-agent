"""Fake Anthropic / OpenAI-compatible servers for adapter contract tests.

Each SDK accepts an injected ``http_client``; we give it one whose transport
is an ``httpx.MockTransport`` that replays queued responses and records the
JSON bodies the adapter sent.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:  # openai>=3 / anthropic>=2 build on httpx2; older SDKs use httpx.
    import httpx2 as hx
except ImportError:  # pragma: no cover - depends on installed SDK versions
    import httpx as hx

from dataset_agent.adapters.agent_anthropic import AnthropicAgent
from dataset_agent.adapters.agent_openai import OpenAICompatibleAgent

ANTHROPIC = "anthropic"
OPENAI = "openai"
STYLES = (ANTHROPIC, OPENAI)

Reply = Any  # hx.Response or a callable(request) -> hx.Response


@dataclass
class FakeLLMServer:
    style: str
    queue: list[Reply] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)

    def _handle(self, request: Any) -> Any:
        self.paths.append(request.url.path)
        self.headers.append({k.lower(): v for k, v in request.headers.items()})
        self.requests.append(json.loads(request.content or b"{}"))
        if not self.queue:
            raise AssertionError("fake LLM server received an unexpected request")
        item = self.queue.pop(0)
        return item(request) if callable(item) else item

    def http_client(self) -> Any:
        return hx.Client(transport=hx.MockTransport(self._handle))

    # -- reply builders ------------------------------------------------

    def push(self, *replies: Reply) -> None:
        self.queue.extend(replies)

    def text(self, content: str, **message_extra: Any) -> Any:
        if self.style == ANTHROPIC:
            return hx.Response(
                200,
                json=_anthropic_message([{"type": "text", "text": content}], "end_turn"),
            )
        message = {"role": "assistant", "content": content, **message_extra}
        return hx.Response(200, json=_openai_completion(message, "stop"))

    def tool_call(self, name: str, args: dict[str, Any], call_id: str = "call_1") -> Any:
        if self.style == ANTHROPIC:
            block = {"type": "tool_use", "id": call_id, "name": name, "input": args}
            return hx.Response(200, json=_anthropic_message([block], "tool_use"))
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }
        return hx.Response(200, json=_openai_completion(message, "tool_calls"))

    def structured(self, tool_name: str, obj: dict[str, Any]) -> Any:
        """A reply the adapter's first structured strategy accepts."""
        if self.style == ANTHROPIC:
            return self.tool_call(tool_name, obj)
        return self.text(json.dumps(obj))

    def error(self, status: int, message: str = "unsupported parameter") -> Any:
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
        return hx.Response(status, json=body)

    @staticmethod
    def timeout() -> Callable[[Any], Any]:
        def _raise(request: Any) -> Any:
            raise hx.ReadTimeout("timed out", request=request)

        return _raise

    # -- request helpers -----------------------------------------------

    def user_texts(self, index: int) -> list[str]:
        """Plain-text user message contents of the *index*-th request."""
        out = []
        for m in self.requests[index]["messages"]:
            if m["role"] == "user" and isinstance(m["content"], str):
                out.append(m["content"])
        return out


def _anthropic_message(content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "fake",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _openai_completion(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 0,
        "model": "fake",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
    }


def make_agent(style: str, server: FakeLLMServer, *, model: str = "fake-model", **kwargs: Any):
    common = {
        "model_name": model,
        "api_key": "test",
        "http_client": server.http_client(),
        "max_retries": 0,
        **kwargs,
    }
    if style == ANTHROPIC:
        return AnthropicAgent(base_url="http://fake-llm", **common)
    return OpenAICompatibleAgent(base_url="http://fake-llm/v1", **common)

"""LangChain tool-calling agent (Ollama or OpenRouter)."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from dataset_agent.domain.ports import AgentPort

logger = logging.getLogger(__name__)


def _coerce_tool_str(
    v: Any,
    *,
    dict_keys: tuple[str, ...] = (),
) -> str:
    """Normalize inconsistent LLM outputs to a string (avoids ValidationError on tools)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, list):
        parts = [_coerce_tool_str(x, dict_keys=dict_keys) for x in v]
        return " ".join(p for p in parts if p).strip()
    if isinstance(v, dict):
        for key in dict_keys:
            if key in v and v[key] not in (None, "", [], {}):
                return _coerce_tool_str(v[key], dict_keys=dict_keys)
        if len(v) == 1:
            return _coerce_tool_str(next(iter(v.values())), dict_keys=dict_keys)
        return ""
    return str(v).strip()


class WebSearchArgs(BaseModel):
    model_config = {"extra": "ignore"}

    query: str = Field(
        default="",
        description="Non-empty search query string for the public web.",
    )

    @field_validator("query", mode="before")
    @classmethod
    def _query_coerce(cls, v: Any) -> str:
        return _coerce_tool_str(
            v,
            dict_keys=("query", "search_query", "q", "search", "input", "text", "term"),
        )


class MakeRequestArgs(BaseModel):
    model_config = {"extra": "ignore"}

    url: str = Field(
        default="",
        description="Full http(s) URL to fetch with GET.",
    )

    @field_validator("url", mode="before")
    @classmethod
    def _url_coerce(cls, v: Any) -> str:
        return _coerce_tool_str(
            v,
            dict_keys=("url", "href", "link", "uri", "address"),
        )


def _web_search_impl(query: str) -> str:
    qprev = (query[:120] + "…") if len(query) > 120 else query
    logger.info("Tool web_search: query=%r", qprev)
    try:
        from ddgs import DDGS
    except ImportError:
        return "web_search: ddgs not installed (pip install ddgs)"
    try:
        hits = list(DDGS().text(query, max_results=6))
        if not hits:
            logger.info("Tool web_search: 0 results")
            return "web_search: no results"
        logger.info("Tool web_search: %s results", len(hits))
        parts = []
        for h in hits:
            body = (h.get("body") or "")[:400]
            title = h.get("title") or ""
            href = h.get("href") or ""
            parts.append(f"- {title}\n  {href}\n  {body}")
        return "\n".join(parts)
    except Exception as e:
        logger.warning("web_search failed: %s", e)
        return f"web_search error: {e}"


def _make_request_impl(url: str) -> str:
    if not url.strip():
        return "request error: empty URL"
    logger.info("Tool make_request: GET %s", url[:200] + ("…" if len(url) > 200 else ""))
    try:
        r = httpx.get(url, timeout=30.0, follow_redirects=True)
        preview = (r.text or "")[:800]
        logger.info(
            "Tool make_request: status=%s bytes=%s",
            r.status_code,
            len(r.content),
        )
        return f"status_code={r.status_code} content_length={len(r.content)} preview={preview!r}"
    except Exception as e:
        return f"request error: {e}"


def _web_search_run(query: str = "") -> str:
    q = (query or "").strip()
    if not q:
        return (
            "web_search: the model sent an empty or invalid query. "
            "Call again with a single non-empty search string in the 'query' argument."
        )
    return _web_search_impl(q)


def _make_request_run(url: str = "") -> str:
    return _make_request_impl(url or "")


web_search = StructuredTool.from_function(
    name="web_search",
    description=(
        "Search the public web for information. "
        "Input: one string argument `query` with the search terms (non-empty)."
    ),
    func=_web_search_run,
    args_schema=WebSearchArgs,
)

make_request = StructuredTool.from_function(
    name="make_request",
    description=(
        "HTTP GET a URL and return status code and a short text preview. "
        "Input: one string argument `url` with a full http(s) URL."
    ),
    func=_make_request_run,
    args_schema=MakeRequestArgs,
)


class LangChainAgent(AgentPort):
    def __init__(
        self,
        *,
        provider: str,
        model_name: str,
        temperature: float = 0.6,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_iterations: int = 5,
        timeout_seconds: int = 120,
    ):
        self._provider = provider
        self._model_name = model_name
        self._temperature = temperature
        self._api_key = api_key
        self._base_url = base_url
        self._max_iterations = max_iterations
        self._timeout_seconds = timeout_seconds
        self._executor: AgentExecutor | None = None

    def _build_executor(self) -> AgentExecutor:
        logger.info(
            "Initializing LLM agent: provider=%s model=%s max_iter=%s timeout_s=%s",
            self._provider,
            self._model_name,
            self._max_iterations,
            self._timeout_seconds,
        )
        tools = [web_search, make_request]

        if self._provider == "ollama":
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model=self._model_name,
                temperature=self._temperature,
                top_k=20,
                top_p=0.95,
                base_url=self._base_url or "http://127.0.0.1:11434",
            )
        elif self._provider == "openrouter":
            from langchain_openai import ChatOpenAI

            if not self._api_key:
                raise ValueError("openrouter requires DATASET_AGENT_OPENROUTER_API_KEY")
            llm = ChatOpenAI(
                model=self._model_name,
                temperature=self._temperature,
                api_key=self._api_key,
                base_url=self._base_url or "https://openrouter.ai/api/v1",
            )
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

        try:
            from langsmith import Client as LangSmithClient

            prompt = LangSmithClient().pull_prompt("hwchase17/openai-tools-agent")
            logger.info("Hub prompt (LangSmith) loaded: hwchase17/openai-tools-agent")
        except Exception as e:
            logger.warning("Hub prompt unavailable (%s); using local fallback prompt", e)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant that can use tools to answer the user's question."),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

        agent = create_tool_calling_agent(llm, tools, prompt)
        logger.info("AgentExecutor created (tool calling)")
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=self._max_iterations,
            max_execution_time=self._timeout_seconds,
            # langchain_classic: only "force" on Agent base (tool-calling does not support "generate")
            early_stopping_method="force",
        )

    def get_information(self, prompt: str) -> str:
        if not self._executor:
            self._executor = self._build_executor()
        head = prompt.strip().replace("\n", " ")[:160]
        if len(prompt) > 160:
            head += "…"
        logger.info("LLM agent call (prompt %s chars): %s", len(prompt), head)
        start = time.time()
        try:
            out = self._executor.invoke({"input": prompt})
            logger.info("LLM agent finished in %.2fs", time.time() - start)
            return str(out.get("output") or "").strip() or "No output from agent."
        except Exception as e:
            logger.exception("Agent failed")
            return f"Error in agent execution: {e}"

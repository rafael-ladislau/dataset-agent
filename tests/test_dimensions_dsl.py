"""Dimensions DSL helpers (pagination, parsers)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from dataset_agent.adapters.dimensions_dsl import (
    build_search_publications_paged_dsl,
    fetch_top_n,
    publication_title,
    _parse_retry_after_seconds,
)
from dataset_agent.adapters.dimensions_dsl import AsyncThrottledDimensionsDsl
from dataset_agent.settings import Settings


def test_build_search_publications_paged_dsl_has_limit_and_skip() -> None:
    q = build_search_publications_paged_dsl(
        '(\\"alpha\\" OR \\"beta\\")',
        search_in="full_data",
        limit=500,
        skip=1000,
    )
    assert "limit 500 skip 1000" in q
    assert "return publications[basics+title+times_cited]" in q


def test_fetch_top_n_stops_when_empty_page() -> None:
    class _Port:
        def __init__(self) -> None:
            self.calls = 0

        async def execute_dsl(self, dsl: str) -> SimpleNamespace:
            self.calls += 1
            if self.calls == 1:
                pubs = [{"basics": {"title": f"t{i}"}} for i in range(1000)]
                return SimpleNamespace(publications=pubs, count_total=2000)
            pubs = [{"basics": {"title": f"u{i}"}} for i in range(500)]
            return SimpleNamespace(publications=pubs, count_total=2000)

    async def _run() -> None:
        p = _Port()
        pubs, ncalls = await fetch_top_n(
            p,
            '(\\"x\\")',
            max_results=1500,
            page_size=1000,
        )
        assert ncalls == 2
        assert len(pubs) == 1500

    asyncio.run(_run())


def test_fetch_top_n_respects_max_results_single_page() -> None:
    class _Port:
        async def execute_dsl(self, dsl: str) -> SimpleNamespace:
            pubs = [{"basics": {"title": f"t{i}"}} for i in range(100)]
            return SimpleNamespace(publications=pubs, count_total=100)

    async def _run() -> None:
        pubs, ncalls = await fetch_top_n(_Port(), r"(\"a\")", max_results=40, page_size=1000)
        assert ncalls == 1
        assert len(pubs) == 40

    asyncio.run(_run())


def test_publication_title_from_basics() -> None:
    assert publication_title({"basics": {"title": " Hello "}}) == "Hello"


def test_parse_retry_after_seconds() -> None:
    assert _parse_retry_after_seconds("HTTP 429 Retry-After: 1.5") == 1.5
    assert _parse_retry_after_seconds("too many requests") is None


def test_execute_dsl_retries_once_on_429(monkeypatch) -> None:
    async def _run() -> None:
        calls = {"n": 0}

        async def fake_to_thread(fn, /, *args, **kwargs):  # noqa: ANN001
            _ = fn, args, kwargs
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("HTTP 429 Retry-After: 0.01")
            return SimpleNamespace(publications=[], count_total=0)

        monkeypatch.setattr(
            "dataset_agent.adapters.dimensions_dsl.asyncio.to_thread",
            fake_to_thread,
        )
        dsl = AsyncThrottledDimensionsDsl(
            Settings(
                dimensions_api_key="k",
                dimensions_rate_limit_seconds=0.0,
            )
        )
        out = await dsl.execute_dsl("search publications return publications limit 1")
        assert calls["n"] == 2
        assert getattr(out, "count_total", -1) == 0

    asyncio.run(_run())

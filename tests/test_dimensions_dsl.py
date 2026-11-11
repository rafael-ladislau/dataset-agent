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


def test_parse_total_count_raises_on_result_with_errors() -> None:
    import pytest

    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError, parse_total_count

    bad = SimpleNamespace(count_total=None, errors={"query": {"header": "Query Error"}})
    with pytest.raises(DimensionsDslError, match="Query Error"):
        parse_total_count(bad)
    assert parse_total_count(SimpleNamespace(count_total=5, errors=None)) == 5
    assert parse_total_count(SimpleNamespace(count_total=0, errors={})) == 0


def test_execute_dsl_and_alias_count_raise_on_errors_result(monkeypatch) -> None:
    import pytest

    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError, run_alias_count

    async def _run() -> None:
        async def fake_to_thread(fn, /, *args, **kwargs):  # noqa: ANN001
            return SimpleNamespace(count_total=None, errors={"query": "QuerySyntaxErrors"})

        monkeypatch.setattr("dataset_agent.adapters.dimensions_dsl.asyncio.to_thread", fake_to_thread)
        dsl = AsyncThrottledDimensionsDsl(
            Settings(dimensions_api_key="k", dimensions_rate_limit_seconds=0.0)
        )
        with pytest.raises(DimensionsDslError, match="QuerySyntaxErrors"):
            await dsl.execute_dsl("search publications return publications limit 1")
        with pytest.raises(DimensionsDslError):
            await run_alias_count(dsl, "Medical Imaging & Data Resource Center")

    asyncio.run(_run())


def test_alias_count_dsl_escapes_ampersand() -> None:
    from dataset_agent.adapters.dimensions_dsl import build_publications_count_dsl

    dsl = build_publications_count_dsl("Medical Imaging & Data Resource Center")
    assert 'Medical Imaging \\& Data' in dsl


_AOU_ERRORS = {
    "query": {
        "header": "Service unavailable",
        "details": ["Unexpected Error 'NoneType' object has no attribute 'simplify'"],
        "query": 'search publications for "AoU"',
    }
}


def _patched_dsl(monkeypatch, outcomes):
    """AsyncThrottledDimensionsDsl whose worker thread yields *outcomes* in order."""
    calls = {"n": 0}

    async def fake_to_thread(fn, /, *args, **kwargs):  # noqa: ANN001
        i = calls["n"]
        calls["n"] += 1
        out = outcomes[min(i, len(outcomes) - 1)]
        if isinstance(out, BaseException):
            raise out
        return out

    monkeypatch.setattr("dataset_agent.adapters.dimensions_dsl.asyncio.to_thread", fake_to_thread)
    dsl = AsyncThrottledDimensionsDsl(
        Settings(dimensions_api_key="k", dimensions_rate_limit_seconds=0.0),
        transient_backoff=(0.0, 0.0),
    )
    return dsl, calls


def test_is_transient_dsl_errors_classification() -> None:
    from dataset_agent.adapters.dimensions_dsl import is_transient_dsl_errors

    assert is_transient_dsl_errors(_AOU_ERRORS) is True
    assert is_transient_dsl_errors({"query": "QuerySyntaxErrors"}) is False
    assert is_transient_dsl_errors({"query": {"header": "LexerErrors"}}) is False
    assert is_transient_dsl_errors({"query": {"header": "Semantic errors"}}) is False
    echoed = {"query": {"header": "QuerySyntaxErrors", "query": "service unavailable"}}
    assert is_transient_dsl_errors(echoed) is False


def test_execute_dsl_retries_transient_service_unavailable(monkeypatch) -> None:
    good = SimpleNamespace(count_total=3, errors=None)
    bad = SimpleNamespace(count_total=None, errors=_AOU_ERRORS)
    dsl, calls = _patched_dsl(monkeypatch, [bad, bad, good])
    out = asyncio.run(dsl.execute_dsl("search publications return publications limit 1"))
    assert calls["n"] == 3 and out is good


def test_execute_dsl_gives_up_after_two_transient_retries(monkeypatch) -> None:
    import pytest

    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError

    bad = SimpleNamespace(count_total=None, errors=_AOU_ERRORS)
    dsl, calls = _patched_dsl(monkeypatch, [bad])
    with pytest.raises(DimensionsDslError) as ei:
        asyncio.run(dsl.execute_dsl("search publications return publications limit 1"))
    assert calls["n"] == 3 and ei.value.transient is True


def test_execute_dsl_never_retries_syntax_errors(monkeypatch) -> None:
    import pytest

    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError

    bad = SimpleNamespace(count_total=None, errors={"query": "QuerySyntaxErrors"})
    dsl, calls = _patched_dsl(monkeypatch, [bad])
    with pytest.raises(DimensionsDslError):
        asyncio.run(dsl.execute_dsl("search publications return publications limit 1"))
    assert calls["n"] == 1


def test_execute_dsl_retries_connection_error_and_5xx(monkeypatch) -> None:
    good = SimpleNamespace(count_total=1, errors=None)
    dsl, calls = _patched_dsl(
        monkeypatch,
        [ConnectionError("reset"), RuntimeError("HTTP 503 Service Unavailable"), good],
    )
    out = asyncio.run(dsl.execute_dsl("search publications return publications limit 1"))
    assert calls["n"] == 3 and out is good


def test_execute_dsl_second_429_still_raises(monkeypatch) -> None:
    import pytest

    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError

    dsl, calls = _patched_dsl(monkeypatch, [RuntimeError("HTTP 429 Retry-After: 0.01")])
    with pytest.raises(DimensionsDslError):
        asyncio.run(dsl.execute_dsl("search publications return publications limit 1"))
    assert calls["n"] == 2


def test_run_dsl_with_transient_retry_sync() -> None:
    import pytest

    from dataset_agent.adapters.dimensions_dsl import (
        DimensionsDslError,
        run_dsl_with_transient_retry,
    )

    good = SimpleNamespace(count_total=1, errors=None)
    bad = SimpleNamespace(count_total=None, errors=_AOU_ERRORS)
    seq = [bad, bad, good]
    sleeps: list[float] = []
    out = run_dsl_with_transient_retry(
        lambda q: seq.pop(0), "q", backoff=(1.0, 4.0), sleep=sleeps.append
    )
    assert out is good and sleeps == [1.0, 4.0]

    sleeps.clear()
    syntax = SimpleNamespace(count_total=None, errors={"query": "QuerySyntaxErrors"})
    with pytest.raises(DimensionsDslError):
        run_dsl_with_transient_retry(lambda q: syntax, "q", sleep=sleeps.append)
    assert sleeps == []


def test_short_dsl_error_omits_query_text() -> None:
    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError, short_dsl_error

    exc = DimensionsDslError("Dimensions rejected the query: {...}", errors=_AOU_ERRORS)
    s = short_dsl_error(exc)
    assert s.startswith("Service unavailable: Unexpected Error")
    assert "search publications" not in s


def _page_runner(fail_if, total=450):
    """Fake Dimensions: serves ids 0..total-1; raises the simplify error when fail_if(limit, skip)."""
    import re

    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError

    calls: list[tuple[int, int]] = []

    def run(dsl: str):
        lim = int(re.search(r"limit (\d+)", dsl).group(1))
        sk = int(re.search(r"skip (\d+)", dsl).group(1))
        calls.append((lim, sk))
        if fail_if(lim, sk):
            raise DimensionsDslError("rejected", errors=_AOU_ERRORS, transient=True)
        rows = [{"id": f"p{i}", "times_cited": total - i} for i in range(sk, min(total, sk + lim))]
        return SimpleNamespace(
            publications=rows, count_total=total, errors=None, get=lambda k, d=None: rows
        )

    return run, calls


def test_fetch_publications_resilient_pages_after_big_pull_fails() -> None:
    from dataset_agent.adapters.dimensions_dsl import fetch_publications_resilient

    run, calls = _page_runner(lambda lim, sk: lim >= 200)
    pubs, total, paged = fetch_publications_resilient(run, "x", fields="basics", limit=1000)
    assert paged and total == 450
    assert [p["id"] for p in pubs] == [f"p{i}" for i in range(450)]  # times_cited order kept
    assert calls[0] == (1000, 0) and calls[1:] == [(100, 0), (100, 100), (100, 200), (100, 300), (100, 400)]


def test_fetch_publications_resilient_halves_then_skips_bad_page() -> None:
    from dataset_agent.adapters.dimensions_dsl import fetch_publications_resilient

    # page at skip 100 fails for any size >= 10 covering rows 100..109 -> skipped at floor
    run, calls = _page_runner(lambda lim, sk: lim >= 200 or sk == 100)
    pubs, _total, paged = fetch_publications_resilient(run, "x", fields="basics", limit=1000)
    ids = [p["id"] for p in pubs]
    assert paged and not {f"p{i}" for i in range(100, 110)} & set(ids) and "p110" in ids
    assert "p99" in ids and "p200" in ids
    assert (50, 100) in calls and (25, 100) in calls and (12, 100) in calls and (10, 100) in calls


def test_fetch_publications_resilient_no_paging_for_syntax_or_success() -> None:
    import pytest

    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError, fetch_publications_resilient

    run, calls = _page_runner(lambda lim, sk: False)
    pubs, total, paged = fetch_publications_resilient(run, "x", fields="basics", limit=1000)
    assert not paged and len(pubs) == 450 and total == 450 and len(calls) == 1

    def syntax(dsl: str):
        raise DimensionsDslError("bad", errors={"query": "QuerySyntaxErrors"}, transient=False)

    with pytest.raises(DimensionsDslError):
        fetch_publications_resilient(syntax, "x", fields="basics", limit=1000)


def test_fetch_publications_resilient_outage_stops_early() -> None:
    import pytest

    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError, fetch_publications_resilient

    run, calls = _page_runner(lambda lim, sk: True)
    with pytest.raises(DimensionsDslError):
        fetch_publications_resilient(run, "x", fields="basics", limit=1000)
    assert len(calls) <= 1 + 5  # big pull + one halving chain, then abort


def test_fetch_publications_resilient_caps_consecutive_skips() -> None:
    from dataset_agent.adapters.dimensions_dsl import fetch_publications_resilient

    run, calls = _page_runner(lambda lim, sk: lim >= 200 or sk >= 100)
    pubs, _t, paged = fetch_publications_resilient(run, "x", fields="basics", limit=1000)
    assert paged and [p["id"] for p in pubs] == [f"p{i}" for i in range(100)]
    assert len(calls) <= 1 + 1 + 2 * 5


def test_fetch_publications_resilient_returns_partial_on_non_server_error() -> None:
    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError, fetch_publications_resilient

    run, _calls = _page_runner(lambda lim, sk: lim >= 200)
    n = {"c": 0}

    def budgeted(dsl: str):
        n["c"] += 1
        if n["c"] > 3:
            raise DimensionsDslError("Dimensions call budget exhausted")
        return run(dsl)

    pubs, _t, paged = fetch_publications_resilient(budgeted, "x", fields="basics", limit=1000)
    assert paged and [p["id"] for p in pubs] == [f"p{i}" for i in range(200)]


def test_sanitize_alias_escapes_extraction_phrase_set() -> None:
    from dataset_agent.adapters.dimensions_dsl import inner_quoted_term, sanitize_alias

    # Same characters as the extraction server's escape_phrase.
    for ch in "&|!(){}[]^~*?:/+":
        assert sanitize_alias(f"A{ch}B") == f"A\\{ch}B"
    assert inner_quoted_term("MIDAS? survey") == "MIDAS\\? survey"
    assert inner_quoted_term("MIDRC Data Commons") == "MIDRC Data Commons"
    # Each special character is escaped exactly once (no double escaping).
    assert sanitize_alias("a?b") == "a\\?b"
    assert sanitize_alias("a\\b") == "a\\\\b"

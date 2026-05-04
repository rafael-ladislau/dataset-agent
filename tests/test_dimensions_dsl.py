"""Tests for Dimensions DSL helpers and throttled client."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dataset_agent.adapters.dimensions_dsl import (
    AsyncThrottledDimensionsDsl,
    DimensionsDslError,
    build_publications_count_dsl,
    inner_quoted_term,
    parse_total_count,
    sanitize_alias,
)
from dataset_agent.settings import Settings


def test_sanitize_alias_escapes_special_characters() -> None:
    assert sanitize_alias("foo:bar") == r"foo\:bar"
    assert sanitize_alias("a|b") == r"a\|b"
    assert sanitize_alias('say "hi"') == r"say \"hi\""


def test_inner_quoted_term_matches_sanitize() -> None:
    assert inner_quoted_term("  O*NET  ") == sanitize_alias("O*NET")


def test_build_publications_count_dsl_contains_escaped_term() -> None:
    q = build_publications_count_dsl("O*NET", limit=1)
    assert "search publications in full_data" in q
    assert "limit 1" in q
    assert "O*NET" in q


def test_parse_total_count_from_count_total() -> None:
    assert parse_total_count(SimpleNamespace(count_total=123)) == 123


def test_parse_total_count_from_stats() -> None:
    r = SimpleNamespace(_stats={"total_count": "7"})
    assert parse_total_count(r) == 7


def test_parse_total_count_none() -> None:
    assert parse_total_count(None) == 0


def test_async_throttled_dimensions_dsl_requires_api_key() -> None:
    settings = Settings(dimensions_api_key="")
    client = AsyncThrottledDimensionsDsl(settings)

    async def _run() -> None:
        with pytest.raises(DimensionsDslError, match="API key"):
            await client.execute_dsl("search publications return year limit 1")

    asyncio.run(_run())


def test_async_throttled_dimensions_dsl_serializes_starts_with_sleep() -> None:
    settings = Settings(dimensions_api_key="test-key", dimensions_rate_limit_seconds=0.2)
    client = AsyncThrottledDimensionsDsl(settings)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def fake_to_thread(fn, /, *args, **kwargs):  # noqa: ANN001, ANN003
        return SimpleNamespace(count_total=1)

    async def _run() -> None:
        with (
            patch("dataset_agent.adapters.dimensions_dsl.asyncio.sleep", side_effect=fake_sleep),
            patch("dataset_agent.adapters.dimensions_dsl.asyncio.to_thread", side_effect=fake_to_thread),
        ):
            await client.execute_dsl("q1")
            await client.execute_dsl("q2")

    asyncio.run(_run())
    assert len(sleeps) == 1
    assert sleeps[0] > 0.15


def test_fp_sample_size_validation() -> None:
    with pytest.raises(ValueError):
        Settings(fp_sample_size=0)
    with pytest.raises(ValueError):
        Settings(fp_sample_size=10_001)

"""Concurrency expectations for Dimensions DSL throttle (Etapa 2)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from dataset_agent.adapters.dimensions_dsl import AsyncThrottledDimensionsDsl
from dataset_agent.settings import Settings


def test_serial_throttle_spaces_starts_when_queries_finish_quickly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent execute_dsl calls must not start the worker back-to-back faster than rate."""

    async def _run() -> None:
        starts: list[float] = []

        async def fake_to_thread(fn, /, *args, **kwargs):  # noqa: ANN001
            starts.append(time.monotonic())
            return SimpleNamespace(count_total=1)

        monkeypatch.setattr(
            "dataset_agent.adapters.dimensions_dsl.asyncio.to_thread",
            fake_to_thread,
        )

        settings = Settings(
            dimensions_api_key="test-key-for-throttle",
            dimensions_rate_limit_seconds=0.12,
        )
        dsl = AsyncThrottledDimensionsDsl(settings)

        await asyncio.gather(
            dsl.execute_dsl("search publications return publications limit 1"),
            dsl.execute_dsl("search publications return publications limit 1"),
        )

        assert len(starts) == 2
        gap = starts[1] - starts[0]
        assert gap >= settings.dimensions_rate_limit_seconds * 0.9, (
            f"expected ~>={settings.dimensions_rate_limit_seconds}s between starts, got {gap:.4f}s"
        )

    asyncio.run(_run())

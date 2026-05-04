"""Dimensions DSL string helpers and rate-limited async client (dimcli)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, cast

from dataset_agent.domain.ports import DimensionsDslPort
from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)

# See docs/spec-gap-analysis.md §2.2 (plus backslash and double-quote for inner terms).
_DSL_INNER_SPECIAL = frozenset('^:~[]{}()!|&+"')


class DimensionsDslError(RuntimeError):
    """Raised when Dimensions credentials, dimcli, or DSL execution fails."""


def sanitize_alias(text: str) -> str:
    """Escape characters that break Dimensions ``for`` / search strings inside ``\\\"...\\\"``."""
    if not text:
        return ""
    out: list[str] = []
    for ch in text.strip():
        if ch == "\\" or ch in _DSL_INNER_SPECIAL:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def inner_quoted_term(text: str) -> str:
    """Sanitized token placed between ``\\\"`` and ``\\\"`` like the literature gate."""
    return sanitize_alias(text)


def build_publications_count_dsl(
    term: str,
    *,
    search_in: str = "full_data",
    limit: int = 1,
) -> str:
    """DSL to fetch at most ``limit`` publication ids (use ``parse_total_count`` for the hit count)."""
    inner = inner_quoted_term(term)
    fragment = f'\\"{inner}\\"'
    lim = max(1, int(limit))
    return (
        f'search publications in {search_in} for "{fragment}" '
        f"return publications[id] limit {lim}"
    )


def build_for_clause_count_dsl(
    for_clause: str,
    *,
    search_in: str = "full_data",
    limit: int = 1,
) -> str:
    """Total hit count DSL for an arbitrary boolean ``for`` clause (scope comparison, etc.)."""
    lim = max(1, int(limit))
    return (
        f'search publications in {search_in} for "{for_clause}" '
        f"return publications[id] limit {lim}"
    )


def build_search_publications_dsl(
    for_clause: str,
    *,
    search_in: str = "full_data",
    limit: int = 20,
) -> str:
    """Full Dimensions search for probing a ``for`` clause (titles + counts)."""
    lim = max(1, int(limit))
    return (
        f'search publications in {search_in} for "{for_clause}" '
        "return publications[basics+title+times_cited] "
        f"sort by times_cited limit {lim}"
    )


def publications_from_result(result: object) -> list[dict]:
    """Normalize dimcli ``.query`` payload to a list of publication dicts."""
    if result is None:
        return []
    pubs: list[Any] = []
    if hasattr(result, "get"):
        got = result.get("publications", [])
        pubs = cast(list[Any], got) if isinstance(got, list) else []
    if not pubs and hasattr(result, "publications"):
        raw = getattr(result, "publications", None) or []
        pubs = cast(list[Any], raw) if isinstance(raw, list) else []
    out: list[dict] = []
    for p in pubs:
        if isinstance(p, dict):
            out.append(p)
    return out


def publication_title(pub: dict) -> str:
    """Best-effort title from a Dimensions publication record."""
    basics = pub.get("basics")
    if isinstance(basics, dict):
        t = basics.get("title")
        if isinstance(t, str) and t.strip():
            return t.strip()
    t2 = pub.get("title")
    return t2.strip() if isinstance(t2, str) else ""


def top_titles_from_result(result: object, *, max_titles: int = 10) -> list[str]:
    titles: list[str] = []
    for pub in publications_from_result(result):
        tt = publication_title(pub)
        if tt:
            titles.append(tt)
        if len(titles) >= max_titles:
            break
    return titles


def parse_total_count(result: object) -> int:
    """Best-effort total hit count from a dimcli DSL result object."""
    if result is None:
        return 0
    if hasattr(result, "count_total"):
        try:
            return int(getattr(result, "count_total"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    stats = getattr(result, "_stats", None)
    if isinstance(stats, dict):
        try:
            return int(stats.get("total_count", 0))
        except (TypeError, ValueError):
            return 0
    return 0


async def run_alias_count(
    dsl_port: DimensionsDslPort,
    alias: str,
    *,
    search_in: str = "full_data",
) -> int:
    """Run a ``limit 1`` search and return Dimensions' total publication count for ``alias``."""
    q = build_publications_count_dsl(alias, search_in=search_in, limit=1)
    result = await dsl_port.execute_dsl(q)
    return parse_total_count(result)


class AsyncThrottledDimensionsDsl(DimensionsDslPort):
    """Serial throttle between query *starts*; runs dimcli in a worker thread."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock: asyncio.Lock | None = None
        self._last_start_monotonic = 0.0

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def execute_dsl(self, dsl: str) -> Any:
        key = (self._settings.dimensions_api_key or "").strip()
        if not key:
            raise DimensionsDslError(
                "Dimensions API key is not configured (set DATASET_AGENT_DIMENSIONS_API_KEY)",
            )

        rate = max(0.0, float(self._settings.dimensions_rate_limit_seconds))

        async with self._get_lock():
            now = time.monotonic()
            if self._last_start_monotonic > 0.0 and rate > 0.0:
                elapsed = now - self._last_start_monotonic
                if elapsed < rate:
                    await asyncio.sleep(rate - elapsed)
            self._last_start_monotonic = time.monotonic()

            def _sync_run() -> Any:
                try:
                    import dimcli  # type: ignore[import-not-found]
                except ImportError as exc:  # pragma: no cover
                    raise DimensionsDslError(
                        "dimcli is not installed; install the optional dependency to query Dimensions",
                    ) from exc
                dimcli.login(key=key)
                dsl_client = dimcli.Dsl()
                logger.debug("Dimensions DSL: %s", dsl[:500] + ("..." if len(dsl) > 500 else ""))
                return dsl_client.query(dsl)

            return await asyncio.to_thread(_sync_run)

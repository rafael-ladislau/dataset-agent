"""Dimensions DSL string helpers and rate-limited async client (dimcli)."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, cast

from dataset_agent.domain.ports import DimensionsDslPort
from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)

# See docs/spec-gap-analysis.md §2.2 (plus backslash and double-quote for inner terms).
# Verified live (2026-11-11): inside a quoted full_data phrase a bare "&" is a lexer
# error, "\\&" lexes and matches (same count as the "and" variant); "\\\\&" fails again.
_DSL_INNER_SPECIAL = frozenset('^:~[]{}()!|&+"')


TRANSIENT_RETRIES = 2
TRANSIENT_BACKOFF_SECONDS: tuple[float, ...] = (2.0, 5.0)

_SYNTAX_MARKERS = ("querysyntaxerror", "lexererror", "semantic error", "syntax error")
_TRANSIENT_TEXT_RE = re.compile(
    r"service unavailable|bad gateway|gateway timeout|timed out|"
    r"connection (?:reset|aborted|refused)|"
    r"(?:server error|http|status)\D{0,12}\b5\d\d\b|\b5\d\d\b\D{0,12}server error",
    flags=re.I,
)


class _TimeoutRequests:
    """Proxy for the ``requests`` module that adds a default ``timeout`` to HTTP verbs."""

    _VERBS = frozenset({"get", "post", "put", "delete", "patch", "head", "request"})

    def __init__(self, real: Any, timeout: tuple[float, float]) -> None:
        self._real = real
        self.timeout = timeout

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._real, name)
        if name not in self._VERBS:
            return attr

        def _call(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = self.timeout
            return attr(*args, **kwargs)

        return _call


def install_dimcli_timeout(settings: Settings) -> tuple[float, float]:
    """Make every dimcli HTTP call use a (connect, read) timeout. Idempotent.

    dimcli calls ``requests.post`` with no timeout (login and queries), so a stalled
    server blocks the thread forever. The ``requests`` name in dimcli's modules is
    replaced by a proxy that fills in the timeout when the caller gave none.
    """
    timeout = (
        float(settings.dimensions_connect_timeout_seconds),
        float(settings.dimensions_timeout_seconds),
    )
    import importlib

    for mod_name in ("dimcli.core.api", "dimcli.core.auth", "dimcli.utils.version_utils"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:  # pragma: no cover
            continue
        cur = getattr(mod, "requests", None)
        if cur is None:
            continue
        if isinstance(cur, _TimeoutRequests):
            cur.timeout = timeout
        else:
            mod.requests = _TimeoutRequests(cur, timeout)
    return timeout


def dimcli_login_dsl(settings: Settings, key: str) -> Any:
    """``dimcli.login`` + ``dimcli.Dsl()`` with the HTTP timeout installed first."""
    import dimcli  # type: ignore[import-not-found]

    install_dimcli_timeout(settings)
    # verbose=False skips the post-login PyPI version GET and the extra describe query.
    dimcli.login(key=key, verbose=False)
    return dimcli.Dsl()


class DimensionsDslError(RuntimeError):
    """Raised when Dimensions credentials, dimcli, or DSL execution fails."""

    def __init__(
        self,
        message: str = "",
        *,
        errors: object | None = None,
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.errors = errors
        self.transient = transient


def _errors_text(errors: object) -> str:
    """Header + details of a Dimensions errors payload, never the echoed query text."""
    if isinstance(errors, dict):
        q = errors.get("query")
        if isinstance(q, dict):
            parts = [str(q.get("header") or "")]
            det = q.get("details")
            if isinstance(det, (list, tuple)):
                parts.extend(str(d) for d in det)
            elif det:
                parts.append(str(det))
            return " ".join(parts)
        if isinstance(q, str):
            return q
    return str(errors)


def is_transient_dsl_errors(errors: object) -> bool:
    """True for server-side hiccups ('Service unavailable', 'Unexpected Error').

    Syntax, lexer and semantic errors are never transient.
    """
    text = _errors_text(errors).casefold()
    if any(m in text for m in _SYNTAX_MARKERS):
        return False
    return "service unavailable" in text or "unexpected error" in text


def is_transient_exception(exc: BaseException) -> bool:
    """True for connection problems and HTTP 5xx style failures (never 429)."""
    if isinstance(exc, DimensionsDslError):
        return exc.transient
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        import requests  # type: ignore[import-untyped]

        if isinstance(
            exc,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ),
        ):
            return True
    except ImportError:  # pragma: no cover
        pass
    msg = str(exc)
    if "429" in msg or "too many requests" in msg.lower():
        return False
    return bool(_TRANSIENT_TEXT_RE.search(msg))


def short_dsl_error(exc_or_text: object, limit: int = 120) -> str:
    """Short human reason for a Dimensions failure; never includes the query text."""
    errors = getattr(exc_or_text, "errors", None)
    if isinstance(errors, dict) and isinstance(errors.get("query"), dict):
        q = errors["query"]
        header = str(q.get("header") or "").strip()
        det = q.get("details")
        first = str(det[0]).strip() if isinstance(det, (list, tuple)) and det else ""
        text = ": ".join(x for x in (header, first) if x)
        if text:
            return re.sub(r"\s+", " ", text)[:limit]
    return re.sub(r"\s+", " ", str(exc_or_text)).strip()[:limit]


def _parse_retry_after_seconds(message: str) -> float | None:
    """Best-effort parse for Retry-After delay (seconds) from 429-like error text."""
    if not message:
        return None
    m = re.search(r"retry[\s_-]*after\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", message, flags=re.I)
    if not m:
        return None
    try:
        return max(0.0, float(m.group(1)))
    except (TypeError, ValueError):
        return None


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


def build_search_publications_paged_dsl(
    for_clause: str,
    *,
    search_in: str = "full_data",
    limit: int = 1000,
    skip: int = 0,
    fields: str = "basics+title+times_cited",
) -> str:
    """Paginated publications fetch (DSL allows up to 1000 records per request)."""
    lim = max(1, min(1000, int(limit)))
    sk = max(0, int(skip))
    return (
        f'search publications in {search_in} for "{for_clause}" '
        f"return publications[{fields}] "
        f"sort by times_cited limit {lim} skip {sk}"
    )


async def fetch_top_n(
    dsl_port: DimensionsDslPort,
    for_clause: str,
    *,
    max_results: int,
    search_in: str = "full_data",
    page_size: int = 1000,
    fields: str = "basics+title+times_cited",
) -> tuple[list[dict], int]:
    """
    Page through Dimensions until *max_results* dicts are collected or pages dry up.

    Returns ``(publications, dsl_query_count)``. Stops at 50_000 cumulative rows
    (Dimensions pagination policy) to avoid runaway loops.
    """
    page_size = min(1000, max(1, int(page_size)))
    goal = max(0, int(max_results))
    if goal == 0 or not (for_clause or "").strip():
        return [], 0
    out: list[dict] = []
    skip = 0
    dsl_calls = 0
    max_skip = 50_000
    while len(out) < goal and skip <= max_skip:
        chunk = min(page_size, goal - len(out))
        dsl = build_search_publications_paged_dsl(
            for_clause,
            search_in=search_in,
            limit=chunk,
            skip=skip,
            fields=fields,
        )
        result = await dsl_port.execute_dsl(dsl)
        dsl_calls += 1
        pubs = publications_from_result(result)
        if not pubs:
            break
        row = pubs[:chunk]
        out.extend(row)
        if len(pubs) < chunk:
            break
        skip += len(row)
    return out[:goal], dsl_calls


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


def dsl_result_errors(result: object) -> object | None:
    """The ``errors`` payload of a dimcli result, or None when it carries none.

    dimcli returns a ``DslDataset`` with ``errors`` set (instead of raising) when
    Dimensions rejects a query, e.g. a ``QuerySyntaxErrors`` lexer error.
    """
    errors = getattr(result, "errors", None)
    if isinstance(errors, (dict, list, tuple, str)) and errors:
        return errors
    return None


def raise_for_dsl_errors(result: object) -> None:
    """Raise DimensionsDslError when *result* carries Dimensions query errors."""
    errors = dsl_result_errors(result)
    if errors is not None:
        raise DimensionsDslError(
            f"Dimensions rejected the query: {str(errors)[:500]}",
            errors=errors,
            transient=is_transient_dsl_errors(errors),
        )


def run_dsl_with_transient_retry(
    run_query: Any,
    dsl: str,
    *,
    retries: int = TRANSIENT_RETRIES,
    backoff: tuple[float, ...] | None = None,
    sleep: Any = time.sleep,
) -> Any:
    """Run ``run_query(dsl)`` and retry transient Dimensions failures (sync)."""
    delays = TRANSIENT_BACKOFF_SECONDS if backoff is None else backoff
    attempt = 0
    while True:
        try:
            result = run_query(dsl)
            raise_for_dsl_errors(result)
            return result
        except Exception as exc:
            if attempt >= retries or not is_transient_exception(exc):
                raise
            delay = delays[min(attempt, len(delays) - 1)] if delays else 0.0
            attempt += 1
            logger.warning(
                "Dimensions transient failure (attempt %s/%s): %s; retrying in %.1fs",
                attempt,
                retries + 1,
                short_dsl_error(exc),
                delay,
            )
            sleep(delay)


def parse_total_count(result: object) -> int:
    """Total hit count from a dimcli DSL result object.

    Raises DimensionsDslError when the result carries query errors, so a rejected
    query is never read as a count of 0.
    """
    if result is None:
        return 0
    raise_for_dsl_errors(result)
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


async def fetch_reference_titles(
    dsl_port: DimensionsDslPort,
    ids: list[str],
    *,
    batch_size: int = 512,
    max_unique_ids: int = 2000,
) -> dict[str, str]:
    """
    Batch-resolve reference ids to titles via Dimensions.

    Deduplicates *ids*, caps to *max_unique_ids*, batches in groups of
    *batch_size*, and returns ``{id: title}`` for successfully resolved ids.
    """
    deduped = list({str(i).strip() for i in ids if isinstance(i, str) and i.strip()})
    if not deduped:
        return {}
    capped = deduped[:max(1, int(max_unique_ids))]
    bs = max(1, min(1000, int(batch_size)))
    out: dict[str, str] = {}
    for i in range(0, len(capped), bs):
        chunk = capped[i : i + bs]
        id_list = ", ".join(f'"{sanitize_alias(i)}"' for i in chunk)
        dsl = f"search publications where id in [{id_list}] return publications[id+title]"
        try:
            result = await dsl_port.execute_dsl(dsl)
        except Exception as exc:
            logger.warning("fetch_reference_titles batch failed: %s", exc)
            continue
        for pub in publications_from_result(result):
            if not isinstance(pub, dict):
                continue
            pid = str(pub.get("id") or "").strip()
            if not pid:
                continue
            t = publication_title(pub)
            if t:
                out[pid] = t
    logger.info("fetch_reference_titles: resolved %s/%s unique ids", len(out), len(capped))
    return out


class AsyncThrottledDimensionsDsl(DimensionsDslPort):
    """Serial throttle between query *starts*; runs dimcli in a worker thread."""

    def __init__(
        self,
        settings: Settings,
        *,
        transient_backoff: tuple[float, ...] | None = None,
    ) -> None:
        self._settings = settings
        self._transient_backoff = transient_backoff
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
        retry_fallback = max(0.1, rate if rate > 0 else 2.1)

        async with self._get_lock():
            def _sync_run() -> Any:
                try:
                    import dimcli  # type: ignore[import-not-found]  # noqa: F401
                except ImportError as exc:  # pragma: no cover
                    raise DimensionsDslError(
                        "dimcli is not installed; install the optional dependency to query Dimensions",
                    ) from exc
                dsl_client = dimcli_login_dsl(self._settings, key)
                logger.debug("Dimensions DSL: %s", dsl[:500] + ("..." if len(dsl) > 500 else ""))
                return dsl_client.query(dsl)
            retry_extra = 0.0
            rate_retries_left = 1
            transient_left = TRANSIENT_RETRIES
            delays = (
                TRANSIENT_BACKOFF_SECONDS
                if self._transient_backoff is None
                else self._transient_backoff
            )
            transient_used = 0

            def _next_delay() -> float:
                return delays[min(transient_used - 1, len(delays) - 1)] if delays else 0.0

            while True:
                now = time.monotonic()
                wait_base = 0.0
                if self._last_start_monotonic > 0.0 and rate > 0.0:
                    elapsed = now - self._last_start_monotonic
                    if elapsed < rate:
                        wait_base = rate - elapsed
                wait_total = max(wait_base, retry_extra)
                retry_extra = 0.0
                if wait_total > 0.0:
                    await asyncio.sleep(wait_total)
                self._last_start_monotonic = time.monotonic()
                try:
                    result = await asyncio.to_thread(_sync_run)
                    raise_for_dsl_errors(result)
                    return result
                except DimensionsDslError as exc:
                    if exc.transient and transient_left > 0:
                        transient_left -= 1
                        transient_used += 1
                        retry_extra = _next_delay()
                        logger.warning(
                            "Dimensions transient failure: %s; retrying in %.1fs",
                            short_dsl_error(exc),
                            retry_extra,
                        )
                        continue
                    raise
                except Exception as exc:
                    msg = str(exc)
                    is_429 = "429" in msg or "too many requests" in msg.lower()
                    if rate_retries_left > 0 and is_429:
                        rate_retries_left -= 1
                        retry_after = _parse_retry_after_seconds(msg)
                        retry_extra = retry_after if retry_after is not None else retry_fallback
                        logger.warning(
                            "Dimensions returned 429; retrying once after %.2fs",
                            retry_extra,
                        )
                        continue
                    if not is_429 and is_transient_exception(exc) and transient_left > 0:
                        transient_left -= 1
                        transient_used += 1
                        retry_extra = _next_delay()
                        logger.warning(
                            "Dimensions transient failure: %s; retrying in %.1fs",
                            short_dsl_error(exc),
                            retry_extra,
                        )
                        continue
                    raise DimensionsDslError(msg) from exc

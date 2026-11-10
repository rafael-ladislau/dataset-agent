"""Client for the extraction pipeline's task API (probe, full-text analysis, submit).

A *probe* is a throw-away extraction task: it pulls the top-cited Dimensions hits for a
candidate query, fetches open-access full text for the top-N and runs only the
dataset-validation LLM step. The per-publication verdicts are read back with
:meth:`ExtractionClient.get_llm_results`. Probe tasks are removed with :meth:`cleanup`.

Contract: see ``docs/fulltext-refinement.md`` (extraction probe contract v1).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)

_SEARCH_DONE = {"DONE", "COMPLETED", "COMPLETE"}


class ExtractionError(RuntimeError):
    """Extraction API call failed or timed out."""


@dataclass
class ProbeStatus:
    """Normalized view of ``GET /pipelines/{task_id}/status``."""

    search_done: bool
    llm_done: bool
    llm_pending: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


def parse_status(payload: dict[str, Any]) -> ProbeStatus:
    """Normalize the status payload.

    ``search_done``: task status DONE. ``llm_done``: the ``llm_analysis`` block is
    completed with nothing pending, or full-text fetch finished with nothing delivered
    (so nothing was ever queued for the LLM). With a ``fulltext`` block the fetch must
    also be finished (``attempted >= selected``).
    """
    search_done = str(payload.get("status") or "").upper() in _SEARCH_DONE
    llm = payload.get("llm_analysis")
    ft = payload.get("fulltext")
    fetch_finished = True
    delivered = 0
    if isinstance(ft, dict):
        selected = ft.get("selected")
        attempted = int(ft.get("attempted") or 0)
        delivered = int(ft.get("delivered") or 0)
        fetch_finished = selected is not None and attempted >= int(selected)
    pending = 0
    llm_done = False
    if isinstance(llm, dict):
        pending = int(llm.get("pending") or 0)
        # total_queued only counts papers delivered so far, so "completed" is
        # trustworthy only once the full-text fetch has finished.
        llm_done = (
            str(llm.get("status") or "").lower() == "completed"
            and pending == 0
            and fetch_finished
        )
    elif isinstance(ft, dict) and fetch_finished and delivered == 0:
        llm_done = True
    return ProbeStatus(search_done=search_done, llm_done=llm_done, llm_pending=pending, raw=payload)


class ExtractionClient:
    """Thin httpx wrapper around one extraction instance (HTTP Basic auth)."""

    def __init__(
        self,
        base_url: str,
        user: str = "",
        password: str = "",
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        poll_timeout: float = 900.0,
        poll_interval: float = 3.0,
        poll_max_interval: float = 20.0,
    ) -> None:
        auth = (user, password) if user else None
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), auth=auth, timeout=timeout, transport=transport
        )
        self._sleep = sleep
        self._clock = clock
        self._poll_timeout = poll_timeout
        self._poll_interval = poll_interval
        self._poll_max_interval = poll_max_interval

    @classmethod
    def for_target(cls, settings: Settings, target: str, **kw: Any) -> "ExtractionClient":
        if target not in ("dev", "prod"):
            raise ValueError(f"unknown extraction target {target!r}")
        url = getattr(settings, f"extraction_{target}_url")
        if not url:
            raise ExtractionError(f"extraction_{target}_url is not configured")
        return cls(
            url,
            getattr(settings, f"extraction_{target}_user"),
            getattr(settings, f"extraction_{target}_password"),
            poll_timeout=settings.probe_poll_timeout_seconds,
            poll_interval=settings.probe_poll_interval_seconds,
            poll_max_interval=settings.probe_poll_max_interval_seconds,
            **kw,
        )

    def close(self) -> None:
        self._http.close()

    # -- low level ---------------------------------------------------------
    def _request(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self._http.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise ExtractionError(f"{method} {path} failed: {exc}") from exc
        if r.status_code >= 400:
            raise ExtractionError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError:
            return {}

    # -- probe lifecycle ---------------------------------------------------
    def start_probe(self, seed: dict[str, Any], top_n: int, pull_cap: int) -> str:
        """Start a probe task, wait for its Dimensions search, then queue full-text analysis."""
        body = dict(seed)
        body["probe"] = True
        body["probe_pull_cap"] = int(pull_cap)
        task_id = self._request("POST", "/pipelines/start", json=body).get("task_id")
        if not task_id:
            raise ExtractionError("POST /pipelines/start returned no task_id")
        try:
            self.poll_status(task_id, lambda s: s.search_done)
            self._request(
                "POST",
                f"/pipelines/{task_id}/full-publications/analyze",
                json={
                    "fulltext_source": "open",
                    "top_n": int(top_n),
                    "steps": {"validate_datasets": True},
                },
            )
        except Exception:
            self.cleanup(task_id)
            raise
        return task_id

    def poll_status(
        self, task_id: str, until: Callable[[ProbeStatus], bool]
    ) -> ProbeStatus:
        """Poll status with exponential backoff until *until* holds or the timeout elapses."""
        deadline = self._clock() + self._poll_timeout
        delay = self._poll_interval
        while True:
            st = parse_status(self._request("GET", f"/pipelines/{task_id}/status"))
            if until(st):
                return st
            if self._clock() + delay > deadline:
                raise ExtractionError(f"probe {task_id} timed out after {self._poll_timeout:.0f}s")
            self._sleep(delay)
            delay = min(delay * 1.5, self._poll_max_interval)

    def wait_llm_done(self, task_id: str) -> ProbeStatus:
        return self.poll_status(task_id, lambda s: s.llm_done)

    def get_llm_results(self, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pipelines/{task_id}/llm-results")

    def run_probe(self, seed: dict[str, Any], top_n: int, pull_cap: int) -> dict[str, Any]:
        """Start, wait, read verdicts; always cleans the probe task up (also on failure)."""
        task_id = self.start_probe(seed, top_n, pull_cap)
        try:
            self.wait_llm_done(task_id)
            return self.get_llm_results(task_id)
        finally:
            self.cleanup(task_id)

    def cleanup(self, task_id: str) -> bool:
        """Delete a probe task's data. Never raises (best effort)."""
        try:
            self._request("POST", "/pipelines/cleanup", json=[task_id])
            return True
        except ExtractionError as exc:
            logger.warning("probe cleanup failed for %s: %s", task_id, exc)
            return False

    # -- real task ---------------------------------------------------------
    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create the real extraction task (``/pipelines/start``, probe=false)."""
        body = dict(payload)
        body["probe"] = False
        return self._request("POST", "/pipelines/start", json=body)

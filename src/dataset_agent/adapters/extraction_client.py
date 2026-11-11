"""Client for the extraction pipeline's task API (probe, full-text analysis, submit).

A *probe* is a throw-away extraction task: it pulls the top-cited Dimensions hits for a
candidate query, fetches open-access full text in citation-ordered batches until enough
papers have full text (:meth:`ExtractionClient.run_probe_topup`) and runs only the
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
_TERMINAL_FAILURE = {"FAILED", "FAILURE", "ERROR", "CANCELLED", "CANCELED"}


_NOT_ATTEMPTED = "not_attempted"
_BUDGET_EXHAUSTED = "budget_exhausted"

STOP_TARGET_MET = "target_met"
STOP_POOL_EXHAUSTED = "pool_exhausted"
STOP_MAX_BATCHES = "max_batches"


class ExtractionError(RuntimeError):
    """Extraction API call failed or timed out."""


def _pubs(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in (results.get("publications") or []) if isinstance(p, dict) and p.get("publication_id")]


def _ft(pub: dict[str, Any]) -> dict[str, Any]:
    ft = pub.get("fulltext")
    return ft if isinstance(ft, dict) else {}


def is_attempted(pub: dict[str, Any]) -> bool:
    """True once extraction recorded a full-text outcome (delivered or a real miss)."""
    ft = _ft(pub)
    return ft.get("status") == "delivered" or (
        bool(ft) and ft.get("reason") != _NOT_ATTEMPTED
    )


def is_delivered(pub: dict[str, Any]) -> bool:
    return _ft(pub).get("status") == "delivered"


def _citation_order(pubs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Stable sort: ties keep the server's order (citation desc, year desc, id asc).
    return sorted(pubs, key=lambda p: -(p.get("citation_count") or 0))


@dataclass
class TopupResult:
    """Outcome of :meth:`ExtractionClient.run_probe_topup`."""

    results: dict[str, Any]
    batches: int
    attempted: int
    delivered: int
    target: int
    stop_reason: str
    #: OpenAlex refused at least one file for budget; PMC / Europe PMC kept going.
    budget_exhausted: bool = False


@dataclass
class ProbeStatus:
    """Normalized view of ``GET /pipelines/{task_id}/status``."""

    search_done: bool
    llm_done: bool
    llm_pending: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


def failure_reason(payload: dict[str, Any]) -> str | None:
    """Reason when the status payload reports a terminal task failure, else None."""
    status = str(payload.get("status") or "").upper()
    err = payload.get("error_message") or payload.get("error")
    if status in _TERMINAL_FAILURE:
        return str(err or f"task status {status}")[:500]
    return None


def parse_status(payload: dict[str, Any]) -> ProbeStatus:
    """Normalize the status payload.

    ``search_done``: task status DONE. ``llm_done``: the ``llm_analysis`` block is
    completed with nothing pending and ``total_queued >= delivered``, or full-text fetch finished with nothing delivered
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
        # trustworthy only once the full-text fetch has finished and every
        # delivered paper has been queued for analysis.
        total_queued = int(llm.get("total_queued") or 0)
        llm_done = (
            str(llm.get("status") or "").lower() == "completed"
            and pending == 0
            and fetch_finished
            and total_queued >= delivered
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
    def start_search(self, seed: dict[str, Any], pull_cap: int) -> str:
        """Start a probe task and wait for its Dimensions search. Cleans up on failure."""
        body = dict(seed)
        body["probe"] = True
        body["probe_pull_cap"] = int(pull_cap)
        task_id = self._request("POST", "/pipelines/start", json=body).get("task_id")
        if not task_id:
            raise ExtractionError("POST /pipelines/start returned no task_id")
        try:
            self.poll_status(task_id, lambda s: s.search_done)
        except Exception:
            self.cleanup(task_id)
            raise
        return task_id

    def analyze_ids(self, task_id: str, publication_ids: list[str]) -> dict[str, Any]:
        """Queue open full-text fetch + dataset validation for explicit publications."""
        return self._request(
            "POST",
            f"/pipelines/{task_id}/full-publications/analyze",
            json={
                "fulltext_source": "open",
                "publication_ids": list(publication_ids),
                "steps": {"validate_datasets": True},
            },
        )

    def wait_batch_done(self, task_id: str, publication_ids: list[str]) -> dict[str, Any]:
        """Poll until every id in the batch has a full-text outcome and the LLM is idle.

        Done = each id has a recorded full-text outcome in ``llm-results`` (not
        ``not_attempted``), and ``/status`` ``llm_analysis`` is completed with nothing
        pending and ``total_queued`` (cumulative over the run) covering every delivered
        paper of the probe. The
        ``fulltext.selected`` counter is not used: extraction overwrites it per
        analyze call while ``attempted`` accumulates, so it is wrong from batch 2 on.
        Returns the last ``llm-results`` body.
        """
        wanted = set(publication_ids)
        deadline = self._clock() + self._poll_timeout
        delay = self._poll_interval
        while True:
            results = self.get_llm_results(task_id)
            batch = [p for p in _pubs(results) if str(p.get("publication_id")) in wanted]
            fetched = len(batch) >= len(wanted) and all(is_attempted(p) for p in batch)
            if fetched:
                if not any(is_delivered(p) for p in batch):
                    return results
                # total_queued is cumulative over the run's analyze calls, so it is
                # compared with every delivered paper, not only this batch's: a paper
                # whose ledger row is written but whose LLM enqueue is still in
                # flight keeps the batch open.
                delivered = sum(1 for p in _pubs(results) if is_delivered(p))
                raw = self._request("GET", f"/pipelines/{task_id}/status")
                llm = raw.get("llm_analysis") if isinstance(raw, dict) else None
                if (
                    isinstance(llm, dict)
                    and str(llm.get("status") or "").lower() == "completed"
                    and int(llm.get("pending") or 0) == 0
                    and int(llm.get("total_queued") or 0) >= delivered
                ):
                    # Re-read: an analysis that finished between the llm-results read
                    # above and this status read would otherwise be missing.
                    return self.get_llm_results(task_id)
            if self._clock() + delay > deadline:
                raise ExtractionError(
                    f"probe {task_id} batch timed out after {self._poll_timeout:.0f}s"
                )
            self._sleep(delay)
            delay = min(delay * 1.5, self._poll_max_interval)

    def poll_status(
        self, task_id: str, until: Callable[[ProbeStatus], bool]
    ) -> ProbeStatus:
        """Poll status with exponential backoff until *until* holds or the timeout elapses."""
        deadline = self._clock() + self._poll_timeout
        delay = self._poll_interval
        while True:
            st = parse_status(self._request("GET", f"/pipelines/{task_id}/status"))
            reason = failure_reason(st.raw)
            if reason:
                raise ExtractionError(f"probe {task_id} failed: {reason}")
            if until(st):
                return st
            if self._clock() + delay > deadline:
                raise ExtractionError(f"probe {task_id} timed out after {self._poll_timeout:.0f}s")
            self._sleep(delay)
            delay = min(delay * 1.5, self._poll_max_interval)

    def get_llm_results(self, task_id: str, *, contexts: bool = False) -> dict[str, Any]:
        """``contexts`` asks for per-publication ``term_contexts`` (snippets of the seed's
        terms in the full text) and ``category_for``; servers without it ignore the flag."""
        path = f"/pipelines/{task_id}/llm-results"
        if contexts:
            return self._request("GET", path, params={"contexts": "true"})
        return self._request("GET", path)

    def run_probe_topup(
        self,
        seed: dict[str, Any],
        *,
        target: int,
        batch_size: int,
        max_batches: int,
        pull_cap: int,
    ) -> TopupResult:
        """Probe until *target* papers have delivered full text, batch by batch.

        Each batch is the next *batch_size* publications in citation order that have
        no full-text outcome yet, sent as explicit ``publication_ids``. Stops when the
        target is met, the probe's pool is exhausted or *max_batches* were sent. An
        OpenAlex budget miss does not stop the loop (the free providers may still
        deliver) but is reported in ``budget_exhausted``. The probe task is always cleaned up,
        exactly once, also on failure.
        """
        task_id = self.start_search(seed, pull_cap)
        try:
            results = self.get_llm_results(task_id)
            batches = 0
            while True:
                pubs = _citation_order(_pubs(results))
                delivered = sum(1 for p in pubs if is_delivered(p))
                if delivered >= target:
                    stop = STOP_TARGET_MET
                    break
                todo = [str(p["publication_id"]) for p in pubs if not is_attempted(p)]
                if not todo:
                    stop = STOP_POOL_EXHAUSTED
                    break
                if batches >= max_batches:
                    stop = STOP_MAX_BATCHES
                    break
                batch = todo[:batch_size]
                batches += 1
                logger.info(
                    "probe %s batch %d: %d id(s), delivered so far %d/%d",
                    task_id, batches, len(batch), delivered, target,
                )
                self.analyze_ids(task_id, batch)
                results = self.wait_batch_done(task_id, batch)
            if batches:
                # One heavier read at the end: snippets of the seed's terms in each
                # delivered paper show why the query matched it.
                results = self.get_llm_results(task_id, contexts=True)
            pubs = _pubs(results)
            return TopupResult(
                results=results,
                batches=batches,
                attempted=sum(1 for p in pubs if is_attempted(p)),
                delivered=sum(1 for p in pubs if is_delivered(p)),
                target=int(target),
                stop_reason=stop,
                budget_exhausted=any(_ft(p).get("reason") == _BUDGET_EXHAUSTED for p in pubs),
            )
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

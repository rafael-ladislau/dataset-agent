"""Client for the extraction pipeline's task API (probe, full-text analysis, submit).

A *probe* is a throw-away extraction task: it pulls the top-cited Dimensions hits for a
candidate query, fetches open-access full text in citation-ordered batches until enough
papers have full text (:meth:`ExtractionClient.run_probe_topup`) and runs only the
dataset-validation LLM step. The per-publication verdicts are read back with
:meth:`ExtractionClient.get_llm_results`. Probe tasks are removed with :meth:`cleanup`.

AI-model probes (``entity="ai_model"``) start on ``/ai-models/start`` and run only the
AI-model validation step; status, ``llm-results`` and cleanup share the
``/pipelines/...`` routes.

Contract: see ``docs/fulltext-refinement.md`` (extraction probe contract v1; model
task v2.1 in ``tests/fixtures/llm-results-model-task-v2.1.json``).
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

#: Probe entity -> (start route, default analyze steps).
_ENTITY_ROUTES: dict[str, tuple[str, dict[str, bool]]] = {
    "dataset": ("/pipelines/start", {"validate_datasets": True}),
    "ai_model": ("/ai-models/start", {"validate_ai_models": True}),
}


def _entity_route(entity: str) -> tuple[str, dict[str, bool]]:
    try:
        return _ENTITY_ROUTES[entity]
    except KeyError:
        raise ValueError(f"unknown probe entity {entity!r}") from None


def _cleanup_item_failed(item: Any) -> bool:
    """A cleanup item reports an error (``status`` error or an HTTP error code)."""
    if not isinstance(item, dict):
        return True
    if str(item.get("status") or "").lower() == "error":
        return True
    try:
        return int(item.get("status_code") or 0) >= 400
    except (TypeError, ValueError):
        return True


class ExtractionError(RuntimeError):
    """Extraction API call failed or timed out."""


class ProbeUnsupportedError(ExtractionError):
    """The extraction target does not declare probe support, so nothing was started.

    Extraction main accepts ``probe: true`` on POST /pipelines/start and ignores it: a
    "probe" there would start a real pipeline with real analyses and write-backs.
    """


#: Seed fields the agent adds that an older extraction target does not accept (its start
#: schemas forbid unknown fields): sent only when the route's schema declares them.
_OPTIONAL_SEED_FIELDS = ("developer",)


def _request_properties(spec: Any, route: str) -> dict[str, Any]:
    try:
        schema = spec["paths"][route]["post"]["requestBody"]["content"]["application/json"]["schema"]
    except (KeyError, TypeError):
        return {}
    ref = str(schema.get("$ref") or "") if isinstance(schema, dict) else ""
    if ref.startswith("#/components/schemas/"):
        schema = (spec.get("components") or {}).get("schemas", {}).get(ref.rsplit("/", 1)[-1])
    props = schema.get("properties") if isinstance(schema, dict) else None
    return props if isinstance(props, dict) else {}


def openapi_declares_probe(spec: Any, route: str) -> bool:
    """The server's own OpenAPI document declares probe support for *route*: the route's
    request schema has a ``probe`` property, and the cleanup and llm-results routes exist.
    A 200 on start is never taken as support."""
    if not isinstance(spec, dict):
        return False
    paths = spec.get("paths") or {}
    if "/pipelines/cleanup" not in paths or "/pipelines/{task_id}/llm-results" not in paths:
        return False
    try:
        schema = paths[route]["post"]["requestBody"]["content"]["application/json"]["schema"]
    except (KeyError, TypeError):
        return False
    ref = str(schema.get("$ref") or "") if isinstance(schema, dict) else ""
    if ref.startswith("#/components/schemas/"):
        schema = (spec.get("components") or {}).get("schemas", {}).get(ref.rsplit("/", 1)[-1])
    return isinstance(schema, dict) and "probe" in (schema.get("properties") or {})


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
class YearSampleResult:
    """Outcome of :meth:`ExtractionClient.run_year_sample` (the release sample)."""

    #: ``{"publications": [...]}``: the kept papers with full text plus the papers sent
    #: that did not deliver, each with ``sample_year``.
    results: dict[str, Any]
    #: Per year: count, allocation, pull_cap, pool, pool_covers_year, sent, delivered,
    #: kept, batches, stop_reason; a large year sliced by month additionally carries
    #: sliced, allocation_sliced, months (per month: count, count_missing, allocation,
    #: pull_cap, pool, pool_covers_month, sent, delivered, kept, batches, stop_reason),
    #: complete and incomplete (months_capped, months_missing_count, months_outside_window,
    #: months_short, months_quota_short, months_quota_revoked).
    years: dict[int, dict[str, Any]]
    rng_seed: int


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
    #: The probe task's id (already cleaned up when the result is returned).
    task_id: str = ""


@dataclass
class ProbeStatus:
    """Normalized view of ``GET /pipelines/{task_id}/status``."""

    search_done: bool
    raw: dict[str, Any] = field(default_factory=dict)


def failure_reason(payload: dict[str, Any]) -> str | None:
    """Reason when the status payload reports a terminal task failure, else None."""
    status = str(payload.get("status") or "").upper()
    err = payload.get("error_message") or payload.get("error")
    if status in _TERMINAL_FAILURE:
        return str(err or f"task status {status}")[:500]
    return None


def parse_status(payload: dict[str, Any]) -> ProbeStatus:
    """Normalize the status payload. ``search_done``: task status DONE."""
    return ProbeStatus(
        search_done=str(payload.get("status") or "").upper() in _SEARCH_DONE, raw=payload
    )


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
        #: One entry per :meth:`cleanup` call: task_id, ok, items (the 202 body), error.
        self.cleanup_log: list[dict[str, Any]] = []
        #: The server's /openapi.json as last read at a probe start ({}: unavailable).
        self._openapi: dict[str, Any] | None = None

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

    def require_probe_support(self, route: str) -> None:
        """Refuse before any probe unless the server declares probe support for *route*
        in its /openapi.json. Read at every probe start, never cached: after an
        extraction rollback a cached "supported" would let a probe start a real
        pipeline. One GET per probe. Fails closed."""
        try:
            r = self._http.request("GET", "/openapi.json")
            self._openapi = r.json() if r.status_code < 400 else {}
        except (httpx.HTTPError, ValueError):
            self._openapi = {}
        if not openapi_declares_probe(self._openapi, route):
            raise ProbeUnsupportedError(
                f"extraction target does not declare probe support for {route} "
                "(its /openapi.json has no `probe` field or no probe cleanup/llm-results "
                "routes); not probing, because a probe there would start a real pipeline"
            )

    # -- probe lifecycle ---------------------------------------------------
    def start_search(self, seed: dict[str, Any], pull_cap: int, *, entity: str = "dataset") -> str:
        """Start a probe task and wait for its Dimensions search. Cleans up on failure.

        ``entity="ai_model"`` posts the seed to ``/ai-models/start`` (the body is the
        seed plus the probe fields, nothing else is added). A seed may carry
        ``date_range`` (inclusive ISO date window, :meth:`run_year_sample`'s month
        slices): sent only when the route's schema declares the field, else the start
        fails closed -- an older target would silently ignore the dates and pull a
        uniform whole-month pool."""
        route, _ = _entity_route(entity)
        self.require_probe_support(route)
        body = dict(seed)
        declared = _request_properties(self._openapi, route)
        for key in _OPTIONAL_SEED_FIELDS:
            if key in body and key not in declared:
                logger.info("probe start: %s not declared by %s; left out of the body", key, route)
                body.pop(key)
        if "date_range" in body and "date_range" not in declared:
            raise ExtractionError(
                f"extraction target does not declare a `date_range` field on {route}: "
                "a silent whole-year reuse of a dated month slice would give a uniform "
                "(never a month-stratified) sample, so refusing"
            )
        body["probe"] = True
        body["probe_pull_cap"] = int(pull_cap)
        task_id = self._request("POST", route, json=body).get("task_id")
        if not task_id:
            raise ExtractionError(f"POST {route} returned no task_id")
        try:
            self.poll_status(task_id, lambda s: s.search_done)
        except Exception:
            self.cleanup(task_id)
            raise
        return task_id

    def analyze_ids(
        self, task_id: str, publication_ids: list[str], *, steps: dict[str, bool] | None = None
    ) -> dict[str, Any]:
        """Queue open full-text fetch + validation for explicit publications.

        *steps* defaults to dataset validation only."""
        return self._request(
            "POST",
            f"/pipelines/{task_id}/full-publications/analyze",
            json={
                "fulltext_source": "open",
                "publication_ids": list(publication_ids),
                "steps": dict(steps) if steps else {"validate_datasets": True},
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
        entity: str = "dataset",
        steps: dict[str, bool] | None = None,
    ) -> TopupResult:
        """Probe until *target* papers have delivered full text, batch by batch.

        Each batch is the next *batch_size* publications in citation order that have
        no full-text outcome yet, sent as explicit ``publication_ids``. Stops when the
        target is met, the probe's pool is exhausted or *max_batches* were sent. An
        OpenAlex budget miss does not stop the loop (the free providers may still
        deliver) but is reported in ``budget_exhausted``. The probe task is always cleaned up,
        exactly once, also on failure.

        *entity* picks the start route and the default analyze *steps*
        (``dataset``: validate_datasets; ``ai_model``: validate_ai_models).
        """
        _, default_steps = _entity_route(entity)
        steps = dict(steps) if steps else default_steps
        task_id = self.start_search(seed, pull_cap, entity=entity)
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
                self.analyze_ids(task_id, batch, steps=steps)
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
                task_id=str(task_id),
            )
        finally:
            self.cleanup(task_id)

    def _draw_fulltext_in_order(
        self, task_id: str, order: list[str], want: int, *, label: str,
        steps: dict[str, bool], batch_size: int, max_batches: int,
    ) -> dict[str, Any]:
        """Send *order*'s papers for open full text in batches until *want* papers of
        them have delivered (or the pool or the batch cap runs out); the first delivered
        papers in draw order are kept downstream. Shared by the whole-year and the month
        paths of :meth:`run_year_sample`. Returns the papers sent in draw order, the
        delivered ids seen, the batches used, the stop reason and the final
        ``llm-results`` body (with ``term_contexts`` when batches were sent)."""
        from dataset_agent.adapters.release_sample import batch_size_for

        sent: list[str] = []
        delivered: set[str] = set()
        results: dict[str, Any] = {}
        batches, pos = 0, 0
        while True:
            have = len([i for i in sent if i in delivered])
            if have >= want:
                stop = STOP_TARGET_MET
                break
            if pos >= len(order):
                stop = STOP_POOL_EXHAUSTED
                break
            if batches >= max_batches:
                stop = STOP_MAX_BATCHES
                break
            batch = order[pos:pos + batch_size_for(want - have, batch_size)]
            pos += len(batch)
            batches += 1
            sent += batch
            logger.info("release sample %s %s batch %d: %d id(s), %d/%d delivered",
                        task_id, label, batches, len(batch), have, want)
            self.analyze_ids(task_id, batch, steps=steps)
            results = self.wait_batch_done(task_id, batch)
            delivered = {str(p["publication_id"]) for p in _pubs(results) if is_delivered(p)}
        if batches:
            # One heavier read at the end: snippets of the seed's terms in each
            # delivered paper show why the query matched it.
            results = self.get_llm_results(task_id, contexts=True)
            delivered = {str(p["publication_id"]) for p in _pubs(results) if is_delivered(p)}
        return {"sent": sent, "delivered": delivered, "batches": batches, "stop": stop,
                "results": results}

    def _run_sliced_year(
        self, seed: dict[str, Any], *, year: int, want: int, count: int,
        month_counts: dict[int, dict[int, int]], month_allocation: dict[int, int],
        rng_seed: int, label: str, steps: dict[str, bool], batch_size: int,
        max_batches: int, entity: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """One large year sliced by month: one capped probe per month with an
        allocation, each pulling the month through an inclusive ``date_range`` window
        (``[YYYY-MM-01 .. YYYY-MM-END]``, clamped into the seed's year window -- every
        month of a sampled year lies inside it, so the clamp is exact) and keeping its
        own ``count + 50`` headroom. The papers kept are the first delivered in draw
        order per month; the pooled months dedupe. The record's ``allocation`` is the
        year quota aimed for; ``allocation_sliced`` is what the month slices summed to.
        Every incompleteness is on the record: a month pool that reaches its cap
        (``pool_covers_month`` false), a missing month count, a month outside the
        window, a month that ended short, a month-slice total below the aimed quota
        (``months_quota_short``), and, since the bound of *want* kept papers is
        enforced by revocation (not an assert: the sampling keeps going and the record
        keeps the evidence), a merged kept total above the quota cut back to it
        (``months_quota_revoked``). Returns the year record and its kept papers with
        ``sample_year``."""
        from dataset_agent.adapters.release_sample import draw_order, month_window, pull_cap

        within: tuple[int, int] | None = None
        years_range = seed.get("years_range") if isinstance(seed, dict) else None
        if isinstance(years_range, dict):
            try:
                within = (int(years_range["start_year"]), int(years_range["end_year"]))
            except (KeyError, TypeError, ValueError):
                within = None
        months: dict[int, dict[str, Any]] = {}
        pubs_out: list[dict[str, Any]] = []
        seen: set[str] = set()
        kept_ids: dict[int, list[str]] = {}
        for month, m_want in sorted(month_allocation.items()):
            if m_want <= 0:
                continue
            got = dict(month_counts or {})
            count_missing = month not in got
            m_count = int(got.get(month, 0))
            window = month_window(year, month, within=within)
            if window is None:
                months[month] = {"outside_window": True, "count": m_count,
                                 "count_missing": count_missing, "allocation": int(m_want),
                                 "pull_cap": pull_cap(m_count), "pool": 0,
                                 "pool_covers_month": False, "sent": 0, "delivered": 0,
                                 "kept": 0, "batches": 0, "stop_reason": STOP_POOL_EXHAUSTED}
                continue
            body = dict(seed)
            body["years_range"] = {"start_year": int(year), "end_year": int(year)}
            body["date_range"] = {"date_from": window[0].isoformat(),
                                  "date_to": window[1].isoformat()}
            cap = pull_cap(m_count)
            task_id = self.start_search(body, cap, entity=entity)
            try:
                results = self.get_llm_results(task_id)
                pool = [str(p["publication_id"]) for p in _pubs(results)]
                draw = self._draw_fulltext_in_order(
                    task_id, draw_order(pool, rng_seed, year, month=month), int(m_want),
                    label=label, steps=steps, batch_size=batch_size, max_batches=max_batches)
                kept: list[str] = []
                for pid in draw["sent"]:  # in draw order; the pooled months dedupe
                    if pid in draw["delivered"] and pid not in seen and len(kept) < m_want:
                        kept.append(pid)
                        seen.add(pid)
                kept_ids[month] = kept
                keep = set(kept) | {i for i in draw["sent"] if i not in draw["delivered"]}
                served = {str(p["publication_id"]): p for p in _pubs(draw["results"])}
                for pid in draw["sent"]:  # in draw order
                    if pid in keep and pid in served:
                        pubs_out.append({**served[pid], "sample_year": int(year)})
                months[month] = {
                    "count": m_count, "count_missing": count_missing,
                    "allocation": int(m_want), "pull_cap": cap, "pool": len(pool),
                    "pool_covers_month": len(pool) < cap,
                    "sent": len(draw["sent"]),
                    "delivered": len([i for i in draw["sent"] if i in draw["delivered"]]),
                    "kept": len(kept), "batches": draw["batches"],
                    "stop_reason": draw["stop"],
                }
            finally:
                self.cleanup(task_id)
        capped = [m for m, r in sorted(months.items()) if not r.get("pool_covers_month")]
        missing = [m for m, r in sorted(months.items()) if r.get("count_missing")]
        outside = [m for m, r in sorted(months.items()) if r.get("outside_window")]
        short = [m for m, r in sorted(months.items()) if r["stop_reason"] != STOP_TARGET_MET]
        incomplete: dict[str, Any] = {
            "months_capped": capped, "months_missing_count": missing,
            "months_outside_window": outside, "months_short": short,
        }
        allocation_sliced = int(sum(month_allocation.values()))
        quota_short = int(want) - allocation_sliced
        if quota_short > 0:  # the month slices aim less than the year quota asked for
            incomplete["months_quota_short"] = quota_short
        # The bound of *want* kept papers is enforced by revocation (the documented
        # choice, not an assert): a merged kept total above the quota is cut back to it,
        # the tail months first, and the cut lands on the record as
        # ``months_quota_revoked`` -- the sampling keeps going, the evidence stays.
        total_kept = sum(r["kept"] for r in months.values())
        if total_kept > int(want):
            revoked = total_kept - int(want)
            incomplete["months_quota_revoked"] = revoked
            drop_ids: set[str] = set()
            for month in sorted(months, reverse=True):
                if revoked <= 0:
                    break
                ids = kept_ids.get(month) or []
                take = min(revoked, len(ids))
                if take:
                    drop_ids.update(ids[-take:])
                    del ids[-take:]
                    months[month]["kept"] -= take
                    revoked -= take
            if drop_ids:
                pubs_out[:] = [p for p in pubs_out
                               if str(p.get("publication_id")) not in drop_ids]
        # A short month (or a slice total below the aimed quota) also breaks the
        # coverage claims: neither ``pool_covers_year`` nor ``complete`` may then report
        # a uniform sample.
        covered = not (capped or missing or outside or short or quota_short > 0)
        complete = covered and "months_quota_revoked" not in incomplete
        record: dict[str, Any] = {
            "count": int(count), "allocation": int(want),
            "allocation_sliced": allocation_sliced,
            "sliced": True,
            "pool_covers_year": covered,
            "sent": sum(r["sent"] for r in months.values()),
            "delivered": sum(r["delivered"] for r in months.values()),
            "kept": sum(r["kept"] for r in months.values()),
            "batches": sum(r["batches"] for r in months.values()),
            "stop_reason": months[short[0]]["stop_reason"] if short else STOP_TARGET_MET,
            "months": months,
            "complete": complete,
            "incomplete": incomplete,
        }
        return record, pubs_out

    def run_year_sample(
        self,
        seed: dict[str, Any],
        *,
        allocation: dict[int, int],
        year_counts: dict[int, int],
        rng_seed: int,
        batch_size: int,
        max_batches_per_year: int,
        entity: str = "dataset",
        steps: dict[str, bool] | None = None,
        month_counts: dict[int, dict[int, int]] | None = None,
        month_allocation: dict[int, dict[int, int]] | None = None,
    ) -> YearSampleResult:
        """The release sample (``adapters.release_sample``): one probe per year with an
        allocation, papers drawn in the seeded random order of that year and sent for open
        full text until the allocation has delivered (or the pool or the batch cap runs
        out). The first delivered papers in draw order are kept; papers sent that did not
        deliver stay in the result as attempted. Every probe task is cleaned up.

        A year present in *month_allocation* (with the year's own *month_counts*) is a
        large year sliced by month: one capped probe per month with its slice of the
        year's allocation, each with an inclusive ``date_range`` window the extraction
        must declare (``start_search`` refuses a target that does not). Years absent from
        it keep the whole-year probe exactly as before (no ``date_range`` key)."""
        from dataset_agent.adapters.release_sample import draw_order, pull_cap

        _, default_steps = _entity_route(entity)
        steps = dict(steps) if steps else default_steps
        month_counts = month_counts or {}
        month_allocation = month_allocation or {}
        pubs_out: list[dict[str, Any]] = []
        years: dict[int, dict[str, Any]] = {}
        for year, want in sorted(allocation.items()):
            if want <= 0:
                continue
            slices = {int(m): int(k) for m, k in dict(month_allocation.get(year) or {}).items()
                      if int(k) > 0}
            if slices:
                record, keep_pubs = self._run_sliced_year(
                    seed, year=int(year), want=int(want), count=int(year_counts.get(year, 0)),
                    month_counts=dict(month_counts.get(year) or {}), month_allocation=slices,
                    rng_seed=rng_seed, label=f"year {year} month", steps=steps,
                    batch_size=batch_size, max_batches=max_batches_per_year, entity=entity)
                years[int(year)] = record
                pubs_out.extend(keep_pubs)
                continue
            body = dict(seed)
            body["years_range"] = {"start_year": int(year), "end_year": int(year)}
            count = int(year_counts.get(year, 0))
            cap = pull_cap(count)
            task_id = self.start_search(body, cap, entity=entity)
            try:
                results = self.get_llm_results(task_id)
                pool = [str(p["publication_id"]) for p in _pubs(results)]
                draw = self._draw_fulltext_in_order(
                    task_id, draw_order(pool, rng_seed, year), want, label=f"year {year}",
                    steps=steps, batch_size=batch_size, max_batches=max_batches_per_year)
                results, delivered = draw["results"], draw["delivered"]
                kept = [i for i in draw["sent"] if i in delivered][:want]
                keep = set(kept) | {i for i in draw["sent"] if i not in delivered}
                served = {str(p["publication_id"]): p for p in _pubs(results)}
                for pid in draw["sent"]:  # in draw order
                    if pid in keep and pid in served:
                        pubs_out.append({**served[pid], "sample_year": int(year)})
                years[int(year)] = {
                    "count": count, "allocation": int(want), "pull_cap": cap,
                    "pool": len(pool), "pool_covers_year": len(pool) < cap,
                    "sent": len(draw["sent"]),
                    "delivered": len([i for i in draw["sent"] if i in delivered]),
                    "kept": len(kept), "batches": draw["batches"],
                    "stop_reason": draw["stop"],
                }
            finally:
                self.cleanup(task_id)
        return YearSampleResult(results={"publications": pubs_out}, years=years,
                                rng_seed=int(rng_seed))

    def cleanup(self, task_id: str) -> bool:
        """Delete a probe task's data. Never raises (best effort).

        Extraction answers 202 with one item per id; the result is False when the call
        fails or any item reports an error (``status`` error or ``status_code`` >= 400).
        Every call is recorded in :attr:`cleanup_log`."""
        entry: dict[str, Any] = {"task_id": task_id, "ok": False, "items": [], "error": None}
        self.cleanup_log.append(entry)
        try:
            body = self._request("POST", "/pipelines/cleanup", json=[task_id])
        except ExtractionError as exc:
            entry["error"] = str(exc)
            logger.warning("probe cleanup failed for %s: %s", task_id, exc)
            return False
        items = body if isinstance(body, list) else []
        entry["items"] = items
        failed = [it for it in items if _cleanup_item_failed(it)]
        if failed:
            entry["error"] = "; ".join(
                str(it.get("detail") if isinstance(it, dict) else it)[:200] for it in failed
            )
            logger.warning("probe cleanup reported errors for %s: %s", task_id, entry["error"])
            return False
        entry["ok"] = True
        return True

    # -- real task ---------------------------------------------------------
    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create the real extraction task (``/pipelines/start``, probe=false)."""
        body = dict(payload)
        body["probe"] = False
        return self._request("POST", "/pipelines/start", json=body)

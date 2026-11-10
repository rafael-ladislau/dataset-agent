"""ExtractionClient against a fake httpx transport."""

from __future__ import annotations

import json

import httpx
import pytest

from dataset_agent.adapters.extraction_client import (
    ExtractionClient,
    ExtractionError,
    parse_status,
)


class _Server:
    def __init__(self, statuses, fail_analyze=False):
        self.calls: list[tuple[str, str, object]] = []
        self.statuses = list(statuses)
        self.fail_analyze = fail_analyze

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        self.calls.append((req.method, req.url.path, body))
        p = req.url.path
        if p == "/pipelines/start":
            return httpx.Response(202, json={"task_id": "T1"})
        if p == "/pipelines/T1/status":
            s = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return httpx.Response(200, json=s)
        if p == "/pipelines/T1/full-publications/analyze":
            if self.fail_analyze:
                return httpx.Response(400, json={"detail": "boom"})
            return httpx.Response(202, json={"task_id": "T1", "status": "started"})
        if p == "/pipelines/T1/llm-results":
            return httpx.Response(200, json={"task_id": "T1", "probe": True, "publications": []})
        if p == "/pipelines/cleanup":
            return httpx.Response(202, json=[])
        return httpx.Response(404)


def _client(server, **kw):
    t = {"now": 0.0}

    def sleep(d):
        t["now"] += d

    return ExtractionClient(
        "http://x", "u", "p", transport=httpx.MockTransport(server),
        sleep=sleep, clock=lambda: t["now"], poll_timeout=60, poll_interval=1, **kw,
    )


DONE_LLM = {"status": "DONE", "llm_analysis": {"status": "completed", "total_queued": 0, "analyzed": 0, "failed": 0, "pending": 0}}


def test_start_search_sends_probe_and_cleans_up_on_timeout():
    srv = _Server([{"status": "PROCESSING"}])
    with pytest.raises(ExtractionError, match="timed out"):
        _client(srv).start_search({"main_dataset_name": "X"}, 300)
    assert srv.calls[0][2]["probe"] is True and srv.calls[0][2]["probe_pull_cap"] == 300
    assert srv.calls[-1] == ("POST", "/pipelines/cleanup", ["T1"])


class _PoolServer:
    """Stateful fake: a probe pool whose full-text outcomes appear after an analyze.

    *outcomes* maps publication id -> True (delivered) / False (miss) / "budget".
    Outcomes of a batch become visible only after *lag* llm-results reads, so a
    batch-done check that trusts stale counters would return too early.
    """

    def __init__(self, outcomes, lag=2, fail_analyze_on=None, enqueue_lag=None,
                 analyze_on_status=False):
        self.outcomes = dict(outcomes)
        self.order = list(outcomes)  # citation order
        self.lag = lag
        self.fail_analyze_on = fail_analyze_on
        # pid -> llm-results reads before a delivered paper is queued for the LLM
        # (ledger row written first, enqueue later, as extraction's fetch worker does).
        self.enqueue_lag = dict(enqueue_lag or {})
        # When set, a queued paper reads as analyzed only after the next /status call
        # (its analysis finishes between a client's llm-results and status reads).
        self.analyze_on_status = analyze_on_status
        self.analyzed: set[str] = set()
        self.pending: dict[str, int] = {}
        self.done: set[str] = set()
        self.batches: list[list[str]] = []
        self.calls: list[tuple[str, str, object]] = []
        self.context_reads = 0

    def _queued(self, pid):
        return pid in self.done and self.outcomes[pid] is True and self.enqueue_lag.get(pid, 0) <= 0

    def _tick(self):
        for pid in list(self.enqueue_lag):
            if pid in self.done:
                self.enqueue_lag[pid] -= 1
        for pid in list(self.pending):
            self.pending[pid] -= 1
            if self.pending[pid] <= 0:
                del self.pending[pid]
                self.done.add(pid)

    def _pub(self, i, pid):
        if pid not in self.done:
            ft = {"status": "missing", "source": None, "reason": "not_attempted"}
        elif self.outcomes[pid] is True:
            ft = {"status": "delivered", "source": "pmc_s3", "reason": None}
        else:
            reason = "budget_exhausted" if self.outcomes[pid] == "budget" else "not_found"
            ft = {"status": "missing", "source": None, "reason": reason}
        delivered = ft["status"] == "delivered"
        return {"publication_id": pid, "citation_count": 1000 - i, "fulltext": ft,
                "analyzed": (pid in self.analyzed) if self.analyze_on_status else self._queued(pid), "validation_failed": False,
                "validated_datasets": [{"name": "X", "is_used": True, "is_mentioned": True}] if delivered else []}

    def __call__(self, req):
        body = json.loads(req.content) if req.content else None
        self.calls.append((req.method, req.url.path, body))
        p = req.url.path
        if p == "/pipelines/start":
            return httpx.Response(202, json={"task_id": "T1"})
        if p == "/pipelines/T1/status":
            self.analyzed.update(pid for pid in self.done if self._queued(pid))
            delivered = sum(1 for pid in self.done if self.outcomes[pid] is True)
            queued = sum(1 for pid in self.done if self._queued(pid))
            # Mimic extraction: `selected` is the LAST batch size, `attempted` accumulates,
            # `total_queued` is cumulative over the run's selective requests.
            last = len(self.batches[-1]) if self.batches else 0
            return httpx.Response(200, json={
                "status": "DONE",
                "fulltext": {"selected": last, "attempted": len(self.done), "delivered": delivered},
                "llm_analysis": {"status": "completed", "total_queued": queued, "pending": 0},
            })
        if p == "/pipelines/T1/full-publications/analyze":
            if self.fail_analyze_on == len(self.batches) + 1:
                return httpx.Response(500, json={"detail": "boom"})
            ids = body["publication_ids"]
            self.batches.append(ids)
            for pid in ids:
                self.pending[pid] = self.lag
            return httpx.Response(202, json={"status": "started"})
        if p == "/pipelines/T1/llm-results":
            if req.url.params.get("contexts") == "true":
                self.context_reads += 1
            self._tick()
            return httpx.Response(200, json={"task_id": "T1", "probe": True, "publications": [
                self._pub(i, pid) for i, pid in enumerate(self.order)]})
        if p == "/pipelines/cleanup":
            return httpx.Response(202, json=[])
        return httpx.Response(404)


def _topup(srv, target=3, batch_size=3, max_batches=8):
    return _client(srv).run_probe_topup({"main_dataset_name": "X"}, target=target,
                                        batch_size=batch_size, max_batches=max_batches, pull_cap=300)


def test_topup_target_met_in_first_batch():
    srv = _PoolServer({f"p{i}": True for i in range(6)})
    out = _topup(srv)
    assert (out.batches, out.delivered, out.stop_reason) == (1, 3, "target_met")
    assert srv.batches == [["p0", "p1", "p2"]]
    analyze = next(c for c in srv.calls if c[1].endswith("/analyze"))
    assert analyze[2] == {"fulltext_source": "open", "publication_ids": ["p0", "p1", "p2"],
                          "steps": {"validate_datasets": True}}
    assert [c for c in srv.calls if c[1] == "/pipelines/cleanup"] == [("POST", "/pipelines/cleanup", ["T1"])]


def test_topup_shortfall_sends_next_untried_batch():
    outcomes = {"p0": False, "p1": True, "p2": False, "p3": True, "p4": False, "p5": True, "p6": True}
    srv = _PoolServer(outcomes)
    out = _topup(srv)
    assert srv.batches == [["p0", "p1", "p2"], ["p3", "p4", "p5"]]
    assert (out.batches, out.attempted, out.delivered, out.stop_reason) == (2, 6, 3, "target_met")
    assert out.results["publications"][1]["fulltext"]["status"] == "delivered"


def test_topup_pool_exhausted():
    srv = _PoolServer({"p0": False, "p1": True, "p2": False, "p3": False})
    out = _topup(srv)
    assert srv.batches == [["p0", "p1", "p2"], ["p3"]]
    assert (out.delivered, out.attempted, out.stop_reason) == (1, 4, "pool_exhausted")


def test_topup_max_batches():
    srv = _PoolServer({f"p{i}": False for i in range(20)})
    out = _topup(srv, max_batches=2)
    assert len(srv.batches) == 2 and (out.stop_reason, out.attempted) == ("max_batches", 6)


def test_topup_budget_miss_is_reported_but_does_not_stop():
    srv = _PoolServer({"p0": "budget", "p1": True, "p2": "budget", "p3": True, "p4": True})
    out = _topup(srv)
    assert out.stop_reason == "target_met" and out.budget_exhausted is True
    assert len(srv.batches) == 2


def test_batch_two_not_done_while_its_ids_are_not_attempted():
    # After batch 2 is sent, /status already says attempted(3) >= selected(3) and the
    # LLM is completed; only the per-id outcomes tell that batch 2 is still running.
    srv = _PoolServer({f"p{i}": (i >= 3) for i in range(6)}, lag=4)
    out = _topup(srv)
    assert srv.batches == [["p0", "p1", "p2"], ["p3", "p4", "p5"]]
    assert out.delivered == 3 and out.stop_reason == "target_met"


def test_topup_cleans_up_once_when_batch_two_fails():
    srv = _PoolServer({f"p{i}": False for i in range(6)}, fail_analyze_on=2)
    with pytest.raises(ExtractionError):
        _topup(srv)
    assert [c[1] for c in srv.calls].count("/pipelines/cleanup") == 1


def test_topup_batch_timeout_raises_and_cleans_up():
    srv = _PoolServer({"p0": True}, lag=10_000)
    with pytest.raises(ExtractionError, match="batch timed out"):
        _topup(srv, target=1)
    assert srv.calls[-1][1] == "/pipelines/cleanup"


def test_parse_status_no_fulltext_delivered_is_llm_done():
    st = parse_status({"status": "DONE", "fulltext": {"selected": 5, "attempted": 5, "delivered": 0}})
    assert st.search_done and st.llm_done


def test_submit_forces_probe_false():
    srv = _Server([{"status": "DONE"}])
    out = _client(srv).submit({"main_dataset_name": "X", "probe": True})
    assert out == {"task_id": "T1"}
    assert srv.calls[0][2]["probe"] is False


def test_parse_status_fetch_running_llm_completed_is_not_done():
    st = parse_status({
        "status": "DONE",
        "llm_analysis": {"status": "completed", "pending": 0},
        "fulltext": {"selected": 30, "attempted": 5, "delivered": 5},
    })
    assert not st.llm_done


def test_parse_status_fetch_finished_llm_completed_is_done():
    st = parse_status({
        "status": "DONE",
        "llm_analysis": {"status": "completed", "total_queued": 12, "pending": 0},
        "fulltext": {"selected": 30, "attempted": 30, "delivered": 12},
    })
    assert st.llm_done


def test_parse_status_total_queued_below_delivered_is_not_done():
    st = parse_status({
        "status": "DONE",
        "llm_analysis": {"status": "completed", "total_queued": 3, "analyzed": 3, "failed": 0, "pending": 0},
        "fulltext": {"selected": 30, "attempted": 30, "delivered": 12},
    })
    assert not st.llm_done


def test_parse_status_total_queued_covers_delivered_is_done():
    st = parse_status({
        "status": "DONE",
        "llm_analysis": {"status": "completed", "total_queued": 12, "analyzed": 11, "failed": 1, "pending": 0},
        "fulltext": {"selected": 30, "attempted": 30, "delivered": 12},
    })
    assert st.llm_done


def test_dev_url_has_no_default_so_a_probe_fails_clearly(tmp_path, monkeypatch):
    import pytest

    from dataset_agent.adapters.extraction_client import ExtractionClient, ExtractionError
    from dataset_agent.settings import Settings

    monkeypatch.delenv("DATASET_AGENT_EXTRACTION_DEV_URL", raising=False)
    s = Settings(tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out", _env_file=None)
    assert s.extraction_dev_url == ""
    with pytest.raises(ExtractionError, match="not configured"):
        ExtractionClient.for_target(s, "dev")


def test_batch_two_waits_for_a_delivered_paper_still_being_enqueued():
    # p4 is delivered in the ledger but not yet queued for the LLM; batch 1's queued
    # papers alone would satisfy a batch-local total_queued check.
    outcomes = {"p0": True, "p1": True, "p2": False, "p3": False, "p4": True, "p5": False}
    srv = _PoolServer(outcomes, enqueue_lag={"p4": 3})
    out = _topup(srv)
    assert out.stop_reason == "target_met" and out.delivered == 3
    p4 = next(p for p in out.results["publications"] if p["publication_id"] == "p4")
    assert p4["analyzed"] is True


def test_batch_result_is_read_after_the_completing_status():
    srv = _PoolServer({"p0": True}, analyze_on_status=True)
    out = _topup(srv, target=1)
    assert out.stop_reason == "target_met"
    assert out.results["publications"][0]["analyzed"] is True


def test_final_read_asks_for_term_contexts_once():
    srv = _PoolServer({"p0": False, "p1": True, "p2": True, "p3": True})
    _topup(srv)
    assert srv.context_reads == 1

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


def test_run_probe_happy_path_and_cleanup():
    srv = _Server([{"status": "PROCESSING"}, {"status": "DONE"}, DONE_LLM])
    res = _client(srv).run_probe({"main_dataset_name": "X"}, 30, 200)
    assert res["probe"] is True
    paths = [c[1] for c in srv.calls]
    assert paths[0] == "/pipelines/start" and srv.calls[0][2]["probe"] is True
    assert srv.calls[0][2]["probe_pull_cap"] == 200
    analyze = next(c for c in srv.calls if c[1].endswith("/analyze"))
    assert analyze[2] == {"fulltext_source": "open", "top_n": 30, "steps": {"validate_datasets": True}}
    assert srv.calls[-1] == ("POST", "/pipelines/cleanup", ["T1"])


def test_cleanup_on_analyze_failure():
    srv = _Server([{"status": "DONE"}], fail_analyze=True)
    with pytest.raises(ExtractionError):
        _client(srv).start_probe({}, 30, 200)
    assert srv.calls[-1][1] == "/pipelines/cleanup"


def test_poll_timeout_cleans_up():
    srv = _Server([{"status": "DONE"}, {"status": "DONE", "llm_analysis": {"status": "in_progress", "pending": 5}}])
    with pytest.raises(ExtractionError, match="timed out"):
        _client(srv).run_probe({}, 30, 200)
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

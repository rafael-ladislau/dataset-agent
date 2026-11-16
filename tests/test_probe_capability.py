"""Fail closed before any probe: the extraction target must declare probe support.

On extraction main, POST /pipelines/start accepts a body with ``probe: true`` and
ignores the field, so a "probe" there starts a real pipeline with real analyses
and write-backs. The client therefore reads the server's /openapi.json and probes
only when the start route's request schema declares ``probe`` and the cleanup and
llm-results routes exist; otherwise it refuses before posting anything.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import httpx
import pytest

from dataset_agent.adapters.extraction_client import ExtractionClient, ProbeUnsupportedError
from tests.test_extraction_client import _openapi
from tests.test_split_release_bar import _pubs, _run


class _MainServer:
    """Extraction main: no probe in the schema; start would run a real pipeline."""

    def __init__(self, openapi=None, status=200):
        self.calls: list[tuple[str, str]] = []
        self.openapi = _openapi(False) if openapi is None else openapi
        self.status = status

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append((req.method, req.url.path))
        if req.url.path == "/openapi.json":
            return httpx.Response(self.status, json=self.openapi)
        if req.url.path in ("/pipelines/start", "/ai-models/start"):
            return httpx.Response(202, json={"task_id": "REAL"})
        return httpx.Response(404)

    def started(self):
        return [c for c in self.calls if c[1].endswith("/start")]


def _client(server):
    return ExtractionClient("http://x", "u", "p", transport=httpx.MockTransport(server),
                            sleep=lambda d: None, poll_timeout=5, poll_interval=1)


@pytest.mark.parametrize("server", [
    _MainServer(),  # main: start schema without probe
    _MainServer(status=404),  # no schema served
    _MainServer(openapi={"paths": {}}),  # schema without the routes
])
def test_a_target_without_declared_probe_support_is_never_probed(server):
    with pytest.raises(ProbeUnsupportedError, match="does not declare probe support"):
        _client(server).start_search({"main_dataset_name": "X"}, 10)
    assert server.started() == []


def test_a_target_that_declares_probe_support_is_probed():
    server = _MainServer(openapi=_openapi(True))
    client = _client(server)
    with pytest.raises(Exception):  # the fake never finishes the search; only the POST matters
        client.start_search({"main_dataset_name": "X"}, 10)
    assert server.started() == [("POST", "/pipelines/start")]


def test_the_capability_is_read_at_every_probe_start():
    server = _MainServer()
    client = _client(server)
    for _ in range(2):
        with pytest.raises(ProbeUnsupportedError):
            client.start_search({"main_dataset_name": "X"}, 10, entity="ai_model")
    assert [c for c in server.calls if c[1] == "/openapi.json"] == [("GET", "/openapi.json")] * 2


def test_the_loop_against_a_target_without_probe_support_refuses_and_stays_low(tmp_path):
    server = _MainServer()
    out = _run(tmp_path, _pubs(55, 20, 15, 10), split=True, client=_client(server))
    assert server.started() == []
    assert out.confidence.value == "low"
    assert "does not declare probe support" in out.notes


def test_an_ai_model_probe_against_a_target_without_probe_support_is_refused(tmp_path):
    from dataset_agent.application.query_optimization import QueryOptimizationUseCase
    from tests.test_fulltext_loop import CountDSL, make_settings

    server = _MainServer()
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=make_settings(tmp_path),
                                  extraction_client=_client(server))
    model, out, notes = MagicMock(), MagicMock(ai_model_seed={"main_dataset_name": "M"}), []
    assert asyncio.run(uc._measure_seed(model, out, notes, 0)) is None
    assert server.started() == []
    assert any("does not declare probe support" in n for n in notes)


def test_loop_off_a_split_dataset_is_not_released(tmp_path):
    out = _run(tmp_path, _pubs(55, 20, 15, 10), split=True,
               fulltext_refinement_enabled=False)
    assert out.confidence.value == "low"
    assert "full-text bar was not evaluated" in out.notes


def test_loop_off_a_non_split_dataset_is_unchanged(tmp_path):
    out = _run(tmp_path, _pubs(85, 0, 0, 15), split=False, fulltext_refinement_enabled=False)
    assert "full-text bar was not evaluated" not in out.notes


def test_an_ai_model_probe_without_an_extraction_target_is_a_note_not_a_failure(tmp_path):
    from dataset_agent.application.query_optimization import QueryOptimizationUseCase
    from tests.test_fulltext_loop import CountDSL, make_settings

    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=make_settings(tmp_path, extraction_dev_url=""))
    model, out, notes = MagicMock(), MagicMock(ai_model_seed={"main_dataset_name": "M"}), []
    assert asyncio.run(uc._measure_seed(model, out, notes, 0)) is None
    assert any("extraction_dev_url is not configured" in n for n in notes)


def test_the_capability_is_read_again_before_every_probe():
    # After an extraction rollback, a cached "supported" must not let a probe start.
    server = _MainServer(openapi=_openapi(True))
    client = _client(server)
    with pytest.raises(Exception):
        client.start_search({"main_dataset_name": "X"}, 10)
    server.openapi = _openapi(False)  # the target was rolled back to main
    with pytest.raises(ProbeUnsupportedError):
        client.start_search({"main_dataset_name": "X"}, 10)
    assert server.started() == [("POST", "/pipelines/start")]


def test_a_seed_field_the_target_does_not_declare_is_left_out_of_the_probe_start():
    """Extraction's start schemas forbid unknown fields: "developer" travels only when
    the target's /openapi.json declares it."""
    import json as _json

    bodies = []

    class _Recorder(_MainServer):
        def __call__(self, req):
            if req.url.path.endswith("/start"):
                bodies.append(_json.loads(req.content))
            return super().__call__(req)

    seed = {"main_dataset_name": "X", "developer": "Microsoft"}
    for spec in (_openapi(True), _with_developer(_openapi(True))):
        client = _client(_Recorder(openapi=spec))
        with pytest.raises(Exception):  # the fake never finishes the search
            client.start_search(dict(seed), 10, entity="ai_model")
    assert "developer" not in bodies[0]
    assert bodies[1]["developer"] == "Microsoft"


def _with_developer(spec):
    spec["components"]["schemas"]["S"]["properties"]["developer"] = {"type": "string"}
    return spec

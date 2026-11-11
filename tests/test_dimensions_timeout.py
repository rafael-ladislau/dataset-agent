"""dimcli HTTP calls get a (connect, read) timeout; stalls are retried then fail."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import requests

from dataset_agent.adapters import dimensions_dsl as dd
from dataset_agent.settings import Settings


class _FakeRequests:
    """Stands in for ``requests``; records kwargs and can raise Timeout."""

    exceptions = requests.exceptions

    def __init__(self, fail_times: int = 0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_times = fail_times

    def post(self, url: str, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if len(self.calls) <= self.fail_times:
            raise requests.exceptions.ReadTimeout("read timed out")
        return "ok"


@pytest.fixture
def fake_dimcli(monkeypatch: pytest.MonkeyPatch) -> dict[str, types.ModuleType]:
    fake = _FakeRequests()
    pkg = types.ModuleType("dimcli")
    core = types.ModuleType("dimcli.core")
    api = types.ModuleType("dimcli.core.api")
    auth = types.ModuleType("dimcli.core.auth")
    api.requests = fake  # type: ignore[attr-defined]
    auth.requests = fake  # type: ignore[attr-defined]
    pkg.login = lambda **kw: None  # type: ignore[attr-defined]
    pkg.Dsl = lambda: object()  # type: ignore[attr-defined]
    pkg.core = core  # type: ignore[attr-defined]
    for name, mod in {
        "dimcli": pkg,
        "dimcli.core": core,
        "dimcli.core.api": api,
        "dimcli.core.auth": auth,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return {"api": api, "auth": auth, "fake": fake}  # type: ignore[dict-item]


def test_default_settings() -> None:
    s = Settings()
    assert s.dimensions_timeout_seconds == 180.0
    assert s.dimensions_connect_timeout_seconds == 15.0


def test_install_adds_timeout_to_api_and_auth(fake_dimcli: Any) -> None:
    s = Settings(dimensions_timeout_seconds=7, dimensions_connect_timeout_seconds=2)
    assert dd.install_dimcli_timeout(s) == (2.0, 7.0)
    fake_dimcli["api"].requests.post("http://x", data=b"q")
    fake_dimcli["auth"].requests.post("http://x", json={})
    assert [c["timeout"] for c in fake_dimcli["fake"].calls] == [(2.0, 7.0), (2.0, 7.0)]


def test_install_is_idempotent_and_respects_explicit_timeout(fake_dimcli: Any) -> None:
    s = Settings()
    dd.install_dimcli_timeout(s)
    proxy = fake_dimcli["api"].requests
    dd.install_dimcli_timeout(Settings(dimensions_timeout_seconds=9))
    assert fake_dimcli["api"].requests is proxy
    assert not isinstance(proxy._real, dd._TimeoutRequests)
    proxy.post("http://x", timeout=3)
    proxy.post("http://x")
    assert fake_dimcli["fake"].calls[0]["timeout"] == 3
    assert fake_dimcli["fake"].calls[1]["timeout"] == (15.0, 9.0)


def test_stalled_call_is_retried_then_raises(fake_dimcli: Any) -> None:
    dd.install_dimcli_timeout(Settings())
    proxy = fake_dimcli["api"].requests
    fake_dimcli["fake"].fail_times = 99
    sleeps: list[float] = []

    with pytest.raises(requests.exceptions.ReadTimeout):
        dd.run_dsl_with_transient_retry(
            lambda q: proxy.post("http://x", data=q), "search", sleep=sleeps.append
        )
    assert len(fake_dimcli["fake"].calls) == 3  # 1 + 2 retries
    assert all(c["timeout"] == (15.0, 180.0) for c in fake_dimcli["fake"].calls)
    assert len(sleeps) == 2


@pytest.mark.asyncio
async def test_async_client_retries_timeout_then_dsl_error(
    fake_dimcli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_dimcli["fake"]
    fake.fail_times = 99
    sys.modules["dimcli"].Dsl = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        query=lambda q: fake_dimcli["api"].requests.post("http://x", data=q)
    )
    s = Settings(dimensions_api_key="k", dimensions_rate_limit_seconds=0)
    client = dd.AsyncThrottledDimensionsDsl(s, transient_backoff=(0.0,))
    with pytest.raises(dd.DimensionsDslError):
        await client.execute_dsl("search publications return publications")
    assert len(fake.calls) == 3
    assert all("timeout" in c for c in fake.calls)

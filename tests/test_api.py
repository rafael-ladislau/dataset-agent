"""API tests with mocked use case."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from dataset_agent.domain.models import (
    DatasetRecord,
    Group,
    LiteratureValidation,
    ResearchRequest,
    YearsRange,
)
from dataset_agent.interfaces import api
from dataset_agent.settings import Settings


def _fake_settings(tmp_path: Path) -> Settings:
    return Settings(
        tasks_db=tmp_path / "tasks.db",
        output_dir=tmp_path / "out",
    )


def _fake_build_use_case(settings: Settings):
    """Return use case that writes a minimal JSON result without LLM."""

    class _UC:
        def execute(self, req: ResearchRequest) -> tuple[DatasetRecord, Path]:
            record = DatasetRecord(
                engine="dimensions",
                group=Group(name="default"),
                main_dataset_name=req.dataset_name,
                home_url=None,
                description="test",
                years_range=YearsRange(start_year=2020, end_year=2021),
                webhook_url=req.webhook_url,
            )
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            path = settings.output_dir / f"{req.dataset_name}.json"
            path.write_text(record.model_dump_json(), encoding="utf-8")
            return record, path

    return _UC()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "build_use_case", _fake_build_use_case)

    def _settings() -> Settings:
        return _fake_settings(tmp_path)

    api.app.dependency_overrides[api.get_settings] = _settings
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health_no_auth(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_tasks_no_auth(client: TestClient) -> None:
    r = client.get("/tasks")
    assert r.status_code == 200


def test_create_task_and_fetch_result(client: TestClient) -> None:
    r = client.post("/tasks", json={"dataset_name": "DS1"})
    assert r.status_code == 200
    tid = r.json()["id"]

    st = client.get(f"/tasks/{tid}")
    assert st.status_code == 200
    assert st.json()["status"] == "completed"

    res = client.get(f"/tasks/{tid}/result")
    assert res.status_code == 200
    body = res.json()
    assert body["main_dataset_name"] == "DS1"


def test_webhook_called_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "build_use_case", _fake_build_use_case)

    def _settings() -> Settings:
        return _fake_settings(tmp_path)

    api.app.dependency_overrides[api.get_settings] = _settings
    with patch.object(api.httpx, "post") as post_mock:
        c = TestClient(api.app)
        c.post(
            "/tasks",
            json={"dataset_name": "WH", "webhook_url": "https://hook.example/x"},
        )
        post_mock.assert_called_once()
        args, kwargs = post_mock.call_args
        assert str(args[0]) == "https://hook.example/x"
        assert "json" in kwargs
    api.app.dependency_overrides.clear()


def test_task_completes_with_failed_literature_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task completes even when literature validation fails, with validation info in result."""

    class _UCWithFailedValidation:
        def execute(self, req: ResearchRequest) -> tuple[DatasetRecord, Path]:
            out = tmp_path / "out"
            out.mkdir(exist_ok=True)
            record = DatasetRecord(
                engine="dimensions",
                group=Group(name="g"),
                main_dataset_name=req.dataset_name,
                description="desc",
                years_range=YearsRange(start_year=2020, end_year=2025),
                literature_validation=LiteratureValidation(
                    passed=False,
                    indicator=0.3,
                    attempts=3,
                    detail={"publications_found": 10, "publications_valid": 3},
                ),
            )
            p = out / "result.json"
            p.write_text(record.model_dump_json(indent=2))
            return record, p

    monkeypatch.setattr(api, "build_use_case", lambda _s: _UCWithFailedValidation())

    def _settings() -> Settings:
        return _fake_settings(tmp_path)

    api.app.dependency_overrides[api.get_settings] = _settings
    c = TestClient(api.app)
    r = c.post("/tasks", json={"dataset_name": "X"})
    tid = r.json()["id"]
    st = c.get(f"/tasks/{tid}")
    assert st.json()["status"] == "completed"

    res = c.get(f"/tasks/{tid}/result")
    assert res.status_code == 200
    body = res.json()
    assert body["literature_validation"]["passed"] is False
    assert body["literature_validation"]["indicator"] == 0.3
    assert body["literature_validation"]["attempts"] == 3
    api.app.dependency_overrides.clear()
    api.app.dependency_overrides.clear()


def test_validate_passes_llm_batch_size_to_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, int] = {}

    class _FakeGate:
        def assess(
            self,
            record: DatasetRecord,
            sample_size: int = 10,
            llm_batch_size: int = 25,
        ):
            _ = record
            captured["sample_size"] = sample_size
            captured["llm_batch_size"] = llm_batch_size
            return types.SimpleNamespace(
                passed=True,
                indicator=0.8,
                detail={"publications_total": 123, "validation_details": []},
            )

    fake_literature = types.ModuleType("dataset_agent.adapters.literature")
    fake_literature.literature_gate_from_settings = lambda settings, agent=None: _FakeGate()
    fake_literature.evaluate_terms_with_llm = lambda **kwargs: {
        "is_effective": True,
        "dataset_names_score": 8,
        "flag_terms_score": 8,
        "issues": [],
        "suggested_dataset_names": [],
        "suggested_flag_terms": [],
        "reasoning": "ok",
    }

    fake_agent_module = types.ModuleType("dataset_agent.adapters.agent_langchain")

    class _FakeAgent:
        def __init__(self, **kwargs):
            _ = kwargs

    fake_agent_module.LangChainAgent = _FakeAgent
    monkeypatch.setitem(sys.modules, "dataset_agent.adapters.literature", fake_literature)
    monkeypatch.setitem(sys.modules, "dataset_agent.adapters.agent_langchain", fake_agent_module)

    def _settings() -> Settings:
        return _fake_settings(tmp_path)

    api.app.dependency_overrides[api.get_settings] = _settings
    client = TestClient(api.app)

    response = client.post(
        "/validate",
        json={
            "main_dataset_name": "Dataset X",
            "dataset_names": ["Dataset X"],
            "flag_terms": ["Org Y"],
            "sample_size": 1000,
            "llm_batch_size": 77,
        },
    )
    assert response.status_code == 200
    assert captured["sample_size"] == 1000
    assert captured["llm_batch_size"] == 77
    api.app.dependency_overrides.clear()

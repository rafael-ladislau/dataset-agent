"""Provider / API-style resolution, adapter selection and the /health LLM probe."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from dataset_agent.adapters.agent_anthropic import AnthropicAgent
from dataset_agent.adapters.agent_openai import OpenAICompatibleAgent
from dataset_agent.bootstrap import _build_agent
from dataset_agent.interfaces import api
from dataset_agent.settings import Settings

_LLM_ENV = (
    "LMSTUDIO_BASE_URL",
    "LMSTUDIO_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENLLM_BASE_URL",
    "OPENLLM_API_KEY",
)
_PREFIXED = (
    "URL",  # main's alias of DATASET_AGENT_OLLAMA_BASE_URL
    "LLM_PROVIDER",
    "LLM_API_STYLE",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LMSTUDIO_BASE_URL",
    "LMSTUDIO_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
    "OPENROUTER_API_KEY",
    "OPENLLM_BASE_URL",
    "OPENLLM_MODEL",
    "OPENLLM_API_KEY",
    "OPENAI_COMPATIBLE_BASE_URL",
    "OPENAI_COMPATIBLE_MODEL",
    "OPENAI_COMPATIBLE_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in _PREFIXED:
        monkeypatch.delenv(f"DATASET_AGENT_{name}", raising=False)


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_lmstudio_defaults_to_openai_style():
    endpoint = _settings().llm_endpoint()
    assert endpoint.provider == "lmstudio"
    assert endpoint.api_style == "openai"
    assert endpoint.base_url == "http://127.0.0.1:1234/v1"
    assert endpoint.model == "gemma-4-31b-it-mlx"
    assert endpoint.api_key == "local"
    assert isinstance(_build_agent(_settings()), OpenAICompatibleAgent)


def test_lmstudio_anthropic_style_still_selectable(monkeypatch):
    monkeypatch.setenv("DATASET_AGENT_LLM_API_STYLE", "anthropic")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://gpu:1234/v1")
    settings = _settings()
    endpoint = settings.llm_endpoint()
    assert endpoint.api_style == "anthropic"
    assert endpoint.base_url == "http://gpu:1234"
    agent = _build_agent(settings)
    assert isinstance(agent, AnthropicAgent)
    assert agent._base_url == "http://gpu:1234"


def test_legacy_lmstudio_env_vars_feed_openai_endpoint(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://gpu:1234/v1/")
    monkeypatch.setenv("DATASET_AGENT_LLM_MODEL", "qwen3.6-35b-a3b")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "secret")
    endpoint = _settings().llm_endpoint()
    assert endpoint.base_url == "http://gpu:1234/v1"
    assert endpoint.model == "qwen3.6-35b-a3b"
    assert endpoint.api_key == "secret"


def test_ollama_keeps_anthropic_style(monkeypatch):
    monkeypatch.setenv("DATASET_AGENT_LLM_PROVIDER", "ollama")
    settings = _settings()
    endpoint = settings.llm_endpoint()
    assert (endpoint.api_style, endpoint.base_url, endpoint.model) == (
        "anthropic",
        "http://127.0.0.1:11434",
        "gemma4-31b",
    )
    assert isinstance(_build_agent(settings), AnthropicAgent)
    monkeypatch.setenv("DATASET_AGENT_LLM_API_STYLE", "openai")
    assert _settings().llm_endpoint().base_url == "http://127.0.0.1:11434/v1"


def test_openrouter_keeps_anthropic_style_and_key_fallback(monkeypatch):
    monkeypatch.setenv("DATASET_AGENT_LLM_PROVIDER", "openrouter")
    endpoint = _settings().llm_endpoint()
    assert endpoint.api_style == "anthropic"
    assert endpoint.base_url == "https://openrouter.ai/api"
    assert endpoint.api_key == "local"
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("DATASET_AGENT_LLM_API_STYLE", "openai")
    endpoint = _settings().llm_endpoint()
    assert endpoint.base_url == "https://openrouter.ai/api/v1"
    assert endpoint.api_key == "or-key"


def test_openllm_provider(monkeypatch):
    monkeypatch.setenv("DATASET_AGENT_LLM_PROVIDER", "openllm")
    with pytest.raises(ValueError, match="DATASET_AGENT_OPENLLM_MODEL"):
        _settings().llm_endpoint()
    monkeypatch.setenv("DATASET_AGENT_OPENLLM_MODEL", "gpt-oss-120b")
    monkeypatch.setenv("OPENLLM_BASE_URL", "http://openllm:3000/v1/")
    monkeypatch.setenv("OPENLLM_API_KEY", "ol-key")
    settings = _settings()
    endpoint = settings.llm_endpoint()
    assert (endpoint.api_style, endpoint.base_url, endpoint.model, endpoint.api_key) == (
        "openai",
        "http://openllm:3000/v1",
        "gpt-oss-120b",
        "ol-key",
    )
    agent = _build_agent(settings)
    assert isinstance(agent, OpenAICompatibleAgent)
    assert agent._base_url == "http://openllm:3000/v1"


def test_generic_openai_compatible_provider(monkeypatch):
    monkeypatch.setenv("DATASET_AGENT_LLM_PROVIDER", "openai_compatible")
    with pytest.raises(ValueError, match="DATASET_AGENT_OPENAI_COMPATIBLE_BASE_URL"):
        _settings().llm_endpoint()
    monkeypatch.setenv("DATASET_AGENT_OPENAI_COMPATIBLE_BASE_URL", "https://llm.example/api/v1")
    monkeypatch.setenv("DATASET_AGENT_OPENAI_COMPATIBLE_MODEL", "qwen3.8-flash-next")
    endpoint = _settings().llm_endpoint()
    assert endpoint.base_url == "https://llm.example/api/v1"
    assert endpoint.model == "qwen3.8-flash-next"
    assert endpoint.api_key == "local"


def test_empty_api_style_means_provider_default(monkeypatch):
    monkeypatch.setenv("DATASET_AGENT_LLM_API_STYLE", "")
    assert _settings().llm_endpoint().api_style == "openai"
    monkeypatch.setenv("DATASET_AGENT_LLM_PROVIDER", "ollama")
    assert _settings().llm_endpoint().api_style == "anthropic"


def test_env_example_works_for_every_provider_with_defaults():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    for provider in ("lmstudio", "ollama", "openrouter"):
        Settings(_env_file=env_example, llm_provider=provider).llm_endpoint()


def test_invalid_api_style_rejected(monkeypatch):
    monkeypatch.setenv("DATASET_AGENT_LLM_API_STYLE", "grpc")
    with pytest.raises(ValueError):
        _settings()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.fixture
def probe_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _fake_get(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers or {}})
        request = httpx.Request("GET", url)
        if url.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [{"name": "a"}]}, request=request)
        return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]}, request=request)

    monkeypatch.setattr(api.httpx, "get", _fake_get)
    return calls


def _health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    settings = _settings(tasks_db=tmp_path / "tasks.db", output_dir=tmp_path / "out")
    api.app.dependency_overrides[api.get_settings] = lambda: settings
    try:
        response = TestClient(api.app).get("/health")
    finally:
        api.app.dependency_overrides.clear()
    assert response.status_code == 200
    return response.json()


def test_health_probes_openai_models_endpoint(tmp_path, monkeypatch, probe_calls):
    body = _health(tmp_path, monkeypatch)
    assert body["status"] == "ok"
    assert probe_calls[0]["url"] == "http://127.0.0.1:1234/v1/models"
    assert probe_calls[0]["headers"] == {"Authorization": "Bearer local"}
    assert body["llm"]["ok"] is True
    assert body["llm"]["api_style"] == "openai"
    assert body["llm"]["models_available"] == 2
    assert body["ollama"]["checked"] is False


def test_health_probes_ollama_tags(tmp_path, monkeypatch, probe_calls):
    monkeypatch.setenv("DATASET_AGENT_LLM_PROVIDER", "ollama")
    body = _health(tmp_path, monkeypatch)
    assert probe_calls[0]["url"] == "http://127.0.0.1:11434/api/tags"
    assert body["llm"]["ok"] is True
    assert body["ollama"] == body["llm"]


def test_health_reports_unreachable_provider(tmp_path, monkeypatch):
    def _refuse(url, headers=None, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(api.httpx, "get", _refuse)
    body = _health(tmp_path, monkeypatch)
    assert body["status"] == "ok"
    assert body["llm"]["ok"] is False
    assert "refused" in body["llm"]["error"]


def test_health_reports_misconfigured_provider(tmp_path, monkeypatch, probe_calls):
    monkeypatch.setenv("DATASET_AGENT_LLM_PROVIDER", "openllm")
    body = _health(tmp_path, monkeypatch)
    assert body["llm"]["ok"] is False
    assert "OPENLLM_MODEL" in body["llm"]["error"]
    assert probe_calls == []

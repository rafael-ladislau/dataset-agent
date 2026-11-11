"""Research pipeline multilingual alias expansion."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dataset_agent.application import research as research_mod
from dataset_agent.application.research import DatasetResearchUseCase
from dataset_agent.domain.models import ResearchRequest
from dataset_agent.settings import Settings


def _minimal_extractor() -> MagicMock:
    ext = MagicMock()
    ext.extract_sections.return_value = {
        "DESCRIPTION": "A national health survey.",
        "HOME_URL": "",
        "DATA_URL": "",
        "SCHEMA_URL": "",
        "DOCUMENTATION_URL": "",
        "ACCESS_TYPE": "Open",
    }
    ext.extract_url.return_value = None
    ext.extract_list.return_value = ["Current Population Survey"]
    return ext


def test_multilingual_aliases_appended_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    ml_calls: list[str] = []

    def _fake_ml(name: str, agent: object) -> list[str]:
        _ = agent
        ml_calls.append(name)
        return ["Encuesta Nacional"]

    monkeypatch.setattr(research_mod, "generate_multilingual_aliases", _fake_ml)
    monkeypatch.setattr(
        research_mod,
        "refine_flag_terms_with_llm",
        lambda _agent, **kw: kw["flag_terms"],
    )
    monkeypatch.setattr(
        research_mod,
        "refine_dataset_names_with_llm",
        lambda _agent, **kw: kw["dataset_names"],
    )
    monkeypatch.setattr(
        research_mod,
        "validate_no_flag_alias_overlap",
        lambda aliases, flags: (aliases, flags, []),
    )
    monkeypatch.setattr(
        research_mod,
        "detect_subdataset_aliases",
        lambda _agent, _main, aliases: (aliases, []),
    )

    agent = MagicMock()
    agent.get_information.return_value = "stub"

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        research_multilingual_aliases=True,
    )
    uc = DatasetResearchUseCase(
        agent=agent,
        extractor=_minimal_extractor(),
        repository=MagicMock(),
        settings=settings,
        literature_gate=None,
    )
    record = uc._run_pipeline_once(ResearchRequest(dataset_name="Current Population Survey"))
    assert ml_calls == ["Current Population Survey"]
    assert any("encuesta" in n.lower() for n in record.dataset_names)


def test_multilingual_aliases_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        research_mod,
        "generate_multilingual_aliases",
        lambda *_a, **_k: pytest.fail("should not run when disabled"),
    )
    monkeypatch.setattr(
        research_mod,
        "refine_flag_terms_with_llm",
        lambda _agent, **kw: kw["flag_terms"],
    )
    monkeypatch.setattr(
        research_mod,
        "refine_dataset_names_with_llm",
        lambda _agent, **kw: kw["dataset_names"],
    )
    monkeypatch.setattr(
        research_mod,
        "validate_no_flag_alias_overlap",
        lambda aliases, flags: (aliases, flags, []),
    )
    monkeypatch.setattr(
        research_mod,
        "detect_subdataset_aliases",
        lambda _agent, _main, aliases: (aliases, []),
    )

    agent = MagicMock()
    agent.get_information.return_value = "stub"

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        research_multilingual_aliases=False,
    )
    uc = DatasetResearchUseCase(
        agent=agent,
        extractor=_minimal_extractor(),
        repository=MagicMock(),
        settings=settings,
        literature_gate=None,
    )
    record = uc._run_pipeline_once(ResearchRequest(dataset_name="Current Population Survey"))
    assert record.dataset_names


class _StrictToolAgent:
    """Mimics real agents: ``get_structured`` needs a tool dict with a ``name``."""

    def __init__(self, suffix_answer: str) -> None:
        self.suffix_answer = suffix_answer
        self.tools: list[str] = []

    def get_information(self, *_a, **_k) -> str:
        return "stub"

    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        self.tools.append(result_tool["name"])  # KeyError: 'name' for a bare schema
        assert "suffixed_form" in result_tool["input_schema"]["properties"]
        return {"suffixed_form": self.suffix_answer}


def _run_suffix_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path, answer: str):
    for attr, fn in {
        "web_search": lambda *_a, **_k: "",
        "refine_flag_terms_with_llm": lambda _agent, **kw: kw["flag_terms"],
        "refine_dataset_names_with_llm": lambda _agent, **kw: kw["dataset_names"],
        "validate_no_flag_alias_overlap": lambda a, f: (a, f, []),
        "detect_subdataset_aliases": lambda _agent, _m, a: (a, []),
    }.items():
        monkeypatch.setattr(research_mod, attr, fn)
    agent = _StrictToolAgent(answer)
    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        research_multilingual_aliases=False,
        research_collision_max_aliases=0,
        research_suffix_check_max_aliases=3,
    )
    uc = DatasetResearchUseCase(
        agent=agent,  # type: ignore[arg-type]
        extractor=_minimal_extractor(),
        repository=MagicMock(),
        settings=settings,
        literature_gate=None,
    )
    record = uc._run_pipeline_once(ResearchRequest(dataset_name="Current Population Survey"))
    return agent, record


def test_llm_suffix_fallback_uses_a_named_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog
) -> None:
    caplog.set_level("WARNING")
    agent, record = _run_suffix_fallback(
        monkeypatch, tmp_path, "Current Population Survey dataset"
    )
    assert agent.tools == ["emit_suffix_form"]
    assert "Current Population Survey dataset" in record.dataset_names
    assert "suffix fallback failed" not in caplog.text


def test_llm_suffix_fallback_empty_answer_adds_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _agent, record = _run_suffix_fallback(monkeypatch, tmp_path, "")
    assert record.dataset_names == ["Current Population Survey"]

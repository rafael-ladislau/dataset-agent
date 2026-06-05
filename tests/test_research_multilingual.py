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

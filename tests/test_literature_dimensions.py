"""Tests for Dimensions literature DSL builder and record field promotion."""

from __future__ import annotations

from dataset_agent.adapters.literature import (
    build_dimensions_literature_query,
    promote_dimensions_metrics_from_detail,
)
from dataset_agent.domain.models import DatasetRecord, Group, YearsRange


def _sample_record() -> DatasetRecord:
    return DatasetRecord(
        engine="dimensions",
        group=Group(name="default"),
        main_dataset_name="O*NET",
        description="Occupational Information Network",
        dataset_names=["O*NET", "Occupational Information Network", "ONET"],
        flag_terms=["U.S. Department of Labor", "ETA"],
        years_range=YearsRange(start_year=2015, end_year=2025),
        official_name="O*NET",
    )


def test_build_dimensions_literature_query_includes_dataset_and_flag_terms() -> None:
    q = build_dimensions_literature_query(_sample_record(), sample_size=10)
    assert q.startswith("search publications in full_data for ")
    assert "O*NET" in q or "O\\*NET" in q
    assert "Occupational Information Network" in q
    assert "U.S. Department of Labor" in q
    assert "ETA" in q
    assert " AND " in q
    assert "limit 10" in q
    assert "return publications[basics + abstract + concepts_scores + times_cited]" in q


def test_build_dimensions_literature_query_without_flag_terms() -> None:
    record = _sample_record()
    record.flag_terms = []
    q = build_dimensions_literature_query(record, sample_size=5)
    assert " AND " not in q
    assert "limit 5" in q


def test_promote_dimensions_metrics_from_detail() -> None:
    record = _sample_record()
    detail = {
        "query": 'search publications in full_data for "test" limit 10',
        "publications_total": 42,
    }
    promote_dimensions_metrics_from_detail(record, detail)
    assert record.dsl_query == detail["query"]
    assert record.publications_total == 42


def test_promote_dimensions_metrics_from_detail_empty() -> None:
    record = _sample_record()
    promote_dimensions_metrics_from_detail(record, {})
    assert record.dsl_query is None
    assert record.publications_total is None


def test_attach_dimensions_preview_if_needed(monkeypatch) -> None:
    from dataset_agent.application.research import _attach_dimensions_preview_if_needed
    from dataset_agent.settings import Settings

    record = _sample_record()
    settings = Settings(
        literature_gate="noop",
        dimensions_api_key="test-key",
    )
    dsl = build_dimensions_literature_query(record, sample_size=10)

    def _fake_fetch(rec, *, sample_size: int, settings):  # noqa: ARG001
        return dsl, 12345

    monkeypatch.setattr(
        "dataset_agent.application.research.fetch_dimensions_publication_metrics",
        _fake_fetch,
    )
    _attach_dimensions_preview_if_needed(record, settings, sample_size=10)
    assert record.dsl_query == dsl
    assert record.publications_total == 12345

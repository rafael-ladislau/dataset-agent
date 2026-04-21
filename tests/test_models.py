"""Tests for domain models and record builder."""

from __future__ import annotations

from types import SimpleNamespace

from dataset_agent.domain.models import (
    DatasetRecord,
    Group,
    ResearchRequest,
    ValidationRequest,
    YearsRange,
    build_record_from_pipeline,
)


def test_build_record_merges_defaults_and_heuristic_official_name() -> None:
    defaults = SimpleNamespace(
        default_years_start=2010,
        default_years_end=2020,
        default_group_name="grp",
        default_publication_types=["article"],
        default_filter_us_affiliation=False,
        default_engine="dimensions",
    )
    req = ResearchRequest(dataset_name="My Dataset")
    rec = build_record_from_pipeline(
        req,
        description="desc",
        dataset_names=["My Dataset"],
        flag_terms=[],
        access_type="Open",
        data_url="https://d.example",
        schema_url=None,
        documentation_url=None,
        defaults=defaults,
    )
    assert rec.main_dataset_name == "My Dataset"
    assert rec.official_name == "My Dataset"
    assert rec.engine == "dimensions"
    assert rec.group == Group(name="grp")
    assert rec.years_range == YearsRange(start_year=2010, end_year=2020)
    assert rec.data_url == "https://d.example"


def test_research_request_defaults_include_llm_batch_size() -> None:
    req = ResearchRequest(dataset_name="Dataset X")
    assert req.sample_size == 10
    assert req.llm_batch_size == 25


def test_validation_request_accepts_sample_size_up_to_1000() -> None:
    req = ValidationRequest(
        main_dataset_name="Dataset X",
        sample_size=1000,
        llm_batch_size=120,
    )
    assert req.sample_size == 1000
    assert req.llm_batch_size == 120


def test_validation_request_exclude_terms_and_record_retry_links() -> None:
    v = ValidationRequest(
        main_dataset_name="Main",
        dataset_names=["a"],
        flag_terms=["b"],
        exclude_terms=["noise", "Other"],
    )
    assert v.exclude_terms == ["noise", "Other"]

    retry = ValidationRequest(
        main_dataset_name="Main",
        dataset_names=["new alias"],
        flag_terms=["org"],
        exclude_terms=["noise", "stem"],
    )
    rec = DatasetRecord(
        engine="dimensions",
        group=Group(name="g"),
        main_dataset_name="Main",
        years_range=YearsRange(start_year=2020, end_year=2021),
        retry_validation=retry,
        links={"retry": {"href": "https://api.example/validate", "method": "POST"}},
    )
    assert rec.retry_validation is not None
    assert rec.retry_validation.exclude_terms == ["noise", "stem"]
    assert rec.links is not None
    assert rec.links["retry"]["href"].endswith("/validate")

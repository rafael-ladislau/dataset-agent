"""Round 2, item 15: the program and every dataset found for it are on the output."""

from __future__ import annotations

from dataset_agent.adapters.resource_map import (
    build_map,
    clause_roles,
    data_kind,
    program_datasets,
    program_datasets_summary,
)
from dataset_agent.domain.models import DatasetRecord, QueryOptimizationRecord
from tests.test_resource_map import PAGES, _count, _raw


def _map():
    raw = _raw(data_resource_names_in_pages=["Controlled Tier", "Hidden Portal", "Data Browser"])
    pages = PAGES + " The Data Browser shows counts."
    return build_map(raw, dataset_name="Example Program", pages=pages,
                     count=_count({"Example Workbench": 40, "Controlled Tier": 12}))


def test_clause_roles():
    roles = clause_roles('(\\"A data\\" OR \\"B\\") OR ((\\"P\\" OR \\"Q\\") AND (\\"W\\" OR \\"T\\")) '
                         'NOT (\\"X\\")')
    assert roles == {"a data": "standalone", "b": "standalone", "p": "gated", "q": "gated",
                     "w": "gate", "t": "gate", "x": "exclude"}


def test_every_examined_name_is_listed_with_its_status():
    m = _map()
    clause = ('(\\"Example Workbench\\") OR ((\\"Example Program\\") AND (\\"Controlled Tier\\"))')
    pd = program_datasets(m, clause=clause)
    rows = {d.name: d for d in pd.datasets}
    assert pd.program_names[0] == "Example Program"
    assert pd.former_or_parent_names == ["Parent Cohort Program"]
    assert rows["Example Workbench"].status == "standalone"
    assert rows["Controlled Tier"].status == "gate" and rows["Controlled Tier"].kind == "access_tier"
    assert rows["Hidden Portal"].status == "not_used"
    assert rows["Hidden Portal"].reason == "not written in the pages read"
    assert rows["Data Browser"].status == "not_used" and "0 hits" in rows["Data Browser"].reason
    assert rows["Example Workbench"].hits_with_program == 40
    text = program_datasets_summary(pd)
    assert "Program: Example Program" in text and "Hidden Portal" in text and "[gate]" in text


def test_research_view_has_no_status_and_old_records_load():
    pd = program_datasets(_map(), built=False)
    assert all(d.status is None for d in pd.datasets)
    assert program_datasets(None) is None and program_datasets_summary(None) == ""
    assert DatasetRecord.model_validate({
        "engine": "dimensions", "group": {"name": "g"}, "main_dataset_name": "X",
        "description": "", "dataset_names": [], "flag_terms": [],
        "years_range": {"start_year": 2015, "end_year": 2026}}).program_datasets is None
    assert QueryOptimizationRecord().program_datasets is None


def test_kinds():
    assert data_kind("Researcher Workbench") == "platform"
    assert data_kind("Curated Data Repository") == "repository"
    assert data_kind("Controlled Tier Dataset v7") == "release"
    assert data_kind("Something Else") == "other"

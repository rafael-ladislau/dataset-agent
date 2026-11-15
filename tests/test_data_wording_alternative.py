"""Round 3, item 3: the data-wording clause is an informational alternative (count and
clause), clearly marked as not the released query; program_datasets stays."""

from __future__ import annotations

from dataset_agent.adapters.resource_map import data_wording_clause, program_datasets
from dataset_agent.domain.models import ResourceMap

MAP = ResourceMap(
    program_names=["All of Example", "AOE"],
    data_resource_names=["Example Workbench", "Data Browser", "DB", "Huge Name"],
    specific_data_names=["Example Workbench"],
    data_name_hits={"Example Workbench": 900, "Data Browser": 12_000, "DB": 5,
                    "Huge Name": 900_000},
)


def test_data_wording_clause_has_the_round2_shape():
    clause = data_wording_clause(MAP, max_hits=500_000)
    safe_part, _, gated_part = clause.partition(" OR ((")
    assert '\\"All of Example data\\"' in safe_part and '\\"Example Workbench\\"' in safe_part
    assert "AOE data" not in safe_part  # acronym phrases never stand alone
    assert '\\"All of Example\\"' in gated_part and '\\"AOE\\"' in gated_part
    assert '\\"Data Browser\\"' in gated_part
    assert "DB" not in clause.replace("Data Browser", "") and "Huge Name" not in clause


def test_no_map_no_alternative():
    assert data_wording_clause(None) is None


def test_gated_names_are_reported_as_gated():
    clause = '(\\"A\\") OR ((\\"Example Workbench\\") AND (\\"Agency\\"))'
    pd = program_datasets(MAP, clause=clause)
    st = {d.name: d.status for d in pd.datasets}
    assert st["Example Workbench"] == "gated"


def test_optimize_record_carries_the_alternative(tmp_path, monkeypatch):
    import asyncio
    from unittest.mock import MagicMock

    from dataset_agent.application import query_optimization as qopt_mod
    from dataset_agent.application.query_optimization import QueryOptimizationUseCase
    from dataset_agent.domain.models import DatasetRecord, Group, OptimizeRequest, YearsRange
    from tests.test_fulltext_loop import CountDSL, FakeClient, make_settings

    def fake_eval(*a, **k):
        return {"fp_rate_pct": 5.0, "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                "total_pubs": 100, "coverage_pct": 100.0, "for_clause": k.get("for_clause", ""),
                "publications_total": 100, "fp_hits": []}

    monkeypatch.setattr(qopt_mod, "evaluate_for_clause_sync", fake_eval)
    rec = DatasetRecord(
        engine="dimensions", group=Group(name="default"), main_dataset_name="Example Program",
        description="A cohort.", dataset_names=["Example Program"], flag_terms=["Agency"],
        years_range=YearsRange(start_year=2015, end_year=2026),
        resource_map=ResourceMap(program_names=["Example Program"],
                                 data_resource_names=["Example Workbench"],
                                 specific_data_names=["Example Workbench"]),
    )
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=make_settings(tmp_path), extraction_client=FakeClient([{}]))

    async def stored(_req):
        return rec.model_copy(deep=True)

    uc._run_research = stored
    out = asyncio.run(uc.execute(OptimizeRequest(dataset_name="Example Program")))
    pd = out.program_datasets
    assert pd.data_wording_clause and "Example Program data" in pd.data_wording_clause
    assert pd.data_wording_clause != out.for_clause
    assert pd.data_wording_count is not None
    assert "not the released query" in pd.data_wording_note
    assert "data-wording alternative" in out.notes


def test_empty_gate_leaves_the_program_names_out():
    # Every data name is a bare short acronym or above max_hits: no gate.
    rmap = ResourceMap(program_names=["All of Example Research Program", "All of Example", "AOE"],
                       data_resource_names=["CDR", "Example Workbench"],
                       specific_data_names=[],
                       data_name_hits={"Example Workbench": 700_000})
    clause = data_wording_clause(rmap, max_hits=500_000)
    assert clause is not None
    assert '\\"All of Example Research Program data\\"' in clause
    for bare in ('\\"All of Example\\"', '\\"AOE\\"', '\\"All of Example Research Program\\"'):
        assert bare not in clause


def test_nothing_safe_and_no_gate_gives_no_alternative():
    rmap = ResourceMap(program_names=["AOE", "All of Example Cohort"],  # acronym; data noun
                       data_resource_names=["CDR"], specific_data_names=[])
    assert data_wording_clause(rmap, max_hits=500_000) is None

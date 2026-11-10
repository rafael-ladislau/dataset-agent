"""Regression tests for evaluate_for_clause_sync and gate build fixes."""

from __future__ import annotations

import sys
import types

from dataset_agent.adapters import literature as lit
from dataset_agent.domain.ports import AgentPort
from dataset_agent.settings import Settings


class _FakeAgent(AgentPort):
    def get_information(self, prompt, *, tools=None):
        return ""

    def get_structured(self, prompt, result_tool):
        return {"is_genuine_dataset_reference": False}


def _install_fake_dimcli(monkeypatch, n_pubs: int, calls: list | None = None):
    pubs = [
        {"id": f"p{i}", "title": "NHIS survey", "abstract": "uses NHIS data"} for i in range(n_pubs)
    ]

    class _Dsl:
        def query(self, q):
            if calls is not None:
                calls.append(q)
            return {"publications": pubs, "_stats": {"total_count": n_pubs}}

    mod = types.SimpleNamespace(login=lambda key=None: None, Dsl=_Dsl)
    monkeypatch.setitem(sys.modules, "dimcli", mod)


def test_fp_rate_uses_classified_denominator(monkeypatch):
    _install_fake_dimcli(monkeypatch, 500)
    s = Settings(dimensions_api_key="k", optimize_abstract_classify_max_calls=12, fp_sample_size=500)
    out = lit.evaluate_for_clause_sync("NHIS", ["NHIS"], "NHIS", s, agent=_FakeAgent())
    assert out["fp_count"] == 12
    assert out["classified_count"] == 12
    assert out["matched_count"] == 500
    assert out["genuine_count"] == 0
    assert out["fp_rate_pct"] == 100.0


def test_gate_for_clause_keeps_flag_terms():
    from dataset_agent.domain.models import DatasetRecord, Group, YearsRange

    rec = DatasetRecord(
        engine="dimensions",
        group=Group(name="default"),
        main_dataset_name="Foo Survey",
        dataset_names=["FS"],
        flag_terms=["health"],
        exclude_terms=["Foo Fighters"],
        years_range=YearsRange(start_year=2015, end_year=2025),
    )
    clause = lit._build_for_clause_from_record(rec)
    assert "health" in clause
    assert " AND " in clause
    assert "NOT" in clause and "Foo Fighters" in clause


def test_phase3_starts_from_lowest_fp_variant_with_real_label():
    from dataset_agent.application.query_optimization import _pick_start_variant
    from dataset_agent.domain.models import VariantTestedEntry

    tested = [
        VariantTestedEntry(label="V1", for_clause="a", expected_count=10, fp_rate_pct=20.0),
        VariantTestedEntry(label="V3", for_clause="b", expected_count=10, fp_rate_pct=2.0),
        VariantTestedEntry(label="V4", for_clause="c", expected_count=10, fp_rate_pct=2.0),
        VariantTestedEntry(label="V2", for_clause="d", expected_count=0, fp_rate_pct=0.0),
    ]
    # V2 has 0 FP but zero hits so it loses; V4 wins the FP tie by preference order.
    assert _pick_start_variant(tested).label == "V4"


def test_evaluator_uses_runner_instead_of_dimcli(monkeypatch):
    monkeypatch.setitem(sys.modules, "dimcli", None)  # import would fail
    seen: list[str] = []

    def runner(q):
        seen.append(q)
        return {"publications": [{"id": "1", "title": "NHIS", "abstract": "NHIS"}],
                "_stats": {"total_count": 1}}

    s = Settings(dimensions_api_key="k")
    out = lit.evaluate_for_clause_sync("NHIS", ["NHIS"], "NHIS", s, dsl_runner=runner)
    assert len(seen) == 1
    assert out["matched_count"] == 1


def test_optimizer_eval_calls_count_against_budget(tmp_path):
    import asyncio

    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from dataset_agent.application.query_optimization import QueryOptimizationUseCase
    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError

    calls: list[str] = []

    class _Dsl:
        async def execute_dsl(self, dsl):
            calls.append(dsl)
            return SimpleNamespace(count_total=1)

    s = Settings(dimensions_api_key="k", optimize_max_dimensions_calls=2)
    uc = QueryOptimizationUseCase(dsl_port=_Dsl(), research=MagicMock(), agent=MagicMock(), settings=s)

    async def go():
        run = uc._make_eval_runner(asyncio.get_running_loop())
        await asyncio.to_thread(run, "q1")
        await asyncio.to_thread(run, "q2")
        try:
            await asyncio.to_thread(run, "q3")
        except DimensionsDslError:
            return True
        return False

    assert asyncio.run(go()) is True
    assert uc._dims_calls == 2 and calls == ["q1", "q2"]

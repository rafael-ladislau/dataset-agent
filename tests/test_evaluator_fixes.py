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

    mod = types.SimpleNamespace(login=lambda key=None, **kw: None, Dsl=_Dsl)
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


def test_phase3_starts_from_non_empty_variant_by_preference_with_real_label():
    from dataset_agent.application.query_optimization import _pick_start_variant
    from dataset_agent.domain.models import VariantTestedEntry

    tested = [
        VariantTestedEntry(label="V1", for_clause="a", expected_count=10, fp_rate_pct=20.0),
        VariantTestedEntry(label="V3", for_clause="b", expected_count=10, fp_rate_pct=2.0),
        VariantTestedEntry(label="V4", for_clause="c", expected_count=10, fp_rate_pct=2.0),
        VariantTestedEntry(label="V2", for_clause="d", expected_count=0, fp_rate_pct=0.0),
    ]
    # V2 has zero hits so it loses; among the rest V4 wins by preference order (FP is ignored).
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


def _install_scripted_dimcli(monkeypatch, outcomes, calls):
    class _Dsl:
        def query(self, q):
            calls.append(q)
            return outcomes[min(len(calls) - 1, len(outcomes) - 1)]

    fake = types.ModuleType("dimcli")
    fake.login = lambda **kw: None  # type: ignore[attr-defined]
    fake.Dsl = _Dsl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dimcli", fake)


def test_evaluator_raw_path_retries_transient_then_succeeds(monkeypatch):
    from types import SimpleNamespace

    from dataset_agent.adapters import dimensions_dsl

    monkeypatch.setattr(dimensions_dsl, "TRANSIENT_BACKOFF_SECONDS", (0.0, 0.0))
    transient = SimpleNamespace(
        count_total=None,
        errors={"query": {"header": "Service unavailable", "details": ["Unexpected Error x"]}},
    )
    good = {"publications": [], "_stats": {"total_count": 0}}
    calls: list = []
    _install_scripted_dimcli(monkeypatch, [transient, good], calls)
    out = lit.evaluate_for_clause_sync("NHIS", ["NHIS"], "NHIS", Settings(dimensions_api_key="k"))
    assert "error" not in out
    assert len(calls) == 2


def test_evaluator_raw_path_errors_result_returns_error_dict(monkeypatch):
    from types import SimpleNamespace

    bad = SimpleNamespace(count_total=None, errors={"query": "QuerySyntaxErrors"})
    calls: list = []
    _install_scripted_dimcli(monkeypatch, [bad], calls)
    out = lit.evaluate_for_clause_sync("NHIS", ["NHIS"], "NHIS", Settings(dimensions_api_key="k"))
    assert out["error"].startswith("dimensions_query_failed")
    assert out["error_reason"]
    assert len(calls) == 1


def test_evaluator_runner_path_is_not_double_retried(monkeypatch):
    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError

    monkeypatch.setitem(sys.modules, "dimcli", None)
    seen: list[str] = []

    def runner(q):
        seen.append(q)
        raise DimensionsDslError("boom", transient=True)

    out = lit.evaluate_for_clause_sync(
        "NHIS", ["NHIS"], "NHIS", Settings(dimensions_api_key="k"), dsl_runner=runner
    )
    assert out["error"].startswith("dimensions_query_failed")
    assert len(seen) == 1


class _FailingAgent(_FakeAgent):
    def get_structured(self, prompt, result_tool):
        raise RuntimeError("LLM down")


class _FlakyAgent(_FakeAgent):
    """Every third classification call fails; the others say false positive."""

    def __init__(self):
        self.n = 0

    def get_structured(self, prompt, result_tool):
        self.n += 1
        if self.n % 3 == 0:
            raise RuntimeError("LLM down")
        return {"is_genuine_dataset_reference": False}


def test_classify_alias_mention_returns_none_when_the_call_fails():
    from dataset_agent.adapters.fp_analysis import classify_alias_mention

    assert classify_alias_mention(_FailingAgent(), dataset_name="X", alias="Y", snippet="z") is None


def test_all_classifications_failing_leaves_the_rate_unknown(monkeypatch):
    _install_fake_dimcli(monkeypatch, 20)
    s = Settings(dimensions_api_key="k", optimize_abstract_classify_max_calls=12, fp_sample_size=500)
    out = lit.evaluate_for_clause_sync("NHIS", ["NHIS"], "NHIS", s, agent=_FailingAgent())
    assert out["fp_rate_pct"] is None
    assert out["classified_count"] == 0 and out["genuine_count"] == 0 and out["fp_count"] == 0
    assert out["failed_count"] == 12 and out["attempted_count"] == 12


def test_failed_classifications_stay_out_of_the_rate_denominator(monkeypatch):
    _install_fake_dimcli(monkeypatch, 20)
    s = Settings(dimensions_api_key="k", optimize_abstract_classify_max_calls=12, fp_sample_size=500)
    out = lit.evaluate_for_clause_sync("NHIS", ["NHIS"], "NHIS", s, agent=_FlakyAgent())
    # 12 calls, 4 fail; the 8 classified are all false positives.
    assert out["failed_count"] == 4 and out["classified_count"] == 8
    assert out["fp_count"] == 8 and out["genuine_count"] == 0
    assert out["fp_rate_pct"] == 100.0


def test_all_calls_failing_is_unknown_even_with_alias_less_papers_counted_genuine(monkeypatch):
    # One matched paper has no alias in its text (counted genuine with no call); the
    # three that need a call all fail. The rate is unknown, not 0%.
    pubs = [{"id": "p0", "title": "NHIS survey", "abstract": "uses NHIS data"}] + [
        {"id": f"p{i}", "title": "NHIS survey", "abstract": "uses NHIS data"} for i in range(1, 4)
    ]

    class _Dsl:
        def query(self, q):
            return {"publications": pubs, "_stats": {"total_count": len(pubs)}}

    monkeypatch.setitem(sys.modules, "dimcli", types.SimpleNamespace(login=lambda **kw: None, Dsl=_Dsl))
    monkeypatch.setattr(lit, "_find_matched_alias", lambda pub, aliases: None if pub["id"] == "p0" else "NHIS")
    s = Settings(dimensions_api_key="k", optimize_abstract_classify_max_calls=12, fp_sample_size=500)
    out = lit.evaluate_for_clause_sync("NHIS", ["NHIS"], "NHIS", s, agent=_FailingAgent())
    assert out["genuine_count"] == 1 and out["failed_count"] == out["attempted_count"] == 3
    assert out["fp_rate_pct"] is None


def test_fp_better_treats_unknown_as_worst_and_zero_as_best():
    from dataset_agent.application.query_optimization import _fp_better

    assert _fp_better(0.0, None)
    assert not _fp_better(2.0, 0.0)  # 0% then 2%: the 0% round is kept
    assert _fp_better(0.0, 2.0)
    assert not _fp_better(None, 50.0)
    assert not _fp_better(None, None)

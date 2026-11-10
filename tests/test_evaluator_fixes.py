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
    from dataset_agent.domain.models import DatasetRecord, Group

    rec = DatasetRecord(
        engine="dimensions",
        group=Group(name="default"),
        main_dataset_name="Foo Survey",
        dataset_names=["FS"],
        flag_terms=["health"],
        exclude_terms=["Foo Fighters"],
    )
    clause = lit._build_for_clause_from_record(rec)
    assert "health" in clause
    assert " AND " in clause
    assert "NOT" in clause and "Foo Fighters" in clause

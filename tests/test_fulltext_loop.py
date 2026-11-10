"""Phase-3 full-text evidence and refinement loop with a fake extraction client."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dataset_agent.application import query_optimization as qopt_mod
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import OptimizeRequest
from dataset_agent.settings import Settings


def pub(pid, cites, used, quote="", status="delivered"):
    return {
        "publication_id": pid, "doi": f"10/{pid}", "title": f"T {pid}", "citation_count": cites,
        "fulltext": {"status": status, "source": "pmc", "reason": None}, "analyzed": True,
        "validated_datasets": [{"name": "X", "is_used": used, "is_mentioned": True, "text_quote": quote}],
    }


class FakeClient:
    def __init__(self, results_seq):
        self.results_seq = list(results_seq)
        self.seeds: list[dict] = []

    def run_probe(self, seed, top_n, pull_cap):
        self.seeds.append(seed)
        r = self.results_seq.pop(0) if len(self.results_seq) > 1 else self.results_seq[0]
        if isinstance(r, Exception):
            raise r
        return r


class CountDSL:
    def __init__(self):
        self.queries: list[str] = []

    async def execute_dsl(self, dsl):
        self.queries.append(dsl)
        return SimpleNamespace(count_total=100, publications=[])


@pytest.fixture(autouse=True)
def _patch_eval(monkeypatch):
    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync",
        lambda *a, **k: {"fp_rate_pct": 5.0, "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                         "total_pubs": 100, "coverage_pct": 100.0, "for_clause": k.get("for_clause", ""),
                         "publications_total": 100, "fp_hits": []},
    )


def make_settings(tmp_path, **kw):
    base = dict(tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out", dimensions_api_key="k",
                optimize_max_dimensions_calls=100, optimize_signal3_max_titles=0,
                optimize_abstract_exclude_max_hits=0, optimize_abstract_classify_max_calls=0,
                optimize_fp_domains_max_aliases=0, optimize_web_disambig_max_aliases=0,
                optimize_suffix_check_max_aliases=0)
    base.update(kw)
    return Settings(**base)


REQ = OptimizeRequest(dataset_name="My Long Dataset Alpha",
                      dataset_names=["My Long Dataset Alpha", "Another Long Title Here"],
                      flag_terms=["NIH"])


def test_disabled_by_default_never_touches_extraction(tmp_path):
    client = FakeClient([{}])
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=make_settings(tmp_path), extraction_client=client)
    out = asyncio.run(uc.execute(REQ))
    assert out.success and client.seeds == [] and out.fulltext_iterations == []


def test_probe_metrics_recorded_when_enabled(tmp_path):
    results = {"publications": [
        pub("a", 9, True, "used My Long Dataset Alpha data"),
        pub("b", 8, False, "unrelated Another Long Title Here"),
        pub("c", 7, True, "Alpha again", ),
        pub("d", 6, True, "", status="missing"),
    ]}
    client = FakeClient([results])
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True, fulltext_top_n=4),
        extraction_client=client)
    out = asyncio.run(uc.execute(REQ))
    assert out.success
    m = out.fulltext_iterations[0]
    assert m.n_evaluated == 3 and m.fulltext_precision == pytest.approx(2 / 3)
    assert m.oa_coverage == 0.75 and m.abstract_fp_rate_pct == 5.0
    assert client.seeds[0]["standalone_names"]


def test_probe_failure_is_recorded_not_fatal(tmp_path):
    from dataset_agent.adapters.extraction_client import ExtractionError

    client = FakeClient([ExtractionError("down")])
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True), extraction_client=client)
    out = asyncio.run(uc.execute(REQ))
    assert out.success and out.fulltext_iterations[0].error == "down"


def test_loop_refines_with_evidence_and_selects_best_iteration(tmp_path):
    bad = [pub(f"n{i}", 50 - i, False, "Another Long Title Here nursing") for i in range(4)] + [
        pub(f"g{i}", 30 - i, True, "My Long Dataset Alpha data") for i in range(4)]
    good = [pub(f"g{i}", 30 - i, True, "My Long Dataset Alpha data") for i in range(4)]
    client = FakeClient([{"publications": bad}, {"publications": good}])
    llm = MagicMock()
    llm.get_structured.return_value = {"exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""}
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=llm,
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True, fulltext_top_n=20,
                               optimize_max_iterations=3, refine_min_alias_papers=3),
        extraction_client=client)
    out = asyncio.run(uc.execute(REQ))
    assert [m.fulltext_precision for m in out.fulltext_iterations] == [0.5, 1.0]
    assert len(client.seeds) == 2
    # the dropped alias is gone from the second seed and from the selected output
    second = client.seeds[1]
    assert "Another Long Title Here" not in second["standalone_names"] + second["dataset_names"]
    assert out.selected_variant.endswith("~gate1")
    assert "Another Long Title Here" not in out.aliases.safe + out.aliases.risky

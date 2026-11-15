"""The release decision reads a random, year-stratified sample of the selected query's papers
with full text (split datasets by default), not the top-cited papers the refiner learns from."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dataset_agent.adapters import release_sample as rs
from dataset_agent.adapters.extraction_client import (
    ExtractionClient,
    ExtractionError,
    YearSampleResult,
)
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import ConfidenceLevel, OptimizeRequest
from dataset_agent.settings import Settings
from tests.test_fulltext_loop import CountDSL, FakeClient, make_settings
from tests.test_split_release_bar import MAP, _pubs

# -- allocation, order, caps -----------------------------------------------------------


def test_allocation_is_proportional_and_sums_to_the_target():
    assert rs.allocate({2020: 60, 2021: 40}, 21) == {2020: 13, 2021: 8}
    alloc = rs.allocate({2018: 3, 2019: 500, 2020: 1200, 2021: 0, 2022: 900}, 30)
    assert sum(alloc.values()) == 30 and 2021 not in alloc
    assert alloc[2020] > alloc[2022] > alloc[2019]
    assert rs.allocate({2020: 5}, 21) == {2020: 5}  # fewer papers than the target
    assert rs.allocate({}, 21) == {} and rs.allocate({2020: 9}, 0) == {}


def test_draw_order_is_seeded_per_year_and_ignores_input_order():
    pool = [f"pub.{i}" for i in range(50)]
    a = rs.draw_order(pool, 7, 2020)
    assert a == rs.draw_order(list(reversed(pool)), 7, 2020)
    assert a != rs.draw_order(pool, 8, 2020) and a != rs.draw_order(pool, 7, 2021)
    assert sorted(a) == sorted(pool)


def test_pull_cap_and_year_facet():
    assert rs.pull_cap(120) == 170 and rs.pull_cap(5000) == 1000
    res = SimpleNamespace(year=[{"id": "2021", "count": 40}, {"id": 2020, "count": 60}])
    assert rs.year_counts_from_result(res) == {2020: 60, 2021: 40}
    assert rs.year_counts_from_result({"year": [{"id": 2019, "count": 3}]}) == {2019: 3}


def test_defaults():
    f = Settings.model_fields
    assert f["fulltext_release_sampling"].default == "random_split"
    assert f["fulltext_split_min_sample"].default == 21
    assert f["fulltext_release_seed"].default is None
    assert f["fulltext_release_shadow"].default is False


# -- the client ------------------------------------------------------------------------


def test_client_draws_each_year_at_random_and_keeps_the_allocation(monkeypatch):
    pools = {2020: [f"a{i}" for i in range(10)], 2021: [f"b{i}" for i in range(20)]}
    c = ExtractionClient("http://extraction.invalid")
    state: dict = {"tasks": {}, "analyzed": {}, "cleaned": [], "caps": {}}

    def start(body, cap, *, entity="dataset"):
        y = body["years_range"]["start_year"]
        assert body["years_range"]["end_year"] == y
        tid = f"t{y}"
        state["tasks"][tid] = y
        state["caps"][y] = cap
        state["analyzed"][tid] = set()
        return tid

    def results(tid, contexts=False):
        y = state["tasks"][tid]
        out = []
        for pid in pools[y]:
            done = pid in state["analyzed"][tid]
            ok = done and int(pid[1:]) % 2 == 0
            out.append({"publication_id": pid,
                        "fulltext": ({"status": "delivered"} if ok else
                                     {"status": "missing", "reason": "no_open_copy"} if done else
                                     {"status": "missing", "reason": "not_attempted"})})
        return {"publications": out}

    monkeypatch.setattr(c, "require_probe_support", lambda route: None)
    monkeypatch.setattr(c, "start_search", start)
    monkeypatch.setattr(c, "get_llm_results", results)
    monkeypatch.setattr(c, "analyze_ids",
                        lambda tid, ids, steps=None: state["analyzed"][tid].update(ids))
    monkeypatch.setattr(c, "wait_batch_done", lambda tid, ids: results(tid))
    monkeypatch.setattr(c, "cleanup", lambda tid: state["cleaned"].append(tid) or True)

    out = c.run_year_sample({"years_range": {"start_year": 2015, "end_year": 2026}},
                            allocation={2020: 2, 2021: 4}, year_counts={2020: 10, 2021: 20},
                            rng_seed=11, batch_size=30, max_batches_per_year=3)
    assert isinstance(out, YearSampleResult) and out.rng_seed == 11
    assert state["caps"] == {2020: 60, 2021: 70} and sorted(state["cleaned"]) == ["t2020", "t2021"]
    assert out.years[2020]["kept"] == 2 and out.years[2021]["kept"] == 4
    assert out.years[2020]["pool_covers_year"] and out.years[2021]["stop_reason"] == "target_met"
    kept = [p for p in out.results["publications"] if p["fulltext"]["status"] == "delivered"]
    assert {p["sample_year"] for p in kept} == {2020, 2021} and len(kept) == 6
    # The kept papers are the first delivered ones in the year's seeded draw order.
    for y, want in ((2020, 2), (2021, 4)):
        order = [i for i in rs.draw_order(pools[y], 11, y) if int(i[1:]) % 2 == 0]
        assert [p["publication_id"] for p in kept if p["sample_year"] == y] == order[:want]


# -- the release decision ----------------------------------------------------------------


class YearDSL(CountDSL):
    async def execute_dsl(self, dsl):
        self.queries.append(dsl)
        if " return year " in dsl:
            return SimpleNamespace(count_total=100, year=[{"id": 2020, "count": 60},
                                                          {"id": 2021, "count": 40}])
        return SimpleNamespace(count_total=100, publications=[])


class YearClient(FakeClient):
    def __init__(self, top_cited, sample, **kw):
        super().__init__([{"publications": top_cited}], **kw)
        self.sample = sample
        self.sample_calls: list[dict] = []

    def run_year_sample(self, seed, **kw):
        self.sample_calls.append(kw)
        if isinstance(self.sample, Exception):
            raise self.sample
        years = {y: {"count": 0, "allocation": k, "kept": k, "pool_covers_year": True}
                 for y, k in kw["allocation"].items()}
        return YearSampleResult(results={"publications": self.sample}, years=years,
                                rng_seed=kw["rng_seed"])


def _run(tmp_path, top_cited, sample, *, split=True, **settings):
    llm = MagicMock()
    llm.get_structured.return_value = {"exclude_terms": [], "flag_terms": [], "new_aliases": [],
                                       "reasoning": ""}
    client = YearClient(top_cited, sample)
    uc = QueryOptimizationUseCase(
        dsl_port=YearDSL(), research=MagicMock(), agent=llm,
        settings=make_settings(tmp_path, **{
            "fulltext_refinement_enabled": True, "optimize_max_iterations": 1,
            "fulltext_recall_enabled": False, "fulltext_target_delivered": 10,
            "fulltext_release_seed": 12345, **settings}),
        extraction_client=client)

    async def label(ev, _request):
        if uc._resource_map is None:
            return
        ev.scope_ran = True
        for v in ev.verdicts:
            if v.publication_id.startswith("d"):
                v.scope = "data"
            elif v.publication_id.startswith("p"):
                v.scope = "program_only"
            elif v.publication_id.startswith("r"):
                v.scope_references_only = True

    uc._apply_scope_check = label
    req = OptimizeRequest(dataset_name="Example Cohort Program",
                          dataset_names=["Example Cohort Program", "Example Cohort Study"],
                          flag_terms=["NIH"], resource_map=MAP if split else None)
    return asyncio.run(uc.execute(req)), client


@pytest.fixture(autouse=True)
def _zero_abstract_fp(monkeypatch):
    from dataset_agent.application import query_optimization as qopt_mod

    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync",
        lambda *a, **k: {"fp_rate_pct": 0.0, "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                         "total_pubs": 100, "coverage_pct": 100.0, "for_clause": k.get("for_clause", ""),
                         "publications_total": 100, "fp_hits": []},
    )


def test_split_release_is_decided_on_the_random_sample_not_the_top_cited(tmp_path):
    top = _pubs(5, 2, 2, 3)  # 3 of 12 unrelated: 0.25 would fail the bar
    sample = _pubs(14, 3, 3, 1)  # 1 of 21 unrelated, data 14 of 21
    out, client = _run(tmp_path, top, sample)
    assert client.kwargs, "the refinement probe still pulls top-cited papers"
    (call,) = client.sample_calls
    assert call["allocation"] == {2020: 13, 2021: 8} and call["rng_seed"] == 12345
    assert out.release_bar_met is True and out.data_floor_met is True
    assert out.confidence in (ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH)
    rel = out.release_sample
    assert rel["sampling"] == "random_year_stratified" and rel["decides"] is True
    assert rel["seed"] == 12345 and rel["min_sample"] == 21 and rel["target"] == 21
    assert rel["allocation"] == {2020: 13, 2021: 8}
    assert rel["year_counts"] == {2020: 60, 2021: 40}
    assert rel["n_evaluated"] == 21
    share = rel["unrelated_share"]
    assert share["value"] == pytest.approx(1 / 21, abs=1e-4)
    assert share["wilson95"][0] < share["value"] < share["wilson95"][1]
    assert out.fulltext_four_way["unrelated"]["count"] == 1
    assert "release sample: random year-stratified" in out.notes


def test_split_sample_below_the_minimum_is_not_released(tmp_path):
    out, _ = _run(tmp_path, _pubs(10, 1, 1, 0), _pubs(9, 1, 1, 1))  # 12 of 21
    assert out.release_bar_met is False and out.confidence == ConfidenceLevel.LOW
    assert "12 of 21 papers, below the minimum sample" in out.notes


def test_split_sample_failure_fails_closed(tmp_path):
    out, _ = _run(tmp_path, _pubs(10, 1, 1, 0), ExtractionError("probe start refused"))
    assert out.release_bar_met is False and out.confidence == ConfidenceLevel.LOW
    assert out.release_sample["error"] == "probe start refused"
    assert "release sample failed" in out.notes


def test_other_datasets_keep_the_top_cited_sample_by_default(tmp_path):
    out, client = _run(tmp_path, _pubs(10, 0, 0, 0), _pubs(0, 0, 0, 30), split=False)
    assert client.sample_calls == []
    assert out.release_bar_met is True
    assert out.release_sample["sampling"] == "top_cited" and out.release_sample["decides"]
    assert out.release_sample["precision"]["wilson95"][1] == pytest.approx(1.0)


def test_shadow_sample_is_recorded_without_changing_the_decision(tmp_path):
    out, client = _run(tmp_path, _pubs(10, 0, 0, 0), _pubs(0, 0, 0, 30), split=False,
                       fulltext_release_shadow=True)
    assert len(client.sample_calls) == 1 and out.release_bar_met is True
    shadow = out.release_sample["shadow"]
    assert shadow["sampling"] == "random_year_stratified" and shadow["decides"] is False
    assert shadow["precision"]["value"] == 0.0


def test_random_all_moves_other_datasets_to_the_random_sample(tmp_path):
    out, client = _run(tmp_path, _pubs(10, 0, 0, 0), _pubs(0, 0, 0, 30), split=False,
                       fulltext_release_sampling="random_all")
    assert len(client.sample_calls) == 1
    assert out.release_bar_met is False and out.confidence == ConfidenceLevel.LOW
    assert out.release_sample["min_sample"] == 10 and out.release_sample["decides"]


def test_top_cited_mode_still_needs_the_split_minimum(tmp_path):
    out, client = _run(tmp_path, _pubs(10, 1, 1, 0), [], fulltext_release_sampling="top_cited")
    assert client.sample_calls == [] and out.release_bar_met is False
    assert out.release_sample["sampling"] == "top_cited" and out.release_sample["min_sample"] == 21

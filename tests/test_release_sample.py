"""The release decision reads a random, year-stratified sample of the selected query's papers
with full text (split datasets by default), not the top-cited papers the refiner learns from."""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datetime import date

from dataset_agent.adapters import release_sample as rs
from dataset_agent.adapters.extraction_client import (
    ExtractionClient,
    ExtractionError,
    YearSampleResult,
)
from dataset_agent.adapters.seed_mapping import QueryCandidate
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
    assert f["fulltext_release_month_count_calls"].default == 60


# -- large years sliced by month -------------------------------------------------------


def test_a_large_year_starts_above_the_whole_year_pull_bound():
    """``pull_cap`` gives count + 50 capped at 1,000: above 950 facet papers the cap
    itself has no headroom left, so one pull cannot pool the year's whole population any
    more and the draw leans on the year's most-cited 1,000. Only such a year is large."""
    assert rs.LARGE_YEAR_COUNT == rs.MAX_PULL - rs.PULL_HEADROOM == 950
    assert rs.is_large_year(950) is False and rs.is_large_year(951) is True
    assert rs.is_large_year(10_000) is True


def test_month_counts_allocate_the_year_quota_proportionally():
    """The year's quota splits over the months with the same largest-remainder rule."""
    got = {m: (300 if m == 1 else 20) for m in range(1, 13)}  # month 1 has most papers
    alloc = rs.allocate(got, 40)
    assert sum(alloc.values()) == 40
    assert alloc[1] > alloc[2]
    assert rs.allocate({m: 0 for m in range(1, 13)}, 40) == {}


def test_month_seed_is_deterministic_and_separates_the_months():
    """A month's draw does not depend on the other months or on the year draw."""
    assert rs.month_seed(11, 2021, 3) == rs.month_seed(11, 2021, 3)
    assert len({rs.month_seed(11, 2021, m) for m in range(1, 13)}) == 12
    assert rs.month_seed(11, 2021, 3) != rs.month_seed(11, 2022, 3)
    assert rs.month_seed(11, 2021, 3) != rs.year_seed(11, 2021)


def test_month_draw_order_is_seeded_per_month_and_ignores_input_order():
    pool = [f"m{i}" for i in range(30)]
    drawn = rs.draw_order(pool, 7, 2020, month=3)
    assert drawn == rs.draw_order(list(reversed(pool)), 7, 2020, month=3)
    assert drawn != rs.draw_order(pool, 7, 2020, month=4)
    assert drawn != rs.draw_order(pool, 7, 2020)  # whole-year draw, no month
    assert sorted(drawn) == sorted(pool)


def test_month_window_bounds_are_the_month_first_and_last_days():
    assert rs.month_window(2020, 2) == (date(2020, 2, 1), date(2020, 2, 29))
    assert rs.month_window(2021, 2) == (date(2021, 2, 1), date(2021, 2, 28))
    assert rs.month_window(2020, 12) == (date(2020, 12, 1), date(2020, 12, 31))
    assert rs.month_window(2021, 1) == (date(2021, 1, 1), date(2021, 1, 31))


def test_month_window_clamps_to_the_query_year_window():
    assert rs.month_window(2021, 12, within=(2020, 2021)) == (date(2021, 12, 1),
                                                              date(2021, 12, 31))
    assert rs.month_window(2021, 1, within=(2021, 2025)) == (date(2021, 1, 1), date(2021, 1, 31))
    assert rs.month_window(2019, 5, within=(2020, 2025)) is None  # does not touch it


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


def _month_client(monkeypatch, pools: dict):
    """A fake whose pools are keyed by year (whole-year probe) or (year, month) tuple
    (month probe); delivered full text is the pool's even-numbered entries, as before."""
    c = ExtractionClient("http://extraction.invalid")
    state: dict = {"tasks": {}, "task_of": {}, "bodies": {}, "caps": {}, "analyzed": {},
                   "cleaned": []}

    def start(body, cap, *, entity="dataset"):
        yr = body["years_range"]["start_year"]
        key = ((yr, int(str(body["date_range"]["date_from"])[5:7]))
               if "date_range" in body else yr)
        tid = f"t{key}"
        state["tasks"][tid] = key
        state["task_of"][key] = tid
        state["bodies"][key] = dict(body)
        state["caps"][key] = cap
        state["analyzed"][tid] = set()
        return tid

    def results(tid, contexts=False):
        key = state["tasks"][tid]
        out = []
        for pid in pools[key]:
            done = pid in state["analyzed"][tid]
            ok = done and ("_e" in pid.split("/")[-1] or "even" in pid)
            out.append({"publication_id": pid,
                        "fulltext": ({"status": "delivered"} if ok else
                                     {"status": "missing", "reason": "no_open_copy"} if done else
                                     {"status": "missing", "reason": "not_attempted"})})
        return {"publications": out}

    for k, v in (("require_probe_support", lambda r: None), ("start_search", start),
                 ("get_llm_results", results),
                 ("analyze_ids", lambda t, ids, steps=None: state["analyzed"][t].update(ids)),
                 ("wait_batch_done", lambda t, ids: results(t)),
                 ("cleanup", lambda t: state["cleaned"].append(t) or True)):
        monkeypatch.setattr(c, k, v)
    return c, state


def test_a_large_year_runs_one_probe_per_month_and_small_years_stay_whole(monkeypatch):
    """Every month with an allocation gets its own capped probe whose start body carries
    the month's inclusive ``date_range`` of January first to the month's last day; a
    month's cap is its own facet count + 50. The small year's body keeps exactly the
    legacy shape: no ``date_range`` key at all."""
    months = {1: 300, 2: 200}
    pools = {2020: [f"a{i}" for i in range(10)],
             **{(2021, m): [f"m{m}_e{i}" if i % 2 == 0 else f"m{m}_n{i}" for i in range(12)]
                for m in (1, 2)}}
    c, state = _month_client(monkeypatch, pools)
    out = c.run_year_sample(
        {"years_range": {"start_year": 2015, "end_year": 2021}, "main_dataset_name": "X"},
        allocation={2020: 2, 2021: 5}, year_counts={2020: 10, 2021: 1200},
        rng_seed=11, batch_size=30, max_batches_per_year=3,
        month_counts={2021: months}, month_allocation={2021: {1: 3, 2: 2}},
    )
    # 2020 is not large: whole-year probe, legacy body, no date_range.
    assert "date_range" not in state["bodies"][2020]
    assert state["caps"][2020] == rs.pull_cap(10) == 60
    # 2021 is sliced: one probe per allocated month, inclusive date_range of that
    # month, and no whole-year probe for it.
    assert set(state["caps"]) == {2020, (2021, 1), (2021, 2)}
    assert 2021 not in state["bodies"]
    assert state["caps"][(2021, 1)] == rs.pull_cap(300)  # 350
    assert state["caps"][(2021, 2)] == rs.pull_cap(200)  # 250
    assert state["bodies"][(2021, 1)]["date_range"] == {"date_from": "2021-01-01",
                                                           "date_to": "2021-01-31"}
    assert state["bodies"][(2021, 2)]["date_range"] == {"date_from": "2021-02-01",
                                                           "date_to": "2021-02-28"}
    kept = [p for p in out.results["publications"]
            if p.get("sample_year") == 2021 and p["fulltext"]["status"] == "delivered"]
    assert len(kept) == 5  # 3 from month 1, 2 from month 2, in each month's draw order
    for m, want in ((1, 3), (2, 2)):
        order = [i for i in rs.draw_order(pools[(2021, m)], 11, 2021, month=m)
                 if "_e" in i]
        got = [p["publication_id"] for p in kept if p["publication_id"].startswith(f"m{m}_")]
        assert got == order[:want]
    rec = out.years[2021]
    assert rec["sliced"] is True and rec["complete"] is True
    assert rec["pool_covers_year"] is True
    assert rec["kept"] == 5 and rec["allocation"] == 5
    assert rec["stop_reason"] == "target_met"
    assert set(rec["months"]) == {1, 2}
    assert rec["months"][1]["pool_covers_month"] is True
    assert sorted(state["cleaned"]) == sorted(state["tasks"])  # every probe cleaned up


def test_a_month_reaching_its_cap_and_a_missing_count_mark_the_year_incomplete(monkeypatch):
    """A month pool that fills the pull cap and a month without a facet count are both
    incompleteness: the year record never reports a uniform sample silently."""
    pools = {(2021, 1): [f"cap_n_{i}" for i in range(350)],  # exactly the cap: capped pull
             (2021, 2): [f"mis_even_{i}" if i % 2 else f"mis_n_{i}" for i in range(6)]}
    c, state = _month_client(monkeypatch, pools)
    out = c.run_year_sample(
        {"years_range": {"start_year": 2021, "end_year": 2021}, "main_dataset_name": "X"},
        allocation={2021: 4}, year_counts={2021: 1200},
        rng_seed=11, batch_size=30, max_batches_per_year=9,
        # the caller sent no count for month 2 (never happens through the optimizer:
        # it falls back then) -- the marker is what the test pins
        month_counts={2021: {1: 300}}, month_allocation={2021: {1: 2, 2: 2}},
    )
    rec = out.years[2021]
    assert rec["complete"] is False
    assert rec["incomplete"]["months_capped"] == [1]
    assert rec["incomplete"]["months_missing_count"] == [2]
    assert rec["incomplete"]["months_short"] == [1]
    assert rec["pool_covers_year"] is False
    assert rec["months"][1]["pool_covers_month"] is False  # the cap was reached
    assert rec["months"][2]["count_missing"] is True
    assert rec["months"][2]["pool_covers_month"] is True  # 6 ids below the cap of 50
    assert rec["stop_reason"] == "max_batches"  # month 1's shortfall first


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


def test_the_sample_strata_stay_inside_the_query_year_window(tmp_path):
    """A year the facet reports outside the query's own window can never enter the
    random release sample: no allocation for it, hence no probe whose seed carries it."""
    out, client = _run(tmp_path, _pubs(5, 2, 2, 3), _pubs(14, 3, 3, 1),
                       default_years_start=2016, default_years_end=2020)
    (call,) = client.sample_calls
    assert call["year_counts"] == {2020: 60}  # 2021 never a stratum
    assert call["allocation"] == {2020: 21}  # the whole allocation in the window's year
    assert out.release_sample["year_counts"] == {2020: 60}


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



@pytest.mark.parametrize("shadow", [False, True])
def test_the_release_facet_call_is_reserved_in_the_dimensions_budget(tmp_path, shadow):
    """The recall check and the gates spend calls while ``_budget_left()``; they must leave
    the year-facet call free, so a split dataset is not withheld (LOW) only because the
    run spent its Dimensions cap before the release sample."""
    uc = QueryOptimizationUseCase(
        dsl_port=YearDSL(), research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True,
                               optimize_max_dimensions_calls=10,
                               fulltext_release_shadow=shadow),
        extraction_client=FakeClient([{}]))
    uc._resource_map = None if shadow else MAP  # shadow: a non-split dataset
    uc._reserve_release_calls()
    while uc._budget_left():  # an elastic spender (recall check, gate variants)
        uc._dims_calls += 1
    assert uc._dims_calls == 9
    assert asyncio.run(uc._year_counts('\\"Example Cohort Program\\"')) == {2020: 60, 2021: 40}
    assert uc._dims_calls == 10


def test_no_call_is_reserved_when_no_random_sample_is_drawn(tmp_path):
    uc = QueryOptimizationUseCase(
        dsl_port=YearDSL(), research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True,
                               optimize_max_dimensions_calls=10),
        extraction_client=FakeClient([{}]))
    uc._resource_map = None  # non-split, default random_split, no shadow
    uc._reserve_release_calls()
    while uc._budget_left():
        uc._dims_calls += 1
    assert uc._dims_calls == 10


def test_the_year_facet_matches_the_probe_population(tmp_path):
    """The probe search (extraction ``/pipelines/start``) never filters on publication
    types -- the seed's ``publication_types`` do not reach the Dimensions where -- so on a
    non-article population (e.g. a ``book``) the facet that sizes the sample must report
    the years the probe would pull: no ``type`` filter, counts from the probe's predicate
    (the year window and the US filter it does apply)."""
    dsl = YearDSL()
    uc = QueryOptimizationUseCase(
        dsl_port=dsl, research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path, default_publication_types=["book"],
                               default_filter_us_affiliation=True),
        extraction_client=FakeClient([{}]))
    counts = asyncio.run(uc._year_counts('\\"X\\"'))
    (q,) = dsl.queries
    assert "type in" not in q  # every publication type: the probe population
    assert 'research_org_countries = "US"' in q  # ...but the US filter still applies
    assert "year in [2015:" in q and "return year limit 50" in q
    assert counts == {2020: 60, 2021: 40}


def test_a_year_outside_the_query_window_never_enters_the_facet(tmp_path):
    """The facet is clamped to the query's own years: a strayed or drifted facet year
    outside the configured window can never become a sample stratum (the per-year probes
    of the release sample run exactly on the years the allocation carries)."""
    dsl = YearDSL()
    uc = QueryOptimizationUseCase(
        dsl_port=dsl, research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path, default_years_start=2016, default_years_end=2020),
        extraction_client=FakeClient([{}]))
    counts = asyncio.run(uc._year_counts('\\"X\\"'))
    assert counts == {2020: 60}  # the facet's 2021 is outside 2016-2020: clamped away
    counts = asyncio.run(uc._year_counts('\\"X\\"', years=(2019, 2019)))
    (q,) = dsl.queries[-1:]
    assert "year in [2019:2019]" in q  # the window itself reaches the facet DSL
    assert counts == {}  # nothing of the facet lives in the query's own window


# -- large years sliced by month (the release sample's own budget) ---------------------

SLICE_CLAUSE = '\\"Example Cohort Program\\"'


class SliceDSL(CountDSL):
    """The year facet reports 2021 large (2,000 papers); every month-count call returns
    that month's count (distinct, and derived from the date window the query carries)."""

    async def execute_dsl(self, dsl):
        self.queries.append(dsl)
        if " return year " in dsl:
            return SimpleNamespace(count_total=100, year=[{"id": 2020, "count": 60},
                                                          {"id": 2021, "count": 2000}])
        window = re.search(r"date >= (\d{4})-(\d{2})-\d{2} and date <= \d{4}-\d{2}-\d{2}", dsl)
        assert window, dsl
        month = int(window.group(2))
        return SimpleNamespace(count_total=1000 + month * 10, publications=[])


def _slice_uc(tmp_path, *, month_budget, **settings):
    llm = MagicMock()
    llm.get_structured.return_value = {"exclude_terms": [], "flag_terms": [], "new_aliases": [],
                                       "reasoning": ""}
    client = YearClient([{"publications": []}], _pubs(21, 0, 0, 0))
    dsl = SliceDSL()
    uc = QueryOptimizationUseCase(
        dsl_port=dsl, research=MagicMock(), agent=llm,
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True,
                               optimize_max_iterations=1, fulltext_recall_enabled=False,
                               fulltext_target_delivered=21, fulltext_release_seed=99,
                               fulltext_release_month_count_calls=month_budget, **settings),
        extraction_client=client)

    async def label(ev, _request):
        ev.scope_ran = True
        for v in ev.verdicts:
            v.scope = "data" if v.publication_id.startswith("d") else "program_only"

    uc._apply_scope_check = label
    req = OptimizeRequest(dataset_name="Example Cohort Program",
                          dataset_names=["Example Cohort Program", "Example Cohort Study"],
                          flag_terms=["NIH"], resource_map=MAP)
    return uc, dsl, client, req


def _rnd_sample(uc, req):
    cand = QueryCandidate(main_dataset_name="Example Cohort Program",
                          safe=["Example Cohort Program", "Example Cohort Study"])
    return asyncio.run(uc._random_release_sample(cand, req, SLICE_CLAUSE, 1, decides=True))


def test_large_years_are_sliced_with_month_counts_outside_the_optimizer_budget(tmp_path):
    """A year above the whole-year count bound is sliced: twelve month-count calls run on
    their own budget (never the optimizer's), its quota is allocated across the months,
    and the sample carries month counts and allocations to the client."""
    uc, dsl, client, req = _slice_uc(tmp_path, month_budget=48)
    dims_before = uc._dims_calls
    m, info = _rnd_sample(uc, req)
    assert m is not None and len(client.sample_calls) == 1
    kw = client.sample_calls[0]
    assert set(kw["month_counts"]) == {2021} and len(kw["month_counts"][2021]) == 12
    assert set(kw["month_allocation"]) == {2021}
    ma = kw["month_allocation"][2021]
    allocation = info["allocation"]
    assert allocation[2021] > 0 and sum(ma.values()) == allocation[2021]
    # largest remainder over the months: the biggest month's share is never below a
    # proportional share minus one, and the small months stay in proportion
    counts = kw["month_counts"][2021]
    assert ma[max(counts, key=lambda m: counts[m])] == max(ma.values())
    # the twelve month-count calls do not eat the optimize budget (only the facet does)
    assert uc._month_count_calls == 12 and uc._dims_calls - dims_before == 1
    month_qs = [q for q in dsl.queries if "date >=" in q]
    assert len(month_qs) == 12 and all("2021-" in q for q in month_qs)
    # the month counts count the probe population: no publication-type filter, and the
    # US filter the probe applies (keep the facet invariant of cf325e0)
    assert all("type in" not in q for q in month_qs)
    uc_usecase = QueryOptimizationUseCase
    slicing = info["slicing"]
    assert slicing["month_count_calls"] == 12 and slicing["month_count_budget"] == 48
    assert list(slicing["years_sliced"]) == [2021]
    assert slicing["degraded_years"] == {}
    note = uc_usecase._release_sample_note(info)
    assert "2021 sliced by month" in note


def test_a_month_budget_spent_mid_year_degrades_the_year_to_a_whole_year_probe(tmp_path):
    """When the month budget runs out before a year's twelve counts, the year falls back
    to its whole-year probe and the run record marks the degraded coverage -- never a
    silent uniform sample."""
    uc, dsl, client, req = _slice_uc(tmp_path, month_budget=5)
    m, info = _rnd_sample(uc, req)
    assert m is not None and len(client.sample_calls) == 1
    kw = client.sample_calls[0]
    assert kw["month_allocation"] == {} and kw["month_counts"] == {}
    assert uc._month_count_calls == 5
    slicing = info["slicing"]
    assert slicing["years_sliced"] == {}
    assert "5 of 12" in slicing["degraded_years"][2021]
    # the degraded marker is inspectable in the run record's own year structure
    yr = info["years"][2021]
    assert yr["coverage"] == "degraded_whole_year"
    assert "degraded_reason" in yr
    note = QueryOptimizationUseCase._release_sample_note(info)
    assert "2021" in note and "whole-year" in note

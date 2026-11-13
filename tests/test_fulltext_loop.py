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


def pub(pid, cites, used, quote="", status="delivered", mentioned=None):
    mentioned = used if mentioned is None else mentioned
    return {
        "publication_id": pid, "doi": f"10/{pid}", "title": f"T {pid}", "citation_count": cites,
        "fulltext": {"status": status, "source": "pmc", "reason": None}, "analyzed": True,
        "validated_datasets": [{"name": "X", "is_used": used, "is_mentioned": mentioned, "text_quote": quote}],
    }


class FakeClient:
    def __init__(self, results_seq, stop_reason="target_met", batches=1):
        self.results_seq = list(results_seq)
        self.seeds: list[dict] = []
        self.kwargs: list[dict] = []
        self.stop_reason = stop_reason
        self.batches = batches

    def run_probe_topup(self, seed, *, target, batch_size, max_batches, pull_cap):
        from dataset_agent.adapters.extraction_client import TopupResult, is_attempted, is_delivered

        self.seeds.append(seed)
        self.kwargs.append(dict(target=target, batch_size=batch_size, max_batches=max_batches,
                                pull_cap=pull_cap))
        r = self.results_seq.pop(0) if len(self.results_seq) > 1 else self.results_seq[0]
        if isinstance(r, Exception):
            raise r
        pubs = r.get("publications") or []
        return TopupResult(results=r, batches=self.batches,
                           attempted=sum(1 for p in pubs if is_attempted(p)),
                           delivered=sum(1 for p in pubs if is_delivered(p)),
                           target=target, stop_reason=self.stop_reason)


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
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True),
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
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True,
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


def test_proposal_created_and_persisted(tmp_path):
    from dataset_agent.adapters.proposals_json import JsonProposalRepository

    bad = [pub(f"n{i}", 50 - i, False, "Another Long Title Here nursing") for i in range(4)] + [
        pub(f"g{i}", 30 - i, True, "My Long Dataset Alpha data") for i in range(4)]
    good = [pub(f"g{i}", 30 - i, True, "My Long Dataset Alpha data") for i in range(4)]
    llm = MagicMock()
    llm.get_structured.return_value = {"exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""}
    s = make_settings(tmp_path, fulltext_refinement_enabled=True)
    repo = JsonProposalRepository(s.output_dir)
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=llm, settings=s,
                                  extraction_client=FakeClient([{"publications": bad}, {"publications": good}]),
                                  proposal_repo=repo)
    out = asyncio.run(uc.execute(REQ))
    p = out.proposal
    assert p is not None and repo.get(p.id) is not None
    assert [d.action for d in p.diffs if d.term == "Another Long Title Here"] == ["drop"]
    assert p.start_payload["probe"] is False
    assert "Another Long Title Here" in p.initial.seed["standalone_names"] + p.initial.seed["dataset_names"]
    assert "Another Long Title Here" not in p.start_payload["standalone_names"] + p.start_payload["dataset_names"]
    assert [m.iteration for m in p.iterations] == [0, 1]


def test_unbuildable_proposal_does_not_abort_the_run(tmp_path, monkeypatch):
    def boom(**_kw):
        raise ValueError("raw dimensions_query does not contain aliases: ['The \"Big\" Survey']")

    monkeypatch.setattr(qopt_mod, "build_proposal", boom)
    good = [pub(f"g{i}", 30 - i, True, "My Long Dataset Alpha data") for i in range(4)]
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True),
        extraction_client=FakeClient([{"publications": good}]))
    out = asyncio.run(uc.execute(REQ))
    assert out.success and out.proposal is None
    assert any("proposal skipped" in n for n in (out.notes if isinstance(out.notes, list) else [out.notes or ""]))


def test_failed_gate_evaluation_is_not_zero_fp(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync",
        lambda *a, **k: {"fp_rate_pct": 0.0, "genuine_count": 0, "matched_count": 0, "total_pubs": 0,
                         "error": "dimensions_query_failed: budget"},
    )
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=make_settings(tmp_path))
    out = asyncio.run(uc.execute(REQ))
    assert "~gate" not in out.selected_variant
    assert out.confidence.value.lower() == "low"
    assert "evaluation failed" in str(out.notes)


def test_gate_iteration_dimensions_failure_is_noted_and_never_selected(tmp_path, monkeypatch):
    calls = {"n": 0}
    ok = {"fp_rate_pct": 5.0, "genuine_count": 10, "fp_count": 0, "matched_count": 10,
          "total_pubs": 100, "coverage_pct": 100.0, "publications_total": 100, "fp_hits": []}

    def fake_eval(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return dict(ok, for_clause=k.get("for_clause", ""))
        return {
            "fp_rate_pct": 0.0, "genuine_count": 0, "matched_count": 0, "total_pubs": 0,
            "error": "dimensions_query_failed: Dimensions rejected the query: {'query': {...}}",
            "error_reason": "Service unavailable: Unexpected Error 'NoneType'",
        }

    monkeypatch.setattr(qopt_mod, "evaluate_for_clause_sync", fake_eval)

    def fp(pid, cites):
        p = pub(pid, cites, False, "")
        p["validated_datasets"] = []
        p["term_contexts"] = [{"term": "Another Long Title Here", "role": "alias", "part": "body",
                               "text": "Another Long Title Here in the nursing ward"}]
        p["category_for"] = ["Nursing"]
        return p

    first = [fp(f"n{i}", 50 - i) for i in range(3)] + [
        pub(f"g{i}", 30 - i, True, "My Long Dataset Alpha data") for i in range(3)]
    llm = MagicMock()
    llm.get_structured.return_value = {
        "false_positive_groups": [{"label": "nursing wards", "cause_term": "Another Long Title Here",
                                   "papers": 3, "exclude_terms": ["nursing ward"]}],
        "exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""}
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=llm,
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True, optimize_max_iterations=3,
                               refine_min_alias_papers=10),
        extraction_client=FakeClient([{"publications": first}]))
    out = asyncio.run(uc.execute(REQ))
    notes = str(out.notes)
    assert "gate_iter1: Dimensions query failed (Service unavailable" in notes
    assert "iteration not evaluated" in notes
    assert "search publications" not in notes
    assert out.selected_variant.endswith("~gate0")
    assert not any(v.label.endswith("~gate1") for v in out.all_variants_tested)
    last = out.fulltext_iterations[-1]
    assert last.iteration == 1 and last.error and last.fulltext_precision is None
    assert out.confidence.value == "low"
    assert any("Gate iteration 1 could not be evaluated" in n for n in out.proposal.notes)


def test_small_fulltext_sample_does_not_win_selection(tmp_path):
    bad = [pub(f"n{i}", 50 - i, False, "Another Long Title Here nursing") for i in range(4)] + [
        pub(f"g{i}", 30 - i, True, "My Long Dataset Alpha data") for i in range(4)]
    one = [pub("g0", 30, True, "My Long Dataset Alpha data")]
    llm = MagicMock()
    llm.get_structured.return_value = {"exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""}
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=llm,
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True,
                               optimize_max_iterations=3, refine_min_alias_papers=3,
                               fulltext_min_evidence=3),
        extraction_client=FakeClient([{"publications": bad}, {"publications": one}]))
    out = asyncio.run(uc.execute(REQ))
    assert [m.n_evaluated for m in out.fulltext_iterations][:2] == [8, 1]
    assert out.selected_variant.endswith("~gate0")
    assert "n_evaluated=1" in str(out.notes)
    # A 1-paper sample must not trigger the early stop.
    assert "gate_iter1: full-text precision within threshold." not in str(out.notes)


@pytest.fixture
def _zero_abstract_fp(monkeypatch):
    """Abstract evaluator says 0% FP, which alone would mean HIGH confidence."""
    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync",
        lambda *a, **k: {"fp_rate_pct": 0.0, "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                         "total_pubs": 100, "coverage_pct": 100.0, "for_clause": k.get("for_clause", ""),
                         "publications_total": 100, "fp_hits": []},
    )


def _ft_uc(tmp_path, results_seq, **kw):
    llm = MagicMock()
    llm.get_structured.return_value = {"exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""}
    return QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=llm,
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True,
                               optimize_max_iterations=1, **kw),
        extraction_client=FakeClient(results_seq))


def test_confidence_follows_low_fulltext_precision(tmp_path, _zero_abstract_fp):
    bad = [pub(f"n{i}", 50 - i, False, "My Long Dataset Alpha mentioned") for i in range(3)]
    out = asyncio.run(_ft_uc(tmp_path, [{"publications": bad}]).execute(REQ))
    assert out.fulltext_iterations[0].fulltext_precision == 0.0
    assert out.confidence.value == "low"


def test_confidence_high_when_fulltext_precision_meets_threshold(tmp_path, _zero_abstract_fp):
    good = [pub(f"g{i}", 50 - i, True, "My Long Dataset Alpha data") for i in range(3)]
    out = asyncio.run(_ft_uc(tmp_path, [{"publications": good}]).execute(REQ))
    assert out.confidence.value == "high"


def test_confidence_low_below_min_evidence(tmp_path, _zero_abstract_fp):
    one = [pub("g0", 50, True, "My Long Dataset Alpha data")]
    out = asyncio.run(_ft_uc(tmp_path, [{"publications": one}]).execute(REQ))
    assert out.fulltext_iterations[0].n_evaluated == 1
    assert out.confidence.value == "low"
    assert any("below the minimum sample" in n for n in out.proposal.notes)


def test_confidence_low_when_probe_fails(tmp_path, _zero_abstract_fp):
    from dataset_agent.adapters.extraction_client import ExtractionError

    out = asyncio.run(_ft_uc(tmp_path, [ExtractionError("down")]).execute(REQ))
    assert out.fulltext_iterations[0].error == "down"
    assert out.confidence.value == "low"


def test_topup_settings_reach_the_client_and_shortfall_is_noted(tmp_path, _zero_abstract_fp):
    few = [pub(f"g{i}", 50 - i, True, "My Long Dataset Alpha data") for i in range(3)] + [
        pub("m0", 10, True, "", status="missing")]
    client = FakeClient([{"publications": few}], stop_reason="pool_exhausted", batches=2)
    llm = MagicMock()
    llm.get_structured.return_value = {"exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""}
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=llm,
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True, optimize_max_iterations=1,
                               fulltext_target_delivered=30, fulltext_batch_size=25,
                               fulltext_max_batches=4, probe_pull_cap=150),
        extraction_client=client)
    out = asyncio.run(uc.execute(REQ))
    assert client.kwargs[0] == dict(target=30, batch_size=25, max_batches=4, pull_cap=150)
    m = out.fulltext_iterations[0]
    assert (m.n_publications, m.n_fulltext_delivered, m.n_evaluated) == (4, 3, 3)
    assert (m.fulltext_target, m.fulltext_batches, m.fulltext_stop_reason) == (30, 2, "pool_exhausted")
    assert "full-text shortfall: 3/30 papers with full text (pool_exhausted" in str(out.notes)
    assert any("shortfall" in n for n in out.proposal.notes)


def test_exclude_impact_is_counted_in_dimensions_and_patterns_reach_the_proposal(tmp_path):
    def fp(pid, cites):
        p = pub(pid, cites, False, "")
        p["validated_datasets"] = []
        p["term_contexts"] = [{"term": "Another Long Title Here", "role": "alias", "part": "body",
                               "text": "Another Long Title Here in the nursing ward"}]
        p["category_for"] = ["Nursing"]
        return p

    first = [fp(f"n{i}", 50 - i) for i in range(3)] + [
        pub(f"g{i}", 30 - i, True, "My Long Dataset Alpha data") for i in range(3)]
    good = [pub(f"g{i}", 30 - i, True, "My Long Dataset Alpha data") for i in range(4)]
    llm = MagicMock()
    llm.get_structured.return_value = {
        "false_positive_groups": [{"label": "nursing wards", "cause_term": "Another Long Title Here",
                                   "papers": 3, "exclude_terms": ["nursing ward"]}],
        "exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""}
    dsl = CountDSL()
    uc = QueryOptimizationUseCase(
        dsl_port=dsl, research=MagicMock(), agent=llm,
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True, optimize_max_iterations=2,
                               refine_min_alias_papers=10),
        extraction_client=FakeClient([{"publications": first}, {"publications": good}]))
    out = asyncio.run(uc.execute(REQ))
    d = next(x for x in out.proposal.diffs if x.kind == "exclude" and x.term == "nursing ward")
    assert d.action == "add" and d.stats["removes"] == 100
    assert any('AND \\"nursing ward\\"' in q for q in dsl.queries)
    notes = " ".join(out.proposal.notes)
    assert "false-positive group 'nursing wards'" in notes and "'Nursing'" in notes


def test_impact_checks_stop_at_the_budget_reserve(tmp_path):
    import asyncio as _a

    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path, optimize_max_dimensions_calls=20, refine_impact_budget_reserve=8))
    uc._dims_calls = 11
    assert _a.run(uc._exclude_impact("A", "b")) == 100
    assert uc._dims_calls == 12
    with pytest.raises(Exception, match="reserve"):
        _a.run(uc._exclude_impact("A", "b"))


def test_refine_count_raises_at_the_budget_reserve_instead_of_returning_zero(tmp_path):
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path, optimize_max_dimensions_calls=20, refine_impact_budget_reserve=8))
    uc._dims_calls = 11
    assert asyncio.run(uc._refine_count("b")) == 100
    assert uc._dims_calls == 12
    with pytest.raises(Exception, match="reserve"):
        asyncio.run(uc._refine_count("b"))
    # Phase 1's counter is unchanged: it returns 0 only when the whole budget is gone.
    uc._dims_calls = 20
    assert asyncio.run(uc._count("b")) == 0


def test_backtick_placeholder_alias_never_reaches_the_probe(tmp_path):
    req = OptimizeRequest(dataset_name="My Long Dataset Alpha",
                          dataset_names=["My Long Dataset Alpha", "All `` survey"], flag_terms=["NIH"])
    good = [pub(f"g{i}", 30 - i, True, "My Long Dataset Alpha data") for i in range(4)]
    client = FakeClient([{"publications": good}])
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True), extraction_client=client)
    out = asyncio.run(uc.execute(req))
    seed = client.seeds[0]
    assert "All `` survey" not in str(seed) and "All survey" not in str(seed)
    assert "All `` survey" not in out.aliases.safe + out.aliases.risky
    assert "All survey" not in out.aliases.safe + out.aliases.risky


# --- Recall check ------------------------------------------------------------------------

RC_NAME = "Beta Health Survey"
RC_FLAGS = ["Beta Health Survey Research Program", "NIH"]
RC_REQ = OptimizeRequest(dataset_name=RC_NAME, dataset_names=[RC_NAME], flag_terms=RC_FLAGS)


class RecallDSL(CountDSL):
    """Totals: complement queries (contain NOT) return *not_total*; a query holding
    ``risky_marker`` (and no NOT) is huge so that alias is classified risky."""

    def __init__(self, not_total=100, risky_marker=None, fail_not=False):
        super().__init__()
        self.not_total, self.risky_marker, self.fail_not = not_total, risky_marker, fail_not

    async def execute_dsl(self, dsl):
        self.queries.append(dsl)
        if " NOT " in dsl:
            if self.fail_not:
                from dataset_agent.adapters.dimensions_dsl import DimensionsDslError

                raise DimensionsDslError("QuerySyntaxError: bad", errors={"x": "QuerySyntaxError"})
            return SimpleNamespace(count_total=self.not_total, publications=[])
        if self.risky_marker and self.risky_marker in dsl:
            return SimpleNamespace(count_total=5000, publications=[])
        return SimpleNamespace(count_total=100, publications=[])

    def n_not(self):
        return sum(1 for q in self.queries if " NOT " in q)


def comp_pubs(term, n, genuine, tag="c"):
    return [pub(f"{tag}{i}", 40 - i, i < genuine, f"studied with the {term} data") for i in range(n)]


def gate_pubs():
    return [pub(f"g{i}", 30 - i, True, f"{RC_NAME} data") for i in range(4)]


def recall_uc(tmp_path, results, dsl=None, request_settings=None, **kw):
    client = FakeClient(results, **kw)
    dsl = dsl or RecallDSL()
    s = make_settings(tmp_path, fulltext_refinement_enabled=True, **(request_settings or {}))
    uc = QueryOptimizationUseCase(dsl_port=dsl, research=MagicMock(), agent=MagicMock(), settings=s,
                                  extraction_client=client)
    return uc, client, dsl


def run_recall(tmp_path, results, req=RC_REQ, **kw):
    uc, client, dsl = recall_uc(tmp_path, results, **kw)
    return asyncio.run(uc.execute(req)), client, dsl


PROGRAM = "Beta Health Survey Research Program"


def test_recall_promotes_flag_naming_the_dataset(tmp_path):
    import re

    out, client, dsl = run_recall(tmp_path, [{"publications": gate_pubs()},
                                              {"publications": comp_pubs(PROGRAM, 10, 9)}],
                                  request_settings=dict(fulltext_recall_max_candidates=1))
    assert out.success and len(client.seeds) == 2
    comp = client.seeds[1]
    assert PROGRAM in comp["dataset_names"] and "Beta Health Survey" in str(comp["dimensions_query"])
    assert client.kwargs[1]["target"] == 20
    assert PROGRAM in out.aliases.safe and PROGRAM in out.for_clause
    assert "NIH" in out.flag_terms or out.flag_terms
    assert out.selected_variant.endswith("+recall")
    p = out.proposal
    assert PROGRAM in p.proposed.safe and PROGRAM in str(p.start_payload)
    d = next(d for d in p.diffs if d.term == PROGRAM)
    assert d.action == "add" and d.kind == "alias" and d.evidence
    assert re.match(r"^recall: complement precision 0\.90 on 10 papers; \+100 Dimensions hits", d.reason)
    assert "recall check: not eligible: NIH (short acronym)" in out.notes
    assert "measured before the recall addition" in out.notes
    assert "Selected gate-evaluated variant" not in out.notes


def test_recall_large_expansion_needs_lower_bound(tmp_path):
    # Selected query counts 100; the complement adds 500 (5x), so 17/20 (lower bound ~0.64)
    # is too uncertain even though the point estimate clears 0.80.
    out, _, _ = run_recall(tmp_path, [{"publications": gate_pubs()},
                                      {"publications": comp_pubs(PROGRAM, 20, 17)}],
                           dsl=RecallDSL(not_total=500),
                           request_settings=dict(fulltext_recall_max_candidates=1))
    assert PROGRAM not in out.aliases.safe and not out.selected_variant.endswith("+recall")
    assert (f"recall check: '{PROGRAM}' rejected: complement precision 0.85 on 20 papers is too uncertain "
            "for +500 hits on a query of 100 (lower bound 0.64, need 0.70)") in out.notes


def test_recall_rejects_low_complement_precision(tmp_path):
    out, client, _ = run_recall(tmp_path, [{"publications": gate_pubs()},
                                            {"publications": comp_pubs(PROGRAM, 10, 5)}])
    assert PROGRAM not in out.aliases.safe and not out.selected_variant.endswith("+recall")
    assert f"recall check: '{PROGRAM}' rejected: complement precision 0.50 on 10 papers (need 0.80)" in out.notes
    assert any("complement precision 0.50" in n for n in out.proposal.notes)


def test_recall_shortfall_is_reported_not_promoted(tmp_path):
    uc, client, _ = recall_uc(tmp_path, [{"publications": gate_pubs()},
                                         {"publications": comp_pubs(PROGRAM, 4, 4)}])
    client.stop_reason = "pool_exhausted"
    out = asyncio.run(uc.execute(RC_REQ))
    assert PROGRAM not in out.aliases.safe
    assert "only 4 full-text paper(s) evaluated (need 10)" in out.notes


def test_recall_disabled_runs_no_extra_probe(tmp_path):
    out, client, dsl = run_recall(tmp_path, [{"publications": gate_pubs()}],
                                  request_settings=dict(fulltext_recall_enabled=False))
    assert len(client.seeds) == 1 and dsl.n_not() == 0 and "recall check" not in out.notes


def test_recall_candidate_cap_and_additive_second_candidate(tmp_path):
    other = "Beta Health Survey Data Center"
    third = "Beta Health Survey Consortium"
    req = OptimizeRequest(dataset_name=RC_NAME, dataset_names=[RC_NAME], flag_terms=[PROGRAM, other, third])
    out, client, dsl = run_recall(tmp_path, [{"publications": gate_pubs()},
                                              {"publications": comp_pubs(PROGRAM, 10, 10, "a")},
                                              {"publications": comp_pubs(other, 10, 10, "b")}], req=req)
    assert len(client.seeds) == 3
    # second complement is measured beyond the first promoted alias
    assert PROGRAM in str(client.seeds[2]["dimensions_query"]) and other in client.seeds[2]["dataset_names"]
    assert {PROGRAM, other} <= set(out.aliases.safe) and third not in out.aliases.safe
    assert f"recall check: '{third}' not evaluated: over the candidate cap" in out.notes
    assert dsl.n_not() == 3 + 1  # three counts + one recount for the second candidate


def test_recall_small_gain_skips_probe(tmp_path):
    out, client, _ = run_recall(tmp_path, [{"publications": gate_pubs()}], dsl=RecallDSL(not_total=4))
    assert len(client.seeds) == 1
    assert f"recall check: '{PROGRAM}' rejected: adds only 4 Dimensions hits" in out.notes


def test_recall_too_broad_candidate_is_not_probed(tmp_path):
    out, client, _ = run_recall(tmp_path, [{"publications": gate_pubs()}], dsl=RecallDSL(not_total=600_000))
    assert len(client.seeds) == 1
    assert f"recall check: '{PROGRAM}' rejected: 600000 hits, too broad" in out.notes


def test_recall_stops_on_dimensions_budget(tmp_path):
    uc, client, dsl = recall_uc(tmp_path, [{"publications": gate_pubs()}])
    orig = uc._recall_check

    async def wrapped(*a, **k):
        uc._dims_calls = 10**6  # budget gone when the recall check starts
        return await orig(*a, **k)

    uc._recall_check = wrapped
    uc._settings.optimize_max_dimensions_calls = 10**6 - 1
    out = asyncio.run(uc.execute(RC_REQ))
    assert out.success and "recall check stopped: Dimensions call budget exhausted" in out.notes
    assert len(client.seeds) == 1 and PROGRAM not in out.aliases.safe


def test_recall_syntax_error_and_probe_failure_are_noted_not_fatal(tmp_path):
    from dataset_agent.adapters.extraction_client import ExtractionError

    out, client, _ = run_recall(tmp_path, [{"publications": gate_pubs()}], dsl=RecallDSL(fail_not=True))
    assert out.success and "Dimensions rejected the complement query" in out.notes
    out, client, _ = run_recall(tmp_path, [{"publications": gate_pubs()}, ExtractionError("down")])
    assert out.success and f"recall check: '{PROGRAM}' not evaluated: down" in out.notes
    assert PROGRAM not in out.aliases.safe


def test_recall_caps_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync",
        lambda *a, **k: {"fp_rate_pct": 0.5, "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                         "total_pubs": 100, "coverage_pct": 100.0, "for_clause": k.get("for_clause", ""),
                         "publications_total": 100, "fp_hits": []},
    )
    low, _, _ = run_recall(tmp_path, [{"publications": gate_pubs()},
                                      {"publications": comp_pubs(PROGRAM, 10, 8)}],
                           request_settings=dict(fulltext_recall_max_candidates=1))
    assert PROGRAM in low.aliases.safe and low.confidence.value == "medium"
    high, _, _ = run_recall(tmp_path, [{"publications": gate_pubs()},
                                       {"publications": comp_pubs(PROGRAM, 20, 20)}],
                            request_settings=dict(fulltext_recall_max_candidates=1))
    assert PROGRAM in high.aliases.safe and high.confidence.value == "high"


def test_recall_skipped_without_selected_fulltext_iteration(tmp_path):
    from dataset_agent.adapters.extraction_client import ExtractionError

    out, client, _ = run_recall(tmp_path, [ExtractionError("down")])
    assert out.success and "recall check skipped" in out.notes and len(client.seeds) == 1
    out, client, _ = run_recall(tmp_path, [{"publications": gate_pubs()[:1]}])
    assert "recall check skipped: no full-text evidence" in out.notes


def test_recall_promotes_risky_alias_out_of_the_flag_gate(tmp_path):
    risky = "Beta Health Survey Network"
    req = OptimizeRequest(dataset_name=RC_NAME, dataset_names=[RC_NAME, risky], flag_terms=[PROGRAM])
    out, client, _ = run_recall(
        tmp_path, [{"publications": gate_pubs()}, {"publications": comp_pubs(risky, 10, 10)}],
        req=req, dsl=RecallDSL(risky_marker="Network"),
        request_settings=dict(fulltext_recall_max_candidates=1))
    # cap 1: only the flag (tier 0) is probed, as a hybrid raw complement; the alias stays gated
    assert out.aliases.risky == [risky] and PROGRAM in out.aliases.safe
    assert "AND" in client.seeds[1]["dimensions_query"]
    out2, client2, _ = run_recall(
        tmp_path, [{"publications": gate_pubs()}, {"publications": comp_pubs(PROGRAM, 10, 4, "x")},
                   {"publications": comp_pubs(risky, 10, 10, "y")}],
        req=req, dsl=RecallDSL(risky_marker="Network"))
    assert risky in out2.aliases.safe and risky not in out2.aliases.risky
    d = next(d for d in out2.proposal.diffs if d.term == risky)
    assert d.action == "add" and d.reason.endswith("(was flag-gated)") and d.stats["source"] == "alias"

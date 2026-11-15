"""Round 5 (owner decision "Broad, bar on 'unrelated'"): for a dataset the resource map
splits, the query is released when the unrelated share is at most
``fulltext_split_unrelated_max`` (0.20); data, program-only and references-only papers all
count toward the bar. Every other dataset keeps the precision bar
(``fulltext_recall_min_precision``) exactly as before, and the minimum sample
(``fulltext_recall_min_evidence``) applies to both bars."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dataset_agent.adapters.fulltext_evidence import Evidence, PubVerdict
from dataset_agent.application import query_optimization as qopt_mod
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import OptimizeRequest, ResourceMap
from tests.test_fulltext_loop import CountDSL, FakeClient, make_settings

MAP = ResourceMap(program_names=["Example Cohort Program"],
                  data_resource_names=["Example Researcher Workbench"])


def _pub(pid, kind):
    """kind: d (used, data) | p (mentioned, program only) | r (references only) | u (unrelated)."""
    part = "references" if kind == "r" else "body"
    return {
        "publication_id": pid, "doi": None, "title": f"T {pid}", "citation_count": 1000 - int(pid[1:]),
        "fulltext": {"status": "delivered", "source": "pmc", "reason": None}, "analyzed": True,
        "validated_datasets": [{"name": "X", "is_used": kind == "d",
                                "is_mentioned": kind in ("d", "p"), "text_quote": ""}],
        "term_contexts": [{"term": "Example Cohort Program", "role": "alias", "part": part,
                           "text": "Example Cohort Program"}],
    }


def _pubs(n_data, n_prog, n_refs, n_unrel):
    out, i = [], 0
    for kind, k in (("d", n_data), ("p", n_prog), ("r", n_refs), ("u", n_unrel)):
        for _ in range(k):
            out.append(_pub(f"{kind}{i}", kind))
            i += 1
    return out


@pytest.fixture(autouse=True)
def _zero_abstract_fp(monkeypatch):
    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync",
        lambda *a, **k: {"fp_rate_pct": 0.0, "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                         "total_pubs": 100, "coverage_pct": 100.0, "for_clause": k.get("for_clause", ""),
                         "publications_total": 100, "fp_hits": []},
    )


def _run(tmp_path, pubs, *, split, **settings):
    llm = MagicMock()
    llm.get_structured.return_value = {"exclude_terms": [], "flag_terms": [], "new_aliases": [],
                                       "reasoning": ""}
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=llm,
        settings=make_settings(tmp_path, fulltext_refinement_enabled=True,
                               optimize_max_iterations=1, fulltext_recall_enabled=False,
                               **settings),
        extraction_client=FakeClient([{"publications": pubs}]))

    async def label(ev, _request):
        # Deterministic stand-in for the scope check: the label is in the paper id.
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
    return asyncio.run(uc.execute(req))


def _fv(pid, verdict, *, scope=None, part="body", refs_rule=False, overridden_from=None):
    return PubVerdict(publication_id=pid, doi=None, title="", citation_count=0,
                      verdict=verdict, quote="", scope=scope, scope_references_only=refs_rule,
                      overridden_from=overridden_from,
                      contexts=[{"term": "X", "part": part, "text": "X"}])


def test_four_way_split_is_canonical_on_the_verdict_and_evidence():
    ev = Evidence(verdicts=[
        _fv("a", "used", scope="data"), _fv("b", "mentioned", scope="program_only"),
        _fv("c", "mentioned", scope="unclear"),  # genuine, not program-only: data
        # turned by the references-only override
        _fv("d", "not_mentioned", part="references", overridden_from="mentioned"),
        _fv("e", "not_mentioned", refs_rule=True,
            overridden_from="mentioned"),  # turned by the scope check's rule
        _fv("f", "not_mentioned"),
        # The judge itself said not_mentioned; its snippets sit in the references (a
        # homonym in a cited title): unrelated, references-only only in the legacy split.
        _fv("g", "not_mentioned", part="references"),
    ])
    assert [x.four_way for x in ev.verdicts] == [
        "data", "program_only", "data", "references_only", "references_only", "unrelated",
        "unrelated"]
    assert ev.four_way_counts() == {"data": 2, "program_only": 1, "references_only": 2,
                                    "unrelated": 2}
    assert [x.four_way_legacy for x in ev.verdicts] == [
        "data", "program_only", "data", "references_only", "references_only", "unrelated",
        "references_only"]
    assert ev.four_way_counts(legacy=True) == {"data": 2, "program_only": 1,
                                               "references_only": 3, "unrelated": 1}


def test_unrelated_share_is_reported_both_ways():
    ev = Evidence(scope_ran=True, verdicts=[
        *[_fv(f"a{i}", "used", scope="data") for i in range(6)],
        _fv("d", "not_mentioned", part="references", overridden_from="mentioned"),
        *[_fv(f"g{i}", "not_mentioned", part="references") for i in range(2)],
        _fv("f", "not_mentioned"),
    ])
    m = ev.metrics(0, "q", None)
    assert m.unrelated_share == 0.3 and m.unrelated_share_legacy == 0.1
    assert m.four_way["references_only"]["count"] == 1
    assert ev.four_way_summary(legacy=True)["references_only"]["count"] == 3
    # No scope check: neither share is reported.
    plain = Evidence(verdicts=[_fv("g", "not_mentioned", part="references")])
    assert plain.metrics(0, "q", None).unrelated_share_legacy is None


def test_split_dataset_released_on_unrelated_share(tmp_path):
    # data 0.45, program-only 0.30, references-only 0.13, unrelated 0.12: precision 0.75
    # fails the precision bar, the unrelated bar releases it.
    out = _run(tmp_path, _pubs(45, 30, 13, 12), split=True)
    m = out.fulltext_iterations[0]
    assert m.fulltext_precision == 0.75 and m.data_share == 0.45
    assert m.unrelated_share == 0.12
    assert out.release_bar == "split dataset: unrelated share <= 0.20"
    assert out.release_bar_met is True
    assert out.confidence.value == "medium"
    assert out.fulltext_data_precision == 0.45
    fw = out.fulltext_four_way
    assert {k: fw[k]["count"] for k in fw} == {"data": 45, "program_only": 30,
                                               "references_only": 13, "unrelated": 12}
    assert fw["unrelated"]["share"] == 0.12
    lo, hi = fw["unrelated"]["wilson95"]
    assert 0.0 < lo < 0.12 < hi < 1.0
    assert "release bar: split dataset: unrelated share <= 0.20 (met)" in out.notes
    assert "data-only precision 0.45" in out.notes and "precision 0.75" in out.notes
    assert "legacy unrelated share 0.12" in out.notes


def test_split_dataset_not_released_above_the_unrelated_bar(tmp_path):
    out = _run(tmp_path, _pubs(45, 30, 0, 25), split=True)
    assert out.fulltext_iterations[0].unrelated_share == 0.25
    assert out.release_bar == "split dataset: unrelated share <= 0.20"
    assert out.release_bar_met is False
    assert out.confidence.value == "low"
    assert "release bar: split dataset: unrelated share <= 0.20 (not met)" in out.notes


def test_non_split_dataset_keeps_the_precision_bar(tmp_path):
    # Same papers without a map: precision 0.75 < 0.80 is not released, as before.
    out = _run(tmp_path, _pubs(45, 30, 13, 12), split=False)
    m = out.fulltext_iterations[0]
    assert m.fulltext_precision == 0.75
    assert m.unrelated_share is None and m.four_way is None
    assert out.release_bar == "precision >= 0.80"
    assert out.release_bar_met is False
    assert out.confidence.value == "low"
    assert out.fulltext_four_way is None and out.fulltext_data_precision is None
    # At 0.80 the precision bar releases it (MEDIUM), as before.
    out = _run(tmp_path, _pubs(50, 30, 10, 10), split=False)
    assert out.release_bar_met is True and out.confidence.value == "medium"


def test_split_bar_keeps_the_minimum_sample(tmp_path):
    # 8 papers, none unrelated: below fulltext_recall_min_evidence (10), not released.
    out = _run(tmp_path, _pubs(4, 4, 0, 0), split=True)
    assert out.fulltext_iterations[0].n_evaluated == 8
    assert out.fulltext_iterations[0].unrelated_share == 0.0
    assert out.release_bar_met is False
    assert out.confidence.value == "low"
    assert "below the minimum sample" in out.notes


def test_citers_only_sample_meets_the_split_bar_with_a_warning(tmp_path):
    # 8 references-only citers + 2 unrelated: the unrelated share 0.20 meets the bar (no
    # floor on the data share, an owner question), but the notes say no paper is data.
    out = _run(tmp_path, _pubs(0, 0, 8, 2), split=True)
    m = out.fulltext_iterations[0]
    assert m.unrelated_share == 0.2 and m.data_share == 0.0
    assert out.release_bar_met is True and out.confidence.value == "medium"
    assert ("gate_iter0: full-text unrelated share meets the split-dataset bar (<= 0.20 on "
            ">= 10 papers); data-only precision 0.00") in out.notes
    assert "no sampled paper refers to the data resource" in out.notes


def test_no_warning_when_a_sampled_paper_is_data(tmp_path):
    out = _run(tmp_path, _pubs(1, 0, 7, 2), split=True)
    assert out.release_bar_met is True
    assert "no sampled paper refers to the data resource" not in out.notes


def _metrics(it, n_data, n_refs, n_unrel, precision=None):
    from dataset_agent.domain.proposals import IterationMetrics

    n = n_data + n_refs + n_unrel
    return IterationMetrics(
        iteration=it, for_clause=f"q{it}", n_publications=n, n_fulltext_delivered=n,
        n_evaluated=n, fulltext_precision=precision if precision is not None else n_data / n,
        data_share=n_data / n, unrelated_share=n_unrel / n,
        four_way={"data": {"count": n_data}, "program_only": {"count": 0},
                  "references_only": {"count": n_refs}, "unrelated": {"count": n_unrel}})


def test_selection_prefers_the_higher_data_share_among_iterations_meeting_the_split_bar(tmp_path):
    uc = QueryOptimizationUseCase(
        dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
        settings=make_settings(tmp_path), extraction_client=FakeClient([]))
    uc._resource_map = MAP
    low_data = _metrics(0, 1, 8, 1, precision=0.95)  # bar met, data 0.10
    high_data = _metrics(1, 6, 2, 2, precision=0.60)  # bar met, data 0.60
    missed = _metrics(2, 9, 0, 3)  # unrelated 0.25: bar missed, data 0.75
    best = max([low_data, high_data, missed], key=uc._selection_key)
    assert best.iteration == 1
    # Without a split the key is unchanged: bar, then precision.
    uc._resource_map = None
    a, b = _metrics(0, 9, 0, 1, precision=0.90), _metrics(1, 8, 0, 2, precision=0.85)
    assert max([a, b], key=uc._selection_key).iteration == 0


def test_high_confidence_requires_the_release_bar(tmp_path):
    # optimize_fp_threshold_pct 25: precision 0.77 is within the threshold (>= 0.75) but
    # below the 0.80 release bar; HIGH would contradict release_bar_met False.
    out = _run(tmp_path, _pubs(77, 0, 0, 23), split=False, optimize_fp_threshold_pct=25.0)
    assert out.fulltext_iterations[0].fulltext_precision == 0.77
    assert out.release_bar_met is False
    assert out.confidence.value == "low"
    assert "release bar: precision >= 0.80 (not met)" in out.notes
    # At 0.85 both hold: HIGH with the bar met.
    out = _run(tmp_path, _pubs(85, 0, 0, 15), split=False, optimize_fp_threshold_pct=25.0)
    assert out.release_bar_met is True and out.confidence.value == "high"

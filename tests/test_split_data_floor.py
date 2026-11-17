"""Data floor on the split release bar: a split dataset is released when the
unrelated share is <= fulltext_split_unrelated_max AND the confirmed data share
(scope == data; unclear not counted) is >= fulltext_split_data_min, on the
usual minimum sample. Below the floor the full-text loop keeps refining for at
most optimize_max_iterations rounds; then the best iteration is released with a
warning: MEDIUM at most, release_bar_met and data_floor_met false (owner's
decision; fulltext_split_floor_required=True withholds it instead)."""

from __future__ import annotations

from dataset_agent.application import query_optimization as qopt_mod
from tests.test_split_release_bar import _metrics, _pubs, _run, _uc

BAR = "split dataset: unrelated share <= 0.20 and confirmed data share >= 0.50"


def test_below_the_floor_the_best_iteration_is_released_with_a_warning(tmp_path):
    # data 0.40, program-only 0.45, references-only 0.05, unrelated 0.10.
    out = _run(tmp_path, _pubs(40, 45, 5, 10), split=True)
    assert out.release_bar == BAR
    assert out.release_bar_met is False and out.data_floor_met is False
    assert out.confidence.value == "medium"
    floor = out.fulltext_data_floor
    assert floor["min"] == 0.5 and floor["share"] == 0.4 and floor["met"] is False
    assert floor["required"] is False and floor["rounds"] == 1
    lo, hi = floor["wilson95"]
    assert 0.0 < lo < 0.4 < hi < 0.5
    assert ("not met, released with a warning: data floor 0.50 not reached after 1 round; "
            "released at data share 0.40 (Wilson 95% 0.31-0.50)") in out.notes
    assert {k: v["count"] for k, v in out.fulltext_four_way.items()} == {
        "data": 40, "program_only": 45, "references_only": 5, "unrelated": 10}


def test_the_switch_withholds_a_sample_below_the_floor(tmp_path):
    out = _run(tmp_path, _pubs(40, 45, 5, 10), split=True,
               fulltext_split_floor_required=True)
    assert out.release_bar_met is False and out.data_floor_met is False
    assert out.confidence.value == "low"
    assert out.fulltext_data_floor["required"] is True
    assert "(not met: data floor 0.50 not reached after 1 round)" in out.notes


def test_below_the_floor_confidence_is_never_high(tmp_path):
    # Precision 1.00 (data 0.40 + program-only 0.60, nothing unrelated): HIGH by precision,
    # but the data floor is missed, so MEDIUM with the warning.
    out = _run(tmp_path, _pubs(40, 60, 0, 0), split=True)
    assert out.fulltext_iterations[0].fulltext_precision == 1.0
    assert out.confidence.value == "medium" and out.data_floor_met is False


def test_at_or_above_the_floor_the_split_dataset_is_released(tmp_path):
    out = _run(tmp_path, _pubs(55, 20, 15, 10), split=True)
    assert out.release_bar == BAR and out.release_bar_met is True
    assert out.data_floor_met is True
    floor = out.fulltext_data_floor
    assert floor["share"] == 0.55 and floor["met"] is True
    assert "data floor" not in out.notes.split("release bar:")[-1].split("(")[1]


def test_the_floor_counts_only_scope_data(tmp_path):
    uc = _uc(tmp_path)
    # 4 data + 3 unclear of 10: confirmed data share 0.40, below the floor.
    m = _metrics(0, 4, 2, 1, n_unclear=3)
    assert uc._meets_split_target(m) is False and uc._meets_release_bar(m) is False
    assert uc._meets_split_target(_metrics(0, 7, 2, 1)) is True


def test_the_selection_is_floor_then_precision_then_data_share_within_the_unrelated_limit(
        tmp_path):
    uc = _uc(tmp_path)
    precise = _metrics(0, 3, 0, 1, n_prog=6)  # precision 0.90, data 0.30
    floor = _metrics(1, 6, 2, 2)  # precision 0.60, data 0.60: reaches the floor
    over = _metrics(2, 9, 0, 3)  # unrelated 0.25: outside the limit
    assert max([precise, floor, over], key=uc._selection_key).iteration == 1
    tie_low = _metrics(3, 2, 0, 1, n_prog=7)  # precision 0.90, data 0.20
    assert max([tie_low, precise], key=uc._selection_key).iteration == 0


def test_non_split_datasets_ignore_the_floor(tmp_path):
    out = _run(tmp_path, _pubs(85, 0, 0, 15), split=False)
    assert out.release_bar == "precision >= 0.80" and out.release_bar_met is True
    assert out.fulltext_data_floor is None and out.data_floor_met is None


def test_below_the_floor_the_loop_keeps_refining_until_the_round_limit(tmp_path, monkeypatch):
    calls = []
    real = qopt_mod.refine_candidate

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(qopt_mod, "refine_candidate", spy)
    _run(tmp_path, _pubs(40, 45, 5, 10), split=True, optimize_max_iterations=2)
    assert calls, "the unrelated bar alone must not stop the loop below the floor"
    calls.clear()
    _run(tmp_path, _pubs(55, 20, 15, 10), split=True, optimize_max_iterations=2)
    assert not calls, "a sample at the floor stops the loop"


# -- round 10 ------------------------------------------------------------------


def _changing_refiner(monkeypatch):
    """A refiner that always adds one exclude, so the loop runs a second round."""
    from dataset_agent.adapters.fulltext_refiner import RefineResult
    from dataset_agent.domain.proposals import TermDiff

    def refine(candidate, *a, **k):
        n = len(candidate.exclude_terms)
        import dataclasses

        new = dataclasses.replace(candidate, exclude_terms=[*candidate.exclude_terms,
                                                            f"noise{n}"])
        return RefineResult(candidate=new, diffs=[TermDiff(
            kind="exclude", term=f"noise{n}", action="add", status="verified")])

    monkeypatch.setattr(qopt_mod, "refine_candidate", refine)


def test_an_iteration_that_reaches_the_floor_beats_a_more_precise_one_below_it(
        tmp_path, monkeypatch):
    _changing_refiner(monkeypatch)
    below = {"publications": _pubs(40, 50, 0, 10)}  # precision 0.90, data 0.40
    floor = {"publications": _pubs(55, 0, 35, 10)}  # precision 0.55, data 0.55
    out = _run(tmp_path, [], split=True, results=[below, floor], optimize_max_iterations=2)
    assert len(out.fulltext_iterations) == 2
    assert out.data_floor_met is True and out.release_bar_met is True
    assert out.fulltext_data_floor["share"] == 0.55
    assert "not reached" not in out.notes
    assert "Selected iteration 1: it reaches the data floor (confirmed data share 0.55)" in out.notes
    assert "Selected iteration 1 by full-text precision" not in out.notes


def test_the_floor_only_needs_precision_ranking_when_no_iteration_reaches_it(tmp_path):
    uc = _uc(tmp_path)
    precise = _metrics(0, 3, 0, 1, n_prog=6)  # precision 0.90, data 0.30
    lower = _metrics(1, 4, 4, 2)  # precision 0.40, data 0.40
    assert max([precise, lower], key=uc._selection_key).iteration == 0
    reached = _metrics(2, 6, 2, 2)  # precision 0.60, data 0.60
    assert max([precise, lower, reached], key=uc._selection_key).iteration == 2


def test_a_failed_gate_in_the_floor_round_does_not_downgrade_the_selected_iteration(
        tmp_path, monkeypatch):
    _changing_refiner(monkeypatch)
    calls = {"n": 0}

    def evaluate(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            return {"error": "dimensions_query_failed: 500", "error_reason": "HTTP 500"}
        return {"fp_rate_pct": 0.0, "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                "total_pubs": 100, "coverage_pct": 100.0, "for_clause": k.get("for_clause", ""),
                "publications_total": 100, "fp_hits": []}

    monkeypatch.setattr(qopt_mod, "evaluate_for_clause_sync", evaluate)
    out = _run(tmp_path, _pubs(40, 50, 0, 10), split=True, optimize_max_iterations=2)
    assert out.confidence.value == "medium"  # released with the floor warning, not LOW
    assert out.data_floor_met is False
    assert "a gate evaluation failed" not in out.notes.split("full-text release bar:")[-1]
    assert "iteration 1 could not be evaluated" in out.notes
    assert "confidence is capped at LOW" not in out.notes  # the record is MEDIUM


def test_the_floor_needs_the_minimum_sample(tmp_path):
    uc = _uc(tmp_path)
    small = _metrics(0, 5, 0, 0)  # 5 papers, all data: below the minimum sample of 10
    assert uc._meets_data_floor(small) is False


def test_a_failed_gate_with_nothing_selected_does_not_claim_a_selected_iteration(
        tmp_path, monkeypatch):
    monkeypatch.setattr(qopt_mod, "evaluate_for_clause_sync", lambda *a, **k: {
        "error": "dimensions_query_failed: 500", "error_reason": "HTTP 500"})
    out = _run(tmp_path, _pubs(40, 50, 0, 10), split=True, optimize_max_iterations=1)
    assert out.confidence.value == "low"
    assert "keeps its own confidence" not in out.notes
    assert "cannot be selected" in out.notes
    assert "no full-text iteration could be selected; confidence LOW" in out.notes

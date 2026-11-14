"""Scope fields: a paper that only discusses a program is not a genuine dataset paper."""

from __future__ import annotations

import pytest

from dataset_agent.adapters.fulltext_evidence import build_evidence


def _pub(pid, verdict="mentioned"):
    return {
        "publication_id": pid, "doi": None, "title": f"T {pid}", "citation_count": 1,
        "fulltext": {"status": "delivered", "source": "pmc", "reason": None},
        "analyzed": True,
        "validated_datasets": [{"is_mentioned": verdict != "not_mentioned",
                                "is_used": verdict == "used", "text_quote": "q"}],
        "term_contexts": [{"term": "All of Us", "role": "alias", "part": "body", "text": "All of Us"}],
    }


def _ev(*verdicts):
    return build_evidence({"task_id": "T", "publications": [
        _pub(f"p{i}", v) for i, v in enumerate(verdicts)]})


def _program_only(v):
    v.overridden_from, v.verdict, v.scope = v.verdict, "not_mentioned", "program_only"


def test_defaults_unchanged_when_the_check_did_not_run():
    ev = _ev("used", "mentioned", "not_mentioned")
    assert all(v.scope is None for v in ev.verdicts)
    assert ev.n_program_only is None and ev.n_scope_unclear is None
    m = ev.metrics(0, "q", None)
    assert m.fulltext_precision == pytest.approx(2 / 3)
    assert m.program_only_count is None and m.program_only_rate is None
    assert m.scope_unclear_count is None and m.raw_fulltext_precision is None


def test_program_only_papers_are_not_genuine_and_get_their_own_share():
    ev = _ev("used", "mentioned", "mentioned", "mentioned", "not_mentioned")
    ev.verdicts[0].scope = "data"
    _program_only(ev.verdicts[1])
    _program_only(ev.verdicts[2])
    ev.verdicts[3].scope = "unclear"
    assert not ev.verdicts[1].genuine and ev.verdicts[3].genuine
    assert ev.n_program_only == 2 and ev.n_scope_unclear == 1
    assert ev.raw_precision == pytest.approx(4 / 5)
    assert ev.corrected_precision == pytest.approx(2 / 5)
    m = ev.metrics(0, "q", None)
    assert m.fulltext_precision == pytest.approx(2 / 5)
    assert m.raw_fulltext_precision == pytest.approx(4 / 5)
    assert m.program_only_count == 2 and m.program_only_rate == pytest.approx(2 / 5)
    assert m.scope_unclear_count == 1


def test_served_mention_scope_is_read_as_the_validator_scope():
    p = _pub("a")
    p["validated_datasets"][0]["mention_scope"] = "program_only"
    q = _pub("b")
    ev = build_evidence({"task_id": "T", "publications": [p, q]})
    assert ev.verdicts[0].validator_scope == "program_only"
    assert ev.verdicts[1].validator_scope is None
    assert ev.verdicts[0].genuine and ev.verdicts[0].scope is None  # a label only


def test_scope_metrics_report_when_the_check_ran_without_any_scope():
    from dataset_agent.adapters.scope_check import check_scope

    ev = _ev("not_mentioned", "not_mentioned")
    check_scope(ev, agent=None, dataset_name="E", description="", program_names=["P"],
                data_names=["D"])
    m = ev.metrics(0, "q", None)
    assert ev.scope_checked and m.program_only_count == 0 and m.raw_fulltext_precision == 0
    assert m.paper_details and m.paper_details[0]["scope"] is None


def test_raw_precision_adds_back_references_only_drops_and_details_carry_terms():
    ev = _ev("used", "mentioned", "mentioned", "not_mentioned")
    ev.scope_ran = True
    _program_only(ev.verdicts[1])
    v = ev.verdicts[2]
    v.overridden_from, v.verdict, v.scope_references_only = v.verdict, "not_mentioned", True
    assert ev.n_scope_references_only == 1
    assert ev.raw_precision == pytest.approx(3 / 4)
    m = ev.metrics(0, "q", None)
    assert m.scope_references_only_count == 1
    d = {p["id"]: p for p in m.paper_details}
    assert d["p1"]["scope"] == "program_only" and d["p2"]["references_only"]
    assert d["p0"]["terms"] == ["All of Us"]


def test_served_scope_is_read_from_the_deciding_row_only():
    p = _pub("a")
    p["validated_datasets"] = [
        {"name": "Other", "is_mentioned": False, "is_used": False, "mention_scope": "program_only"},
        {"name": "All of Us", "is_mentioned": True, "is_used": True, "text_quote": "q",
         "mention_scope": "data"},
    ]
    q = _pub("b")
    q["validated_datasets"] = [
        {"name": "All of Us", "is_mentioned": True, "is_used": False, "text_quote": "q"},
        {"name": "Other", "is_mentioned": False, "is_used": False, "mention_scope": "program_only"},
    ]
    ev = build_evidence({"task_id": "T", "publications": [p, q]})
    assert ev.verdicts[0].validator_scope == "data" and ev.verdicts[0].verdict == "used"
    assert ev.verdicts[1].validator_scope is None

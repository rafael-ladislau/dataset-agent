from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_agent.adapters.fulltext_evidence import (
    PubVerdict,
    attribute_terms,
    build_evidence,
    term_in,
    term_in_evidence,
)


def _pub(pid, cites, status="delivered", analyzed=True, vd=None, title=""):
    return {
        "publication_id": pid, "doi": f"10/{pid}", "title": title, "citation_count": cites,
        "fulltext": {"status": status, "source": "pmc", "reason": None},
        "analyzed": analyzed, "validated_datasets": vd if vd is not None else [],
    }


def _vd(used, mentioned=True, quote=""):
    return [{"name": "X", "is_used": used, "is_mentioned": mentioned, "text_quote": quote}]


RESULTS = {"publications": [
    _pub("a", 100, vd=_vd(True, quote="we analysed NHIS microdata")),
    _pub("b", 90, vd=_vd(False, quote="NHIS stands for nursing home")),
    _pub("c", 80, vd=_vd(False, mentioned=True, quote="cf. NHIS")),
    _pub("d", 70, vd=[]),
    _pub("e", 60, status="missing"),
    _pub("f", 50, analyzed=False),
    _pub("g", 40, vd=_vd(None, None)),
    _pub("h", 5, vd=_vd(True, quote="beyond top n")),
]}


def test_metrics_exclude_undelivered_and_unanalyzed():
    ev = build_evidence(RESULTS)
    m = ev.metrics(0, "q", 12.0)
    # sampled = a..h; delivered = a,b,c,d,f,g,h (7); evaluated = a,b,c,d,h (f unanalyzed, g no verdict)
    assert (m.n_publications, m.n_fulltext_delivered, m.n_evaluated) == (8, 7, 5)
    # a used, b and c mentioned (not used), d not mentioned, h used: matches = a,b,c,h.
    assert m.fulltext_precision == 0.8
    assert m.mention_rate == 0.8
    assert m.used_rate == 0.4
    assert m.usage_rate == 0.5
    assert m.oa_coverage == 7 / 8
    assert m.abstract_fp_rate_pct == 12.0


def test_is_mentioned_false_counts_as_fp_and_not_mentioned():
    r = {"publications": [_pub("a", 1, vd=_vd(False, mentioned=False, quote="x"))]}
    ev = build_evidence(r)
    assert ev.verdicts[0].verdict == "not_mentioned" and not ev.verdicts[0].genuine


def test_term_attribution():
    ev = build_evidence(RESULTS)
    st = attribute_terms(ev, ["NHIS", "microdata"])
    # Mentioned-but-not-used papers (b, c) are matches, not false positives.
    assert (st["NHIS"].matched, st["NHIS"].used, st["NHIS"].fp) == (3, 1, 0)
    assert (st["microdata"].matched, st["microdata"].used, st["microdata"].fp) == (1, 1, 0)


def test_not_mentioned_is_the_only_false_positive():
    r = {"publications": [
        _pub("u", 3, vd=_vd(True, quote="we use X")),
        _pub("m", 2, vd=_vd(False, mentioned=True, quote="X is a known survey")),
        _pub("n", 1, vd=_vd(False, mentioned=False, quote="unrelated X-ray")),
    ]}
    ev = build_evidence(r)
    assert [v.publication_id for v in ev.genuine] == ["u", "m"]
    assert [v.publication_id for v in ev.false_positives] == ["n"]
    assert ev.metrics(0, "q", None).fulltext_precision == 2 / 3


def test_validation_failed_is_no_verdict_not_not_mentioned():
    failed = _pub("z", 100, vd=[])
    failed["validation_failed"] = True
    ok = _pub("y", 90, vd=_vd(True, quote="used it"))
    ok["validation_failed"] = False
    ev = build_evidence({"publications": [failed, ok]})
    assert ev.n_delivered == 2
    assert [v.publication_id for v in ev.verdicts] == ["y"]


def test_topup_sample_is_every_attempted_publication():
    not_tried = _pub("x", 1000, status="missing")
    not_tried["fulltext"]["reason"] = "not_attempted"
    miss = _pub("m", 500, status="missing")
    miss["fulltext"]["reason"] = "not_found"
    r = {"publications": [not_tried, miss] + RESULTS["publications"]}
    ev = build_evidence(r)
    # attempted = m + a..h (x never sent); delivered = a,b,c,d,f,g,h; evaluated = a,b,c,d,h
    assert (ev.n_sampled, ev.n_delivered, ev.n_evaluated) == (9, 7, 5)
    assert ev.metrics(0, "q", None).oa_coverage == 7 / 9


def test_shortfall_note():
    ev = build_evidence(RESULTS)
    assert ev.shortfall_note is None
    ev.target, ev.batches, ev.stop_reason = 30, 3, "max_batches"
    assert ev.shortfall_note == "7/30 papers with full text (max_batches; 8 attempted in 3 batch(es))"
    ev.budget_exhausted = True
    assert ev.shortfall_note.endswith("; OpenAlex budget exhausted)")


def _v(quote="", title="", contexts=None, verdict="mentioned"):
    return PubVerdict("p", None, title, 1, verdict, quote, contexts=contexts or [])


def _ctx(text, term="x", part="body", **kw):
    return {"term": term, "role": "alias", "part": part, "text": text, **kw}


def test_partial_snippet_credits_only_aliases_with_the_same_acronym_head():
    v = _v(contexts=[_ctx("wave data", term="ADD Health Study", match="partial")], verdict="not_mentioned")
    ev = build_evidence({"publications": []})
    ev.verdicts = [v]
    st = attribute_terms(ev, ["Add Health", "ADD Health Study"])
    assert st["ADD Health Study"].fp == 1
    assert st["Add Health"].fp == 0


def test_underscore_is_a_word_separator():
    v = _v(contexts=[_ctx("loaded the WR_dataset files")])
    assert term_in("WR dataset", v)


def test_title_snippets_do_not_count_for_evidence_but_do_for_attribution():
    v = _v(contexts=[_ctx("The BLS report", part="title")])
    assert not term_in_evidence("BLS", v)
    assert term_in("BLS", v)
    assert not term_in_evidence("BLS", _v(title="The BLS report"))


def test_whole_words_next_to_an_ellipsis_match():
    assert term_in_evidence("Census Bureau", _v(contexts=[_ctx("filler the Census Bureau…")]))
    assert term_in_evidence("ACE", _v(contexts=[_ctx("…ACE study was used")]))
    assert not term_in("ACE", _v(contexts=[_ctx("the surface…")]))


def test_as_quote_falls_back_to_the_snippet_for_the_term():
    v = _v(contexts=[_ctx("first snippet"), _ctx("the BLS wages table")])
    assert v.as_quote("BLS").quote == "the BLS wages table"
    assert v.as_quote().quote == "first snippet"
    assert _v(quote="validator quote", contexts=[_ctx("snip")]).as_quote("BLS").quote == "validator quote"
    assert _v().as_quote("BLS").quote == ""


def test_metrics_store_a_verdict_per_evaluated_paper():
    m = build_evidence(RESULTS).metrics(0, "q", None)
    assert m.papers == [
        ("a", "used"), ("b", "mentioned"), ("c", "mentioned"), ("d", "not_mentioned"), ("h", "used"),
    ]
    assert [p for p, _v in m.papers] == [v.publication_id for v in build_evidence(RESULTS).verdicts]


def _refpub(pid, contexts, used=False):
    p = _pub(pid, 1, vd=_vd(used, quote="Precision Medicine Initiative Cohort Program"))
    p["term_contexts"] = contexts
    return p


def _c(part, role="alias", term="Precision Medicine Initiative Cohort"):
    return {"term": term, "role": role, "part": part, "text": f"... {term} ..."}


def test_references_only_alias_is_not_mentioned():
    ev = build_evidence({"publications": [_refpub("r", [_c("references"), _c("references")], used=True)]})
    v = ev.verdicts[0]
    assert v.verdict == "not_mentioned" and not v.genuine
    assert v.overridden_from == "used"
    assert ev.metrics(0, "q", None).fulltext_precision == 0.0


def test_body_hit_plus_references_hit_keeps_the_verdict():
    ev = build_evidence({"publications": [_refpub("r", [_c("references"), _c("body")])]})
    assert ev.verdicts[0].verdict == "mentioned" and ev.verdicts[0].overridden_from is None


def test_flag_outside_references_keeps_the_verdict():
    ctx = [_c("references"), _c("acknowledgement", role="flag", term="All of Us Research Program")]
    ev = build_evidence({"publications": [_refpub("r", ctx)]})
    assert ev.verdicts[0].verdict == "mentioned"


def test_flag_only_in_references_still_applies_the_rule():
    ctx = [_c("references"), _c("references", role="flag", term="All of Us Research Program")]
    ev = build_evidence({"publications": [_refpub("r", ctx)]})
    assert ev.verdicts[0].verdict == "not_mentioned"


def test_no_contexts_keeps_the_verdict():
    ev = build_evidence({"publications": [_refpub("r", [])]})
    assert ev.verdicts[0].verdict == "mentioned"


def test_flag_contexts_only_keep_the_verdict():
    ev = build_evidence({"publications": [_refpub("r", [_c("references", role="flag")])]})
    assert ev.verdicts[0].verdict == "mentioned"


# -- AI-model verdicts (seed_type "ai_model", model task v2) -------------------
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "llm-results-model-task-v2.json"


def load_model_task_v2() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _mrow(used=False, mentioned=True, quote="", match_type="llm_validation", usage_type=None):
    return {"name": "PhaseNet", "is_mentioned": mentioned, "is_used": used, "text_quote": quote,
            "match_type": match_type, "usage_type": usage_type}


def _mpub(pid, rows, contexts=None, cites=1, validation_failed=False):
    return {
        "publication_id": pid, "doi": f"10/{pid}", "title": f"T {pid}", "citation_count": cites,
        "fulltext": {"status": "delivered", "source": "pmc", "reason": None},
        "analyzed": True, "validation_failed": validation_failed,
        "validated_ai_models": rows, "category_for": [], "term_contexts": contexts or [],
    }


def _model_body(*pubs):
    return {"task_id": "T", "probe": True, "seed_type": "ai_model", "publications": list(pubs)}


def test_contract_fixture_model_evidence():
    ev = build_evidence(load_model_task_v2())
    assert ev.entity == "ai_model"
    # pub.1004 is not open (sampled, not delivered); pub.1003 is an error row.
    assert (ev.n_sampled, ev.n_delivered, ev.n_evaluated) == (4, 3, 2)
    assert ev.n_error_rows == 1
    used, cited = ev.verdicts
    assert (used.publication_id, used.verdict, used.usage_type, used.cited_only) == (
        "pub.1001", "used", "inference", False)
    assert (cited.publication_id, cited.verdict, cited.usage_type, cited.cited_only) == (
        "pub.1002", "mentioned", "cited", True)
    # Owner decision: a reference-list-only model name is a mention, never overridden.
    assert all(v.overridden_from is None for v in ev.verdicts)
    assert ev.n_cited_only == 1
    m = ev.metrics(0, "q", None)
    assert m.fulltext_precision == 1.0 and m.used_rate == 0.5
    assert m.cited_only_count == 1 and m.cited_rate == 0.5


def test_model_references_only_paper_stays_genuine_and_counts_as_cited_only():
    ctx = [_c("references", term="PhaseNet"), _c("references", term="PhaseNet")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow(used=True)], ctx)))
    v = ev.verdicts[0]
    assert v.verdict == "used" and v.genuine and v.overridden_from is None
    # No served usage_type: the location fallback counts it as cited only.
    assert v.cited_only is True and ev.n_cited_only == 1


def test_same_paper_as_dataset_is_still_overridden():
    ev = build_evidence({"publications": [_refpub("r", [_c("references"), _c("references")], used=True)]})
    assert ev.entity == "dataset"
    assert ev.verdicts[0].overridden_from == "used" and ev.verdicts[0].cited_only is False
    m = ev.metrics(0, "q", None)
    assert ev.n_cited_only is None and m.cited_only_count is None and m.cited_rate is None


def test_model_cited_only_fallback_ignores_flag_contexts():
    # Aurora named only in the references; GraphCast (a flag) in the body.
    ctx = [_c("references", term="Aurora"), _c("body", role="flag", term="GraphCast")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow()], ctx)))
    assert ev.verdicts[0].verdict == "mentioned" and ev.verdicts[0].cited_only is True


def test_model_body_mention_is_not_cited_only():
    ctx = [_c("references", term="PhaseNet"), _c("body", term="PhaseNet")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow()], ctx)))
    assert ev.verdicts[0].cited_only is False


def test_served_usage_type_wins_over_the_location_fallback():
    ctx = [_c("references", term="PhaseNet")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow(used=True, usage_type="fine_tuning")], ctx)))
    assert ev.verdicts[0].cited_only is False


def test_usage_type_cited_is_a_mention():
    ev = build_evidence(_model_body(_mpub("r", [_mrow(mentioned=False, usage_type="cited")])))
    v = ev.verdicts[0]
    assert v.verdict == "mentioned" and v.genuine and v.cited_only is True


def test_empty_model_list_is_no_verdict():
    ev = build_evidence(_model_body(_mpub("r", [])))
    assert ev.n_delivered == 1 and ev.n_evaluated == 0


@pytest.mark.parametrize("mt", ["validation_error", "processing_error"])
def test_model_error_rows_are_no_verdict(mt):
    ev = build_evidence(_model_body(_mpub("r", [_mrow(mentioned=False, match_type=mt)])))
    assert ev.n_evaluated == 0 and ev.n_error_rows == 1


def test_error_row_next_to_a_real_row_keeps_the_real_verdict():
    rows = [_mrow(mentioned=False, match_type="processing_error"), _mrow(mentioned=False)]
    ev = build_evidence(_model_body(_mpub("r", rows)))
    assert ev.verdicts[0].verdict == "not_mentioned" and ev.n_error_rows == 0


def test_model_not_mentioned_is_a_false_positive():
    ev = build_evidence(_model_body(_mpub("r", [_mrow(mentioned=False, quote="phase unwrapping")])))
    v = ev.verdicts[0]
    assert v.verdict == "not_mentioned" and not v.genuine and v.cited_only is False
    assert v.match_type == "llm_validation"


def test_model_body_ignores_validated_datasets():
    p = _mpub("r", [])
    p["validated_datasets"] = [{"name": "X", "is_used": True, "is_mentioned": True}]
    assert build_evidence(_model_body(p)).n_evaluated == 0


def test_seed_type_mismatch_raises():
    with pytest.raises(ValueError, match="seed_type"):
        build_evidence(_model_body(), entity="dataset")
    with pytest.raises(ValueError, match="seed_type"):
        build_evidence({"seed_type": "dataset", "publications": []}, entity="ai_model")


def test_explicit_model_entity_without_seed_type_reads_model_rows():
    body = _model_body(_mpub("r", [_mrow()]))
    del body["seed_type"]
    assert build_evidence(body, entity="ai_model").verdicts[0].verdict == "mentioned"

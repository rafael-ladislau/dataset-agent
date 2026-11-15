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
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "llm-results-model-task-v2.1.json"


def load_model_task_v2() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _mrow(used=False, mentioned=True, quote="", match_type="llm_validation", usage_type=None,
          **v21):
    """A model task v2 row; ``v21`` adds v2.1 fields (mention_location, matched_name)."""
    return {"name": "PhaseNet", "is_mentioned": mentioned, "is_used": used, "text_quote": quote,
            "match_type": match_type, "usage_type": usage_type, **v21}


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
    # pub.1004 is not open (sampled, not delivered); pub.1003 is an error row; pub.1005
    # is a namesake (another PhaseNet); pub.1006 is served with location 'unknown'.
    assert (ev.n_sampled, ev.n_delivered, ev.n_evaluated) == (6, 5, 4)
    assert ev.n_error_rows == 1 and ev.n_incoherent == 0
    used, cited, namesake, unknown = ev.verdicts
    assert (namesake.publication_id, namesake.verdict, namesake.cited_only) == (
        "pub.1005", "not_mentioned", False)
    # The name contexts decide cited-only, so a served 'unknown' location does not.
    assert (unknown.publication_id, unknown.verdict, unknown.mention_location,
            unknown.cited_only) == ("pub.1006", "used", "unknown", False)
    assert (used.publication_id, used.verdict, used.usage_type, used.cited_only) == (
        "pub.1001", "used", "inference", False)
    assert (cited.publication_id, cited.verdict, cited.usage_type, cited.cited_only) == (
        "pub.1002", "mentioned", "cited", True)
    # Owner decision: a reference-list-only model name is a mention, never overridden.
    assert all(v.overridden_from is None for v in ev.verdicts)
    assert ev.n_cited_only == 1
    m = ev.metrics(0, "q", None)
    assert m.fulltext_precision == 0.75 and m.used_rate == 0.5
    assert m.cited_only_count == 1 and m.cited_rate == 1 / 3
    assert m.not_run_count == 1 and m.not_run_rate == 1 / 3 and m.cited_only_unknown_count == 0
    assert (used.mention_location, used.matched_name) == ("text", "PhaseNet")
    assert m.paper_details == [
        {"id": "pub.1001", "title": "Aftershock detection with deep phase picking",
         "verdict": "used", "location": "text", "cited_only": False, "not_run": False,
         "matched_name": "PhaseNet"},
        {"id": "pub.1002", "title": "A review of seismic arrival picking",
         "verdict": "mentioned", "location": "references_only", "cited_only": True,
         "not_run": True, "matched_name": "PhaseNet"},
        {"id": "pub.1005", "title": "Phase-based video frame interpolation",
         "verdict": "not_mentioned", "location": "text", "cited_only": False, "not_run": False,
         "matched_name": "PhaseNet"},
        {"id": "pub.1006", "title": "Swarm seismicity of a volcanic island",
         "verdict": "used", "location": "unknown", "cited_only": False, "not_run": False,
         "matched_name": "PhaseNet"},
    ]


def test_model_references_only_paper_stays_genuine_and_counts_as_cited_only():
    ctx = [_c("references", term="PhaseNet"), _c("references", term="PhaseNet")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow(used=True)], ctx)))
    v = ev.verdicts[0]
    assert v.verdict == "used" and v.genuine and v.overridden_from is None
    assert v.cited_only is True and ev.n_cited_only == 1


def test_same_paper_as_dataset_is_still_overridden():
    ev = build_evidence({"publications": [_refpub("r", [_c("references"), _c("references")], used=True)]})
    assert ev.entity == "dataset"
    assert ev.verdicts[0].overridden_from == "used" and ev.verdicts[0].cited_only is False
    m = ev.metrics(0, "q", None)
    assert ev.n_cited_only is None and m.cited_only_count is None and m.cited_rate is None
    assert m.paper_details is None


def test_model_cited_only_fallback_ignores_flag_contexts():
    # Aurora named only in the references; GraphCast (a flag) in the body.
    ctx = [_c("references", term="Aurora"), _c("body", role="flag", term="GraphCast")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow()], ctx)))
    assert ev.verdicts[0].verdict == "mentioned" and ev.verdicts[0].cited_only is True


def test_model_body_mention_is_not_cited_only():
    ctx = [_c("references", term="PhaseNet"), _c("body", term="PhaseNet")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow()], ctx)))
    assert ev.verdicts[0].cited_only is False


def test_usage_type_never_decides_cited_only():
    """Cited-only is where the name occurs; usage_type 'cited' is reported as not run."""
    refs = [_c("references", term="PhaseNet")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow(used=True, usage_type="fine_tuning")], refs)))
    assert ev.verdicts[0].cited_only is True and ev.n_not_run == 0
    body = [_c("body", term="PhaseNet")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow(usage_type="cited")], body)))
    v = ev.verdicts[0]
    assert v.cited_only is False and v.not_run is True and ev.n_not_run == 1


def test_usage_type_cited_is_a_mention():
    """A row judged not to mention the model is never genuine, whatever its usage_type."""
    ev = build_evidence(_model_body(_mpub("r", [_mrow(mentioned=False, usage_type="cited")])))
    v = ev.verdicts[0]
    assert v.verdict == "not_mentioned" and not v.genuine and v.cited_only is False
    assert ev.n_incoherent == 1


@pytest.mark.parametrize(
    ("mentioned", "used", "usage_type", "verdict", "incoherent"),
    [
        (False, False, "cited", "not_mentioned", 1),
        (False, True, "inference", "not_mentioned", 1),
        (True, False, "cited", "mentioned", 0),
        (None, False, "cited", "mentioned", 0),
        (False, False, None, "not_mentioned", 0),
        (True, True, "inference", "used", 0),
        (None, True, "inference", "used", 0),
    ],
)
def test_model_row_judged_not_this_model_is_never_genuine(mentioned, used, usage_type, verdict, incoherent):
    row = _mrow(mentioned=mentioned, used=used, usage_type=usage_type)
    ev = build_evidence(_model_body(_mpub("r", [row])))
    assert ev.verdicts[0].verdict == verdict
    assert ev.n_incoherent == incoherent


def test_analyzed_papers_without_a_model_verdict_are_counted():
    no_judgement = {"name": "PhaseNet", "is_mentioned": None, "is_used": None, "text_quote": "",
                    "match_type": "llm_validation", "usage_type": None}
    ev = build_evidence(_model_body(_mpub("a", []), _mpub("b", [no_judgement]),
                                    _mpub("c", [_mrow(match_type="processing_error")])))
    assert ev.n_evaluated == 0 and ev.n_no_verdict == 2 and ev.n_error_rows == 1


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



# -- model task v2.1: served mention_location ---------------------------------
def test_name_contexts_decide_cited_only_before_the_served_location():
    # The served location is the judge's quote position; the name contexts win.
    row = _mrow(usage_type="cited", mention_location="text", matched_name="Phase-Net")
    ev = build_evidence(_model_body(_mpub("r", [row], [_c("references", term="PhaseNet")])))
    v = ev.verdicts[0]
    assert v.verdict == "mentioned" and v.cited_only is True
    assert (v.mention_location, v.matched_name) == ("text", "Phase-Net")
    row = _mrow(used=True, usage_type="inference", mention_location="references_only")
    ev = build_evidence(_model_body(_mpub("r", [row], [_c("body", term="PhaseNet")])))
    assert ev.verdicts[0].cited_only is False


@pytest.mark.parametrize(("location", "cited_only"),
                         [("references_only", True), ("text", False), ("unknown", None), (None, None)])
def test_without_name_contexts_the_served_location_decides_else_unknown(location, cited_only):
    flag_only = [_c("references", role="flag", term="GraphCast")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow(mention_location=location)], flag_only)))
    assert ev.verdicts[0].cited_only is cited_only
    assert ev.n_cited_only_unknown == (1 if cited_only is None else 0)
    assert ev.n_cited_only == (1 if cited_only else 0)


def test_cited_only_share_divides_by_genuine_papers():
    refs = [_c("references", term="PhaseNet")]
    ev = build_evidence(_model_body(
        _mpub("a", [_mrow()], refs), _mpub("b", [_mrow()], [_c("body", term="PhaseNet")]),
        _mpub("c", [_mrow(mentioned=False)], refs), _mpub("d", [_mrow(mentioned=False)], refs),
    ))
    m = ev.metrics(0, "q", None)
    assert m.cited_only_count == 1 and m.cited_rate == 0.5


@pytest.mark.parametrize("location", ["unknown", None])
def test_unknown_or_missing_location_falls_back_to_usage_type_and_contexts(location):
    ctx = [_c("references", term="PhaseNet")]
    ev = build_evidence(_model_body(_mpub("r", [_mrow(mention_location=location)], ctx)))
    v = ev.verdicts[0]
    assert v.cited_only is True and v.mention_location == location


def test_not_mentioned_model_paper_is_never_cited_only():
    row = _mrow(mentioned=False, mention_location="references_only")
    ev = build_evidence(_model_body(_mpub("r", [row])))
    assert ev.verdicts[0].cited_only is False


def test_v2_rows_without_location_give_paper_details_with_nulls():
    ev = build_evidence(_model_body(_mpub("r", [_mrow(used=True, usage_type="inference")])))
    assert ev.metrics(0, "q", None).paper_details == [
        {"id": "r", "title": "T r", "verdict": "used", "location": None, "cited_only": None,
         "not_run": False, "matched_name": None},
    ]


def _quoted_refpub(quote, used=True):
    p = _pub("r", 1, vd=_vd(used, quote=quote))
    p["term_contexts"] = [{"term": "Example Research Program", "role": "alias",
                           "part": "references",
                           "text": "Investigators. The Example Research Program. N Engl J Med 2019"}]
    return p


def test_references_only_override_keeps_the_verdict_when_the_quote_is_a_body_sentence():
    # The seeded name is only in the bibliography, but the judge quotes a body sentence
    # with a short form the snippets never searched: the judge's verdict stands.
    ev = build_evidence({"publications": [_quoted_refpub("We analysed EXRP participants' records.")]})
    v = ev.verdicts[0]
    assert v.verdict == "used" and v.overridden_from is None


@pytest.mark.parametrize("quote", ["", "  ", "The Example Research Program", "Investigators. "
                                   "The Example Research Program. N Engl J Med 2019. Extra"])
def test_references_only_override_applies_on_an_empty_or_reference_quote(quote):
    ev = build_evidence({"publications": [_quoted_refpub(quote)]})
    assert ev.verdicts[0].verdict == "not_mentioned"
    assert ev.verdicts[0].overridden_from == "used"


# Real reference snippets (extraction term_contexts._parts) hold only cited titles joined by
# " | " in a ~200-character window, with no authors, numbers or years; the judge quotes the
# full bibliography entry.
_TITLE = "The Example Research Program: a national cohort for precision medicine"
_LONG = ("The Example Research Program: design, recruitment, consent, data collection and "
         "the first results of an international longitudinal cohort of one million adults")


def _title_refpub(quote, snippet):
    p = _pub("r", 1, vd=_vd(False, quote=quote))
    p["term_contexts"] = [{"term": "Example Research Program", "role": "alias",
                           "part": "references", "text": snippet}]
    return p


_SNIPPET = f"Dupilumab in atopic dermatitis | {_TITLE} | Drug-induced liver injury"
# The window cut the long title after "design, recruitment".
_CUT = "…Drug-induced liver injury | The Example Research Program: design, recruitment…"


@pytest.mark.parametrize("quote,snippet", [
    (f"[62] {_TITLE}", _SNIPPET),
    (f"Denny JC, Rutter JL, Goldstein DB, et al. {_TITLE}. N Engl J Med. 2019;381:668-676",
     _SNIPPET),
    (f"{_TITLE} (ref 62)", _SNIPPET),
    (f"62. {_LONG}. Lancet. 2020", _CUT),
    (f"{_TITLE}¹⁴", _SNIPPET),
])
def test_references_only_override_applies_on_bibliography_style_quotes(quote, snippet):
    ev = build_evidence({"publications": [_title_refpub(quote, snippet)]})
    assert ev.verdicts[0].verdict == "not_mentioned"
    assert ev.verdicts[0].overridden_from == "mentioned"


@pytest.mark.parametrize("quote", ["", "[62]"])
def test_references_only_override_applies_on_an_empty_quote_with_title_snippets(quote):
    ev = build_evidence({"publications": [_title_refpub(quote, _SNIPPET)]})
    assert ev.verdicts[0].overridden_from == "mentioned"


def test_body_sentence_sharing_title_words_keeps_the_verdict():
    quote = ("We linked records from the Example Research Program to a national registry "
             "of adults.")
    ev = build_evidence({"publications": [_title_refpub(quote, _SNIPPET)]})
    assert ev.verdicts[0].verdict == "mentioned" and ev.verdicts[0].overridden_from is None


def test_a_body_piece_with_the_name_keeps_the_verdict_next_to_a_cited_title():
    # A review quotes a cited study's title and then its own table of included studies:
    # the table text names the data resource and matches no reference piece.
    cited = ("De-Black-Boxing Health AI: Demonstrating Reproducible Machine Learning "
             "Computable Phenotypes Using the N3C-RECOVER Long COVID Model in the All of Us "
             "Data Repository")
    snippet = (f"…Long COVID after SARS-CoV-2 infection in adults | {cited} | Machine learning "
               "for clinical prediction…")
    quote = f"{cited}... AUROCs (N3C= 0.83 and All of Us = 0.72)"
    p = _pub("r", 1, vd=_vd(False, quote=quote))
    p["term_contexts"] = [{"term": "All of Us", "role": "alias", "part": "references",
                           "text": snippet}]
    ev = build_evidence({"publications": [p]})
    assert ev.verdicts[0].verdict == "mentioned" and ev.verdicts[0].overridden_from is None
    # The cited title alone is a citer: the override applies.
    p["validated_datasets"][0]["text_quote"] = cited
    ev = build_evidence({"publications": [p]})
    assert ev.verdicts[0].overridden_from == "mentioned"

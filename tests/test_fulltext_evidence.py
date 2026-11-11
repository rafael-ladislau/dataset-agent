from __future__ import annotations

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
    ev = build_evidence(RESULTS, top_n=7)
    m = ev.metrics(0, "q", 12.0)
    # sampled = top 7 (a..g); delivered = a,b,c,d,f,g (6); evaluated = a,b,c,d (f unanalyzed, g no verdict)
    assert (m.n_publications, m.n_fulltext_delivered, m.n_evaluated) == (7, 6, 4)
    # a used, b and c mentioned (not used), d not mentioned: matches = a,b,c.
    assert m.fulltext_precision == 0.75
    assert m.mention_rate == 0.75
    assert m.used_rate == 0.25
    assert m.usage_rate == 1 / 3
    assert m.oa_coverage == 6 / 7
    assert m.abstract_fp_rate_pct == 12.0


def test_is_mentioned_false_counts_as_fp_and_not_mentioned():
    r = {"publications": [_pub("a", 1, vd=_vd(False, mentioned=False, quote="x"))]}
    ev = build_evidence(r, 5)
    assert ev.verdicts[0].verdict == "not_mentioned" and not ev.verdicts[0].genuine


def test_term_attribution():
    ev = build_evidence(RESULTS, top_n=7)
    st = attribute_terms(ev, ["NHIS", "microdata"])
    # Mentioned-but-not-used papers (b, c) are matches, not false positives.
    assert (st["NHIS"].matched, st["NHIS"].used, st["NHIS"].fp) == (3, 1, 0)
    assert st["NHIS"].mentioned_only == 2
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
    ev = build_evidence({"publications": [failed, ok]}, 10)
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
    assert "wr dataset" in v.text


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

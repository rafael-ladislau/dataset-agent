from __future__ import annotations

from dataset_agent.adapters.fulltext_evidence import attribute_terms, build_evidence


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
    assert m.fulltext_precision == 0.25
    assert m.mention_rate == 0.75      # a used, b not_mentioned? b has is_mentioned=True -> mentioned
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
    assert (st["NHIS"].used, st["NHIS"].fp) == (1, 2)
    assert (st["microdata"].used, st["microdata"].fp) == (1, 0)

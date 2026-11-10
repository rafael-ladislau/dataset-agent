from __future__ import annotations

from dataset_agent.adapters.fulltext_evidence import build_evidence
from dataset_agent.adapters.fulltext_refiner import refine_candidate
from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.domain.ports import AgentPort
from dataset_agent.settings import Settings


def pub(pid, cites, used, quote, title=""):
    return {"publication_id": pid, "doi": None, "title": title, "citation_count": cites,
            "fulltext": {"status": "delivered"}, "analyzed": True,
            "validated_datasets": [{"name": "X", "is_used": used, "is_mentioned": True, "text_quote": quote}]}


class FakeLLM(AgentPort):
    def __init__(self, out):
        self.out = out
        self.calls = 0

    def get_information(self, prompt, *, tools=None):
        return ""

    def get_structured(self, prompt, result_tool):
        self.calls += 1
        return self.out


S = Settings(refine_min_alias_papers=3, refine_drop_fp_share=0.8, refine_demote_fp_share=0.5,
             fulltext_min_evidence=3)


def ev(pubs):
    return build_evidence({"publications": pubs}, 100)


def diff(r, kind, term):
    return next(d for d in r.diffs if d.kind == kind and d.term == term)


def test_drop_demote_keep_unverified():
    pubs = (
        [pub(f"n{i}", 50 - i, False, "NHIS nursing home survey") for i in range(4)]       # NHIS: 4 fp
        + [pub(f"a{i}", 30 - i, True, "Alpha Survey data") for i in range(3)]               # Alpha: 3 used
        + [pub(f"b{i}", 20 - i, i == 0, "Beta Survey sample") for i in range(3)]            # Beta: 1 used 2 fp
    )
    cand = QueryCandidate("X", safe=["NHIS", "Alpha Survey", "Beta Survey", "Rare Name"], flag_terms=["health"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=None, dataset_name="X")
    assert diff(r, "alias", "NHIS").action == "drop"
    assert diff(r, "alias", "Beta Survey").action == "demote"
    assert diff(r, "alias", "Alpha Survey").action == "keep"
    rare = diff(r, "alias", "Rare Name")
    assert rare.action == "keep" and rare.status == "unverified"
    assert r.candidate.safe == ["Alpha Survey", "Rare Name"]
    assert r.candidate.risky == ["Beta Survey"]
    assert diff(r, "alias", "NHIS").evidence[0].quote


def test_abstract_signal_only_marks_suspect_and_never_changes():
    pubs = [pub(f"p{i}", 9 - i, True, "NHIS data") for i in range(3)]
    cand = QueryCandidate("X", safe=["NHIS", "Rare Name"], flag_terms=[])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=None, dataset_name="X",
                         abstract_fp_aliases={"rare name"})
    d = diff(r, "alias", "Rare Name")
    assert d.status == "suspect" and d.action == "keep"
    assert not r.changed


def test_last_alias_is_not_dropped():
    pubs = [pub(f"n{i}", 9 - i, False, "NHIS nursing") for i in range(4)]
    r = refine_candidate(QueryCandidate("X", safe=["NHIS"]), ev(pubs), settings=S, agent=None, dataset_name="X")
    assert diff(r, "alias", "NHIS").action == "keep"


def test_llm_proposals_are_validated_against_quotes_and_counted():
    pubs = (
        [pub(f"g{i}", 40 - i, True, "we used the National Health Interview Survey (NHIS) microdata") for i in range(3)]
        + [pub(f"f{i}", 20 - i, False, "NHIS nursing home staffing") for i in range(2)]
    )
    llm = FakeLLM({
        "exclude_terms": ["nursing home", "microdata"],   # 2nd occurs in genuine -> rejected
        "flag_terms": ["microdata", "bogus"],              # bogus not in quotes -> rejected
        "new_aliases": ["National Health Interview Survey", "Made Up Name", "NHIS microdata"],
        "reasoning": "",
    })
    counts = {"National Health Interview Survey": 500, "NHIS microdata": 0}
    cand = QueryCandidate("X", safe=["NHIS"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=llm, dataset_name="X",
                         count_fn=lambda t: counts.get(t, 0), reference_count=100)
    assert [d.term for d in r.diffs if d.kind == "exclude" and d.action == "add"] == ["nursing home"]
    assert [d.term for d in r.diffs if d.kind == "flag" and d.action == "add"] == ["microdata"]
    added = [d.term for d in r.diffs if d.kind == "alias" and d.action == "add"]
    assert added == ["National Health Interview Survey"]       # unseen / zero-count rejected
    assert r.candidate.exclude_terms == ["nursing home"]
    assert "National Health Interview Survey" in r.candidate.safe


def test_too_little_evidence_changes_nothing():
    pubs = [pub("a", 1, False, "NHIS nursing")]
    llm = FakeLLM({"exclude_terms": ["nursing"], "flag_terms": [], "new_aliases": [], "reasoning": ""})
    r = refine_candidate(QueryCandidate("X", safe=["NHIS", "Other"]), ev(pubs), settings=S, agent=llm, dataset_name="X")
    assert not r.changed and llm.calls == 0


def _refine_with(pubs, **proposal):
    out = {"exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""}
    out.update(proposal)
    return refine_candidate(QueryCandidate("X", safe=["NHIS"]), ev(pubs), settings=S, agent=FakeLLM(out),
                            dataset_name="X", count_fn=lambda t: 5, reference_count=100)


def test_title_only_flag_is_rejected():
    pubs = [pub(f"g{i}", 40 - i, True, "we used NHIS data", title="A cohort study") for i in range(3)] + [
        pub("f0", 5, False, "NHIS nursing"), pub("f1", 4, False, "NHIS nursing")]
    r = _refine_with(pubs, flag_terms=["cohort study"])
    assert not [d for d in r.diffs if d.kind == "flag" and d.action == "add"]


def test_flag_needs_two_genuine_quotes_even_with_one_genuine_paper():
    pubs = [pub("g0", 40, True, "we used NHIS microdata")] + [
        pub(f"f{i}", 5 - i, False, "NHIS nursing") for i in range(3)]
    r = _refine_with(pubs, flag_terms=["microdata"])
    assert not [d for d in r.diffs if d.kind == "flag" and d.action == "add"]


def test_title_only_new_alias_is_rejected():
    pubs = [pub(f"g{i}", 40 - i, True, "we used NHIS data", title="NHANES III survey") for i in range(3)] + [
        pub("f0", 5, False, "NHIS nursing")]
    r = _refine_with(pubs, new_aliases=["NHANES III"])
    assert not [d for d in r.diffs if d.kind == "alias" and d.action == "add"]


def test_exclude_reason_says_quote_versus_title():
    pubs = [pub(f"g{i}", 40 - i, True, "we used NHIS data") for i in range(3)] + [
        pub("f0", 5, False, "NHIS staffing", title="nursing home study"),
        pub("f1", 4, False, "NHIS nursing home staffing")]
    r = _refine_with(pubs, exclude_terms=["nursing home"])
    d = diff(r, "exclude", "nursing home")
    assert d.action == "add"
    assert "1 evidence quote(s) and 1 title(s) only" in d.reason


def test_llm_terms_are_unquoted_and_inner_quotes_dropped():
    pubs = (
        [pub(f"g{i}", 40 - i, True, 'we used the NHIS "public use" microdata files') for i in range(3)]
        + [pub(f"f{i}", 20 - i, False, "NHIS nursing home staffing") for i in range(2)]
    )
    llm = FakeLLM({
        "exclude_terms": ['"nursing home"'],
        "flag_terms": ['"microdata"', 'NHIS "public use" microdata'],
        "new_aliases": [],
        "reasoning": "",
    })
    r = refine_candidate(QueryCandidate("X", safe=["NHIS"]), ev(pubs), settings=S, agent=llm,
                         dataset_name="X", count_fn=lambda t: 0, reference_count=100)
    assert r.candidate.exclude_terms == ["nursing home"]
    assert r.candidate.flag_terms == ["microdata"]
    for term in [*r.candidate.exclude_terms, *r.candidate.flag_terms, *r.candidate.safe, *r.candidate.risky]:
        assert '"' not in term


def test_unquote_strips_every_kind_of_wrapping_quote_but_keeps_apostrophes():
    from dataset_agent.adapters.fulltext_refiner import _unquote

    for raw in ['"nursing home"', "'nursing home'", "\u2018nursing home\u2019", "\u201cnursing home\u201d",
                "\u00abnursing home\u00bb", " \u2018 nursing home \u2019 "]:
        assert _unquote(raw) == "nursing home", raw
    assert _unquote("Children's Health Study") == "Children's Health Study"
    assert _unquote("the \u2018Children\u2019s\u2019 survey") == "the \u2018Children\u2019s\u2019 survey"


def test_terms_match_whole_words_only():
    from dataset_agent.adapters.fulltext_evidence import PubVerdict, term_in, term_in_quote

    v = PubVerdict(publication_id="p", doi=None, title="A placebo trial", citation_count=1,
                   verdict="not_mentioned", quote="patients furnished data; the ACE-III score")
    assert not term_in("ACE", PubVerdict(**{**v.__dict__, "quote": "a placebo arm"}))
    assert term_in_quote("ACE", v)            # "ACE-III" -> "ace iii"
    assert not term_in_quote("NIS", v)        # inside "furnished"
    assert term_in("placebo trial", v)        # title phrase


def test_whole_word_match_tolerates_plurals_and_unicode_forms():
    from dataset_agent.adapters.fulltext_evidence import PubVerdict, term_in_quote

    def v(quote):
        return PubVerdict(publication_id="p", doi=None, title="", citation_count=1,
                          verdict="used", quote=quote)

    assert term_in_quote("Family Survey", v("data from the Family Surveys of 2010"))
    assert term_in_quote("nursing homes", v("residents of a nursing home"))
    assert term_in_quote("nursing home", v("residents of nursing homes"))
    assert term_in_quote("Café Study", v("the Café Study cohort"))   # decomposed accent
    assert not term_in_quote("ACE", v("a placebo arm"))
    assert not term_in_quote("FS", v("FSS scores"))
    assert not term_in_quote("NIS", v("NI data only"))   # short tokens stay exact
    assert term_in_quote("ACE", v("the ACEs questionnaire"))

from __future__ import annotations

import pytest

from dataset_agent.adapters.fulltext_evidence import build_evidence
from dataset_agent.adapters.fulltext_refiner import refine_candidate
from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.domain.ports import AgentPort
from dataset_agent.settings import Settings


def pub(pid, cites, used, quote, title="", mentioned=None):
    # A paper that does not use the dataset is a false positive only when it does not
    # mention it either; pass mentioned=True for a mentioned-but-not-used match.
    mentioned = used if mentioned is None else mentioned
    return {"publication_id": pid, "doi": None, "title": title, "citation_count": cites,
            "fulltext": {"status": "delivered"}, "analyzed": True,
            "validated_datasets": [{"name": "X", "is_used": used, "is_mentioned": mentioned, "text_quote": quote}]}


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
    return build_evidence({"publications": pubs})


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
        [pub(f"g{i}", 40 - i, True, "we used the National Health Interview Survey (NHIS) NCHS microdata")
         for i in range(3)]
        + [pub(f"f{i}", 20 - i, False, "NHIS nursing home staffing") for i in range(2)]
    )
    llm = FakeLLM({
        "exclude_terms": ["nursing home", "microdata"],   # 2nd occurs in genuine -> rejected
        "flag_terms": ["NCHS", "bogus"],                   # bogus not in quotes -> rejected
        "new_aliases": ["National Health Interview Survey", "Made Up Name", "NHIS microdata"],
        "reasoning": "",
    })
    counts = {"National Health Interview Survey": 500, "NHIS microdata": 0, "NCHS": 40_000}
    cand = QueryCandidate("X", safe=["NHIS"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=llm, dataset_name="X",
                         count_fn=lambda t: counts.get(t, 0), reference_count=100)
    assert [d.term for d in r.diffs if d.kind == "exclude" and d.action == "add"] == ["nursing home"]
    assert [d.term for d in r.diffs if d.kind == "flag" and d.action == "add"] == ["NCHS"]
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
    assert "1 full-text snippet(s)/quote(s) and 1 title(s) only" in d.reason


def test_llm_terms_are_unquoted_and_inner_quotes_dropped():
    pubs = (
        [pub(f"g{i}", 40 - i, True, 'we used the NHIS "public use" NCHS microdata files') for i in range(3)]
        + [pub(f"f{i}", 20 - i, False, "NHIS nursing home staffing") for i in range(2)]
    )
    llm = FakeLLM({
        "exclude_terms": ['"nursing home"'],
        "flag_terms": ['"NCHS"', 'NHIS "public use" microdata'],
        "new_aliases": [],
        "reasoning": "",
    })
    r = refine_candidate(QueryCandidate("X", safe=["NHIS"]), ev(pubs), settings=S, agent=llm,
                         dataset_name="X", count_fn=lambda t: 1000, reference_count=100)
    assert r.candidate.exclude_terms == ["nursing home"]
    assert r.candidate.flag_terms == ["NCHS"]
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
    assert term_in_quote("nursing home", v("residents of nursing homes"))
    assert term_in_quote("Café Study", v("the Café Study cohort"))   # decomposed accent
    assert not term_in_quote("ACE", v("a placebo arm"))
    assert not term_in_quote("FS", v("FSS scores"))
    assert not term_in_quote("NIS", v("NI data only"))   # short tokens stay exact
    assert not term_in_quote("ACE", v("the ACEs questionnaire"))   # short: exact only
    assert not term_in_quote("NHIS", v("the NHI programme"))       # term never altered
    assert term_in_quote("NHIS", v("we used NHIS data"))
    assert not term_in_quote("MEP", v("we used MEPS data"))
    assert term_in_quote("Family Survey", v("Family Surveys"), exact=False)
    assert not term_in_quote("Family Survey", v("Family Surveys"), exact=True)


def test_new_alias_must_occur_literally_in_a_genuine_quote():
    pubs = [pub(f"g{i}", 40 - i, True, "we used the National Family Surveys microdata") for i in range(3)]
    llm = FakeLLM({"exclude_terms": [], "flag_terms": [],
                   "new_aliases": ["National Family Survey", "National Family Surveys"], "reasoning": ""})
    r = refine_candidate(QueryCandidate("X", safe=["NFS"]), ev(pubs), settings=S, agent=llm,
                         dataset_name="X", count_fn=lambda t: 500, reference_count=100)
    added = [d.term for d in r.diffs if d.kind == "alias" and d.action == "add"]
    assert added == ["National Family Surveys"]


def test_consonant_y_plural_in_text_counts_for_the_term():
    from dataset_agent.adapters.fulltext_evidence import PubVerdict, term_in_quote

    v = PubVerdict(publication_id="p", doi=None, title="", citation_count=1, verdict="used",
                   quote="pooled Family Studies cohorts")
    assert term_in_quote("Family Study", v)
    assert not term_in_quote("Family Day", PubVerdict(**{**v.__dict__, "quote": "Family Daies"}))


def test_exclude_blocked_by_a_genuine_singular_of_a_plural_term():
    pubs = (
        [pub(f"g{i}", 40 - i, True, "NHIS data from a clinical trial") for i in range(3)]
        + [pub(f"f{i}", 20 - i, False, "NHIS clinical trials registry") for i in range(2)]
    )
    llm = FakeLLM({"exclude_terms": ["clinical trials", "registry"], "flag_terms": [],
                   "new_aliases": [], "reasoning": ""})
    r = refine_candidate(QueryCandidate("X", safe=["NHIS"]), ev(pubs), settings=S, agent=llm,
                         dataset_name="X", count_fn=lambda t: 0, reference_count=100)
    assert r.candidate.exclude_terms == ["registry"]


def test_exclude_guard_tries_every_plausible_singular():
    from dataset_agent.adapters.fulltext_evidence import singular_variants

    for plural, singular in [("rare diseases", "rare disease"), ("boxes", "box"), ("analyses", "analysis"),
                             ("cohort studies", "cohort study"), ("censuses", "census")]:
        assert singular in singular_variants(plural), plural
    pubs = (
        [pub(f"g{i}", 40 - i, True, "NHIS data on a rare disease") for i in range(3)]
        + [pub(f"f{i}", 20 - i, False, "NHIS rare diseases registry") for i in range(2)]
    )
    llm = FakeLLM({"exclude_terms": ["rare diseases", "registry"], "flag_terms": [],
                   "new_aliases": [], "reasoning": ""})
    r = refine_candidate(QueryCandidate("X", safe=["NHIS"]), ev(pubs), settings=S, agent=llm,
                         dataset_name="X", count_fn=lambda t: 0, reference_count=100)
    assert r.candidate.exclude_terms == ["registry"]


def test_exclude_that_overlaps_an_alias_or_flag_is_rejected():
    # Genuine quotes use only the acronym; one FP spells the name out. An exclude
    # inside (or containing) an alias, or inside a flag, would silently drop it.
    pubs = (
        [pub(f"g{i}", 40 - i, True, "we used NHIS 2015 data") for i in range(3)]
        + [pub(f"f{i}", 20 - i, False,
               "the National Health Interview Survey reported in a NHIS registry cohort") for i in range(2)]
    )
    llm = FakeLLM({"exclude_terms": ["Health Interview", "National Health Interview Survey",
                                     "NHIS registry", "cohort", "registry"],
                   "flag_terms": [], "new_aliases": [], "reasoning": ""})
    cand = QueryCandidate("X", safe=["NHIS"], risky=["National Health Interview Survey"],
                          flag_terms=["cohort study"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=llm,
                         dataset_name="X", count_fn=lambda t: 0, reference_count=100)
    assert r.candidate.exclude_terms == ["registry"]


def test_alias_mostly_mentioned_not_used_is_kept():
    # The CPS-FSS case: most papers cite the survey without using it. They are
    # correct matches for the query, so the official name must not be dropped.
    c = QueryCandidate(main_dataset_name="Survey", safe=["Food Security Supplement", "FSS"],
                       risky=[], flag_terms=["USDA"])
    pubs = [pub("u0", 10, True, "we analysed the Food Security Supplement")] + [
        pub(f"m{i}", 9 - i, False, "according to the Food Security Supplement", mentioned=True)
        for i in range(5)]
    r = refine_candidate(c, ev(pubs), settings=S, agent=None, dataset_name="Survey")
    d = diff(r, "alias", "Food Security Supplement")
    assert (d.action, d.status) == ("keep", "verified")
    assert d.stats["matched"] == 6 and d.stats["used"] == 1 and d.stats["fp"] == 0
    assert "Food Security Supplement" in r.candidate.safe


def test_exclude_never_comes_from_a_mentioned_paper():
    c = QueryCandidate(main_dataset_name="Survey", safe=["NHIS"], risky=[], flag_terms=[])
    pubs = [pub("m0", 9, False, "NHIS nursing home index", mentioned=True),
            pub("m1", 8, False, "NHIS cited", mentioned=True),
            pub("n0", 7, False, "NHIS hospital ward", mentioned=False),
            pub("n1", 5, False, "NHIS hospital ward staff", mentioned=False),
            pub("u0", 6, True, "we use NHIS microdata")]
    llm = FakeLLM({"exclude_terms": ["nursing home", "hospital ward"], "flag_terms": [], "new_aliases": []})
    r = refine_candidate(c, ev(pubs), settings=S, agent=llm, dataset_name="Survey")
    assert r.candidate.exclude_terms == ["hospital ward"]


def ctx_pub(pid, cites, verdict, title, snippets, categories=(), quote=""):
    """Production shape: a not-mentioned paper has no verdict entry and no quote;
    what it carries is snippets of the query terms in its full text."""
    vd = [] if verdict == "not_mentioned" else [
        {"name": "X", "is_used": verdict == "used", "is_mentioned": True, "text_quote": quote}]
    return {"publication_id": pid, "doi": None, "title": title, "citation_count": cites,
            "fulltext": {"status": "delivered"}, "analyzed": True, "validated_datasets": vd,
            "term_contexts": [{"term": t, "role": "alias", "part": "body", "text": x} for t, x in snippets],
            "category_for": list(categories)}


UI_PUBS = (
    [ctx_pub(f"g{i}", 50 - i, "used", "Earnings after layoffs", [("UI", "linked UI wage records from the BLS")],
             ["Economics"], quote="we use UI wage records") for i in range(3)]
    + [ctx_pub(f"h{i}", 40 - i, "not_mentioned", f"App study {i}", [("UI", "the mobile app user interface (UI) design")],
               ["Information and Computing Sciences"]) for i in range(4)]
)


def test_snippets_attribute_a_false_positive_to_the_alias_that_matched_it():
    c = QueryCandidate(main_dataset_name="UI wage records", safe=["UI", "UI wage records"], risky=[],
                       flag_terms=["BLS"])
    r = refine_candidate(c, ev(UI_PUBS), settings=S, agent=None, dataset_name="UI wage records")
    d = diff(r, "alias", "UI")
    # Titles alone ("App study") never carry "UI"; the full-text snippets do.
    assert d.stats["fp"] == 4 and d.stats["matched"] == 3
    assert d.action == "demote"


def test_groups_become_notes_and_excludes_need_two_papers_and_an_impact():
    c = QueryCandidate(main_dataset_name="UI wage records", safe=["UI wage records"], risky=["UI"],
                       flag_terms=["BLS"])
    llm = FakeLLM({"false_positive_groups": [
        {"label": "UI = user interface", "cause_term": "UI", "papers": 4,
         "exclude_terms": ["user interface", "mobile app", "earnings"]}],
        "exclude_terms": ["App study 1"], "flag_terms": ["wage records"], "new_aliases": [], "reasoning": ""})
    impacts = {"user interface": 900, "mobile app": 0}
    r = refine_candidate(c, ev(UI_PUBS), settings=S, agent=llm, dataset_name="UI wage records",
                         impact_fn=lambda t: impacts[t])
    added = {d.term: d for d in r.diffs if d.kind == "exclude" and d.action == "add"}
    # "App study 1" is in one title only; "mobile app" removes nothing; "earnings" is in a match title.
    assert list(added) == ["user interface"]
    assert added["user interface"].stats == {"fp_papers": 4, "removes": 900}
    assert "removes 900 of the current results" in added["user interface"].reason
    assert any("false-positive group 'UI = user interface'" in n for n in r.notes)
    assert any("'Information and Computing Sciences'" in n for n in r.notes)
    flags = [d.term for d in r.diffs if d.kind == "flag" and d.action == "add"]
    assert flags == []   # lower-case common words never become a flag
    assert any("wage records (not a proper name (lower-case word))" in n for n in r.notes)


def test_flag_needs_a_share_of_the_matching_papers():
    pubs = [ctx_pub(f"g{i}", 50 - i, "used", "t", [("UI", "UI records")], quote="UI records") for i in range(10)]
    pubs[0]["term_contexts"][0]["text"] = "UI records from the state agency"
    pubs[1]["term_contexts"][0]["text"] = "UI records from the state agency"
    c = QueryCandidate(main_dataset_name="UI", safe=["UI"], risky=[], flag_terms=[])
    llm = FakeLLM({"exclude_terms": [], "flag_terms": ["State Agency"], "new_aliases": [], "reasoning": ""})
    strict = Settings(refine_min_alias_papers=3, fulltext_min_evidence=3, refine_flag_min_share=0.3)
    r = refine_candidate(c, ev(pubs), settings=strict, agent=llm, dataset_name="UI")
    assert not [d for d in r.diffs if d.kind == "flag" and d.action == "add"]   # 2/10 < 30%
    loose = Settings(refine_min_alias_papers=3, fulltext_min_evidence=3, refine_flag_min_share=0.2)
    r = refine_candidate(c, ev(pubs), settings=loose, agent=llm, dataset_name="UI")
    assert [d.term for d in r.diffs if d.kind == "flag" and d.action == "add"] == ["State Agency"]


def test_prompt_shows_snippets_and_fields_for_false_positives():
    llm = FakeLLM({"exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""})
    seen = {}
    orig = llm.get_structured

    def capture(prompt, tool):
        seen["p"] = prompt
        return orig(prompt, tool)

    llm.get_structured = capture
    c = QueryCandidate(main_dataset_name="UI", safe=["UI"], risky=[], flag_terms=[])
    refine_candidate(c, ev(UI_PUBS), settings=S, agent=llm, dataset_name="UI")
    fp_part = seen["p"].split("does NOT mention")[1]
    assert "user interface (UI) design" in fp_part and "Information and Computing Sciences" in fp_part


def _many_fp_pubs():
    # e0 is in 4 false positives, e1 in 3, e2 in 2.
    pubs = [ctx_pub(f"g{i}", 50 - i, "used", "t", [("UI", "UI wage records")], quote="UI wage records")
            for i in range(3)]
    for i in range(4):
        words = ["e0"] + (["e1"] if i < 3 else []) + (["e2"] if i < 2 else [])
        pubs.append(ctx_pub(f"h{i}", 20 - i, "not_mentioned", "x",
                            [("UI", "UI " + " ".join(f"word{w}" for w in words))]))
    return pubs


def _ui_llm(excludes):
    return FakeLLM({"exclude_terms": excludes, "flag_terms": [], "new_aliases": [], "reasoning": ""})


def test_impact_checks_are_capped_strongest_first():
    c = QueryCandidate(main_dataset_name="UI", safe=["UI"], risky=[], flag_terms=[])
    checked = []

    def impact(t):
        checked.append(t)
        return 10

    capped = Settings(refine_min_alias_papers=3, fulltext_min_evidence=3, refine_max_exclude_checks=2)
    r = refine_candidate(c, ev(_many_fp_pubs()), settings=capped, agent=_ui_llm(["worde2", "worde0", "worde1"]),
                         dataset_name="UI", impact_fn=impact)
    assert checked == ["worde0", "worde1"]
    assert r.candidate.exclude_terms == ["worde0", "worde1"]


def test_exclude_with_unknown_impact_is_rejected():
    c = QueryCandidate(main_dataset_name="UI", safe=["UI"], risky=[], flag_terms=[])

    def impact(t):
        raise RuntimeError("Dimensions budget reserve reached")

    r = refine_candidate(c, ev(_many_fp_pubs()), settings=S, agent=_ui_llm(["worde0"]),
                         dataset_name="UI", impact_fn=impact)
    assert r.candidate.exclude_terms == []


def test_scalar_groups_are_ignored():
    c = QueryCandidate(main_dataset_name="UI", safe=["UI"], risky=[], flag_terms=[])
    llm = FakeLLM({"false_positive_groups": 3, "exclude_terms": [], "flag_terms": [], "new_aliases": [],
                   "reasoning": ""})
    r = refine_candidate(c, ev(_many_fp_pubs()), settings=S, agent=llm, dataset_name="UI")
    assert not any("false-positive group" in n for n in r.notes)


def test_generic_flags_are_rejected_and_rejections_are_noted():
    pubs = (
        [pub(f"g{i}", 40 - i, True, "state UI wage records administrative data and Administrative Records from the BLS (Bureau of the Census)") for i in range(3)]
        + [pub(f"f{i}", 20 - i, False, "UI design study") for i in range(2)]
    )
    llm = FakeLLM({"exclude_terms": [], "flag_terms": ["state", "administrative data", "state UI", "Administrative Records", "BLS", "Wage Records",
                                  "Bureau of the Census"],
                   "new_aliases": [], "reasoning": ""})
    counts = {"Administrative Records": 2_000_000, "BLS": 90_000, "Bureau of the Census": 80_000}

    def count(t):
        if t not in counts:
            raise RuntimeError("budget")
        return counts[t]

    r = refine_candidate(QueryCandidate("X", safe=["UI wage records data"]), ev(pubs), settings=S, agent=llm,
                         dataset_name="X", count_fn=count, reference_count=100)
    assert [d.term for d in r.diffs if d.kind == "flag" and d.action == "add"] == ["BLS", "Bureau of the Census"]
    note = next(n for n in r.notes if n.startswith("rejected flag candidates"))
    assert "state UI (not a proper name (lower-case word))" in note
    assert "state (not a proper name (lower-case word))" in note
    assert "administrative data (not a proper name (lower-case word))" in note
    assert "Administrative Records (2000000 Dimensions hits" in note
    assert "Wage Records (hit count unknown" in note
    assert diff(r, "flag", "BLS").stats["dimensions_count"] == 90_000


def test_partial_acronym_snippets_attribute_false_positives_to_the_alias():
    def fp(i):
        p = pub(f"f{i}", 30 - i, False, "")
        p["term_contexts"] = [{"term": "WR dataset", "role": "alias", "part": "body", "match": "partial",
                               "text": "workflow recognition (WR) on assembly lines"}]
        return p

    pubs = [pub(f"g{i}", 40 - i, True, "we used state UI wage records") for i in range(3)] + [fp(i) for i in range(4)]
    cand = QueryCandidate("X", safe=["WR dataset", "UI wage records"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=None, dataset_name="X")
    drop = [d for d in r.diffs if d.kind == "alias" and d.action == "drop"]
    assert [d.term for d in drop] == ["WR dataset"]
    assert drop[0].stats["fp"] == 4


def test_partial_snippet_goes_to_the_shortest_alias_sharing_the_acronym():
    def fp(i):
        p = pub(f"f{i}", 30 - i, False, "")
        # extraction tagged the first alias with the acronym (the precise one)
        p["term_contexts"] = [{"term": "WR Longitudinal Panel", "role": "alias", "part": "body",
                               "match": "partial", "text": "workflow recognition (WR)"}]
        return p

    pubs = [pub(f"g{i}", 40 - i, True, "we used the WR Longitudinal Panel") for i in range(4)] + [fp(i) for i in range(8)]
    cand = QueryCandidate("X", safe=["WR Longitudinal Panel"], risky=["WR dataset"], flag_terms=["BLS"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=None, dataset_name="X")
    assert diff(r, "alias", "WR dataset").action == "drop"
    assert diff(r, "alias", "WR Longitudinal Panel").action == "keep"
    assert diff(r, "alias", "WR Longitudinal Panel").stats["fp"] == 0


def test_new_alias_count_rejections_are_noted():
    pubs = [pub(f"g{i}", 40 - i, True, "we used the NHIS Public Use Files") for i in range(3)]
    llm = FakeLLM({"exclude_terms": [], "flag_terms": [], "new_aliases": ["NHIS Public Use Files"], "reasoning": ""})
    r = refine_candidate(QueryCandidate("X", safe=["NHIS"]), ev(pubs), settings=S, agent=llm,
                         dataset_name="X", count_fn=lambda t: 0, reference_count=100)
    assert any("NHIS Public Use Files (0 Dimensions hits)" in n for n in r.notes)


def test_new_alias_shaped_like_acronym_suffix_is_rejected():
    pubs = [pub(f"g{i}", 40 - i, True, "we linked the WR dataset and UI data from the Department of Workforce Services")
            for i in range(4)]
    llm = FakeLLM({"exclude_terms": [], "flag_terms": [], "new_aliases": ["WR dataset", "UI data"], "reasoning": ""})
    calls = []
    cand = QueryCandidate("X", safe=["Wage Record Data"], flag_terms=["Department of Workforce Services"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=llm, dataset_name="X",
                         count_fn=lambda t: calls.append(t) or 900, reference_count=200)
    assert not [d for d in r.diffs if d.kind == "alias" and d.action == "add"]
    assert calls == []   # rejected before any Dimensions count
    note = next(n for n in r.notes if n.startswith("rejected alias candidates"))
    assert "WR dataset (short acronym plus generic suffix)" in note and "UI data (short acronym" in note


def test_flag_proposed_twice_in_different_case_costs_one_count_and_covered_is_noted():
    pubs = [pub(f"g{i}", 40 - i, True, "we used the NHIS files from the Census Bureau") for i in range(4)]
    llm = FakeLLM({"exclude_terms": [], "flag_terms": ["Census Bureau", "census bureau", "BLS"],
                   "new_aliases": [], "reasoning": ""})
    calls = []

    def count(t):
        calls.append(t)
        return 10

    cand = QueryCandidate("X", safe=["NHIS"], flag_terms=["BLS"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=llm, dataset_name="X", count_fn=count)
    assert calls == ["Census Bureau"]
    assert any("already in the query" in n and "BLS" in n for n in r.notes)


def test_group_note_shows_the_verified_count():
    def fp(i):
        p = pub(f"f{i}", 30 - i, False, "")
        p["validated_datasets"] = []
        p["term_contexts"] = [{"term": "NHIS", "role": "alias", "part": "body", "text": "NHIS nursing home"}]
        return p

    pubs = [pub(f"g{i}", 40 - i, True, "we used Alpha Survey data") for i in range(3)] + [fp(i) for i in range(4)]
    llm = FakeLLM({"false_positive_groups": [{"label": "nursing", "cause_term": "NHIS", "papers": 6,
                                              "exclude_terms": []}],
                   "exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""})
    r = refine_candidate(QueryCandidate("X", safe=["Alpha Survey"]), ev(pubs), settings=S, agent=llm,
                         dataset_name="X")
    assert any("(~6 per LLM; cause term 'NHIS' occurs in 4 false positive(s))" in n for n in r.notes)


def test_refiner_llm_without_a_usable_result_is_noted():
    def fp(i):
        p = pub(f"f{i}", 30 - i, False, "")
        p["validated_datasets"] = []
        p["term_contexts"] = [{"term": "NHIS", "role": "alias", "part": "body", "text": "NHIS nursing home"}]
        return p

    pubs = [pub(f"g{i}", 40 - i, True, "we used Alpha Survey data") for i in range(3)] + [fp(i) for i in range(4)]

    class Boom(FakeLLM):
        def get_structured(self, prompt, result_tool):
            raise RuntimeError("timeout")

    empty4 = {"exclude_terms": [], "flag_terms": [], "new_aliases": [], "false_positive_groups": []}
    marker = "refiner LLM returned no usable result"
    for llm in (FakeLLM({}), Boom({})):
        r = refine_candidate(QueryCandidate("X", safe=["Alpha Survey"]), ev(pubs), settings=S, agent=llm,
                             dataset_name="X")
        assert any(marker in n for n in r.notes)
        assert r.changed is False
    r = refine_candidate(QueryCandidate("X", safe=["Alpha Survey"]), ev(pubs), settings=S,
                         agent=FakeLLM(empty4), dataset_name="X")
    assert not any(marker in n for n in r.notes)


def test_alias_evidence_falls_back_to_snippet_when_the_quote_is_empty():
    def fp(i):
        p = pub(f"f{i}", 30 - i, False, "")
        p["validated_datasets"] = []
        p["term_contexts"] = [{"term": "NHIS", "role": "alias", "part": "body", "text": "NHIS nursing home"}]
        return p

    pubs = [fp(i) for i in range(4)] + [pub(f"g{i}", 40 - i, True, "Alpha Survey data") for i in range(3)]
    cand = QueryCandidate("X", safe=["NHIS", "Alpha Survey"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=None, dataset_name="X")
    assert diff(r, "alias", "NHIS").action == "drop"
    assert all(e.quote == "NHIS nursing home" for e in diff(r, "alias", "NHIS").evidence)


def test_prompt_asks_for_proper_name_flags():
    from dataset_agent.adapters.fulltext_refiner import propose_refinements

    seen = {}

    class Spy(FakeLLM):
        def get_structured(self, prompt, result_tool):
            seen["p"] = prompt
            return {}

    propose_refinements(Spy({}), dataset_name="X", genuine=[], false_positives=[], aliases=[],
                        flag_terms=[], exclude_terms=[])
    assert "proper name" in seen["p"] and "lower-case context word" in seen["p"]


def test_refinement_reply_without_reasoning_still_yields_proposals():
    """The refiner never reads 'reasoning', so a strict adapter must not discard a reply lacking it."""
    from llm_fakes import STYLES, FakeLLMServer, make_agent

    from dataset_agent.adapters.fulltext_refiner import propose_refinements
    from dataset_agent.adapters.tools import EMIT_REFINEMENT_CANDIDATES

    reply = {
        "false_positive_groups": [{"label": "UI = user interface", "cause_term": "UI",
                                   "exclude_terms": ["user interface"]}],
        "exclude_terms": ["interface design"],
        "flag_terms": ["Census Bureau"],
        "new_aliases": ["Alpha Survey"],
    }
    schema = EMIT_REFINEMENT_CANDIDATES["input_schema"]
    assert "reasoning" not in schema["required"]
    for style in STYLES:
        server = FakeLLMServer(style)
        server.push(server.structured("emit_refinement_candidates", reply))
        agent = make_agent(style, server)
        out = propose_refinements(agent, dataset_name="X", genuine=[], false_positives=[],
                                  aliases=[], flag_terms=[], exclude_terms=[])
        assert len(server.requests) == 1
        assert out["flag_terms"] == ["Census Bureau"]
        assert out["new_aliases"] == ["Alpha Survey"]
        assert "interface design" in out["exclude_terms"] and "user interface" in out["exclude_terms"]
        assert out["groups"][0]["papers"] == 0


@pytest.mark.parametrize("mutate", [
    lambda r: r["false_positive_groups"][0].pop("cause_term"),
    lambda r: r["false_positive_groups"][0].pop("label"),
    lambda r: r["false_positive_groups"][0].update(papers=2.5),
    lambda r: r.update(new_aliases=None),
    lambda r: r.update(flag_terms=None),
    lambda r: r.update(false_positive_groups=None),
])
def test_refinement_malformed_group_or_null_list_keeps_valid_proposals(mutate):
    from llm_fakes import STYLES, FakeLLMServer, make_agent

    from dataset_agent.adapters.fulltext_refiner import propose_refinements

    for style in STYLES:
        reply = {
            "false_positive_groups": [{"label": "UI = user interface", "cause_term": "UI",
                                       "papers": 3, "exclude_terms": ["user interface"]}],
            "exclude_terms": ["usability study"], "flag_terms": ["Department of Labor"],
            "new_aliases": ["UI records"],
        }
        mutate(reply)
        server = FakeLLMServer(style)
        server.push(server.structured("emit_refinement_candidates", reply))
        server.push(server.structured("emit_refinement_candidates", reply))
        agent = make_agent(style, server)
        out = propose_refinements(agent, dataset_name="X", genuine=[], false_positives=[],
                                  aliases=[], flag_terms=[], exclude_terms=[])
        assert "usability study" in out["exclude_terms"], style
        if "new_aliases" in reply and reply["new_aliases"] is not None:
            assert out["new_aliases"] == ["UI records"], style
        if reply.get("flag_terms") is not None:
            assert out["flag_terms"] == ["Department of Labor"], style


_AOU_QUOTE = "we used AoU data in the All of Us Researcher Workbench"


def _aou_candidate(**kw):
    base = dict(safe=["AoU survey", "All of Us NIH", "All of Us survey"],
                flag_terms=["All of Us Research Program", "National Institutes of Health", "NIH"])
    base.update(kw)
    return QueryCandidate("All of Us", **base)


def test_flag_equal_to_new_alias_is_rejected_aou_shape():
    from dataset_agent.adapters.seed_mapping import candidate_clause

    pubs = [pub(f"g{i}", 40 - i, True, _AOU_QUOTE) for i in range(4)]
    llm = FakeLLM({"exclude_terms": [], "flag_terms": ["AoU"],
                   "new_aliases": ["AoU", "All of Us Researcher Workbench"], "reasoning": ""})
    calls = []
    r = refine_candidate(_aou_candidate(), ev(pubs), settings=S, agent=llm, dataset_name="All of Us",
                         count_fn=lambda t: calls.append(t) or 90_000, reference_count=100)
    # A bare short acronym is never added as an alias, and the refused alias cannot come
    # back as a flag either.
    assert "AoU" not in r.candidate.risky + r.candidate.safe
    assert "aou" not in [f.casefold() for f in r.candidate.flag_terms]
    assert any("AoU (bare short acronym)" in n for n in r.notes)
    assert any("AoU (already an alias)" in n for n in r.notes)
    assert calls.count("AoU") == 0
    clause = candidate_clause(r.candidate)
    assert '"AoU"' not in clause.replace('\\"', '"')


def test_flag_equal_to_current_alias_is_rejected():
    pubs = [pub(f"g{i}", 40 - i, True, _AOU_QUOTE) for i in range(4)]
    llm = FakeLLM({"exclude_terms": [], "flag_terms": ["aou"], "new_aliases": [], "reasoning": ""})
    calls = []
    cand = _aou_candidate(risky=["AoU"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=llm, dataset_name="All of Us",
                         count_fn=lambda t: calls.append(t) or 5, reference_count=100)
    assert calls == []
    assert any("aou (already an alias)" in n for n in r.notes)
    assert "aou" not in [f.casefold() for f in r.candidate.flag_terms]


def test_new_alias_equal_to_existing_flag_is_rejected():
    pubs = [pub(f"g{i}", 40 - i, True, "we used AoU data with NIH funding") for i in range(4)]
    llm = FakeLLM({"exclude_terms": [], "flag_terms": [], "new_aliases": ["NIH"], "reasoning": ""})
    r = refine_candidate(_aou_candidate(), ev(pubs), settings=S, agent=llm, dataset_name="All of Us",
                         count_fn=lambda t: 5, reference_count=100)
    assert "NIH" not in r.candidate.safe and "NIH" not in r.candidate.risky
    assert any("NIH (already a flag term)" in n for n in r.notes)


def test_flag_not_colliding_still_accepted():
    pubs = [pub(f"g{i}", 40 - i, True, _AOU_QUOTE) for i in range(4)]
    llm = FakeLLM({"exclude_terms": [], "flag_terms": ["All of Us Researcher Workbench"],
                   "new_aliases": ["AoU"], "reasoning": ""})
    r = refine_candidate(_aou_candidate(), ev(pubs), settings=S, agent=llm, dataset_name="All of Us",
                         count_fn=lambda t: 5, reference_count=100)
    assert "All of Us Researcher Workbench" in r.candidate.flag_terms
    assert "AoU" not in r.candidate.safe + r.candidate.risky   # bare short acronym


def test_new_alias_aou_dataset_is_rejected_but_add_health_is_not():
    pubs = [pub(f"g{i}", 40 - i, True, "participants of the AoU dataset and Add Health were linked in All of Us")
            for i in range(4)]
    llm = FakeLLM({"exclude_terms": [], "flag_terms": [], "new_aliases": ["AoU dataset"], "reasoning": ""})
    cand = QueryCandidate("X", safe=["All of Us"], flag_terms=["NIH"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=llm, dataset_name="X",
                         count_fn=lambda t: 900, reference_count=200)
    assert not [d for d in r.diffs if d.kind == "alias" and d.action == "add"]
    note = next(n for n in r.notes if n.startswith("rejected alias candidates"))
    assert "AoU dataset (short acronym plus generic suffix)" in note
    from dataset_agent.adapters.query_heuristics import is_acronym_suffix_form
    assert not is_acronym_suffix_form("Add Health")


# --- gnss-n11 follow-up: sub-phrase flags, short-acronym aliases, accepted-change logging ---

def _gnss_refine(proposal, counted=None):
    pubs = [pub(f"g{i}", 40 - i, True,
                "NHIS stations from PANGA and NOTA archived by EarthScope and the GAGE Facility")
            for i in range(3)] + [pub("f0", 5, False, "NHIS nursing"), pub("f1", 4, False, "NHIS nursing")]
    out = {"exclude_terms": [], "flag_terms": [], "new_aliases": [], "reasoning": ""}
    out.update(proposal)

    def count(t):
        if counted is not None:
            counted.append(t)
        return 5

    cand = QueryCandidate("X", safe=["NHIS"], flag_terms=["EarthScope Consortium"])
    return refine_candidate(cand, ev(pubs), settings=S, agent=FakeLLM(out), dataset_name="X",
                            count_fn=count, reference_count=100)


def test_new_flag_subphrase_of_existing_flag_rejected():
    r = _gnss_refine({"flag_terms": ["EarthScope"]})
    assert r.candidate.flag_terms == ["EarthScope Consortium"]
    assert "EarthScope (broader form of an existing flag)" in " ".join(r.notes)


def test_new_flag_unrelated_still_accepted():
    r = _gnss_refine({"flag_terms": ["GAGE Facility"]})
    assert "GAGE Facility" in r.candidate.flag_terms


def test_new_alias_short_acronym_rejected():
    counted: list[str] = []
    r = _gnss_refine({"new_aliases": ["NOTA", "PANGA"]}, counted)
    added = [d.term for d in r.diffs if d.kind == "alias" and d.action == "add"]
    assert added == ["PANGA"] and "NOTA" not in counted
    assert "NOTA (bare short acronym)" in " ".join(r.notes)


def test_term_proposed_as_alias_and_flag_is_judged_as_a_flag_only():
    counted: list[str] = []
    r = _gnss_refine({"new_aliases": ["GAGE Facility", "PANGA"], "flag_terms": ["GAGE Facility"]},
                     counted)
    added = [d.term for d in r.diffs if d.kind == "alias" and d.action == "add"]
    assert added == ["PANGA"]
    assert "GAGE Facility" not in r.candidate.safe and "GAGE Facility" not in r.candidate.risky
    assert "GAGE Facility (also proposed as a flag term)" in " ".join(r.notes)
    assert "GAGE Facility" in r.candidate.flag_terms
    assert counted.count("GAGE Facility") == 1


def test_accepted_changes_are_logged(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="dataset_agent.adapters.fulltext_refiner"):
        r = _gnss_refine({"new_aliases": ["PANGA"], "flag_terms": ["GAGE Facility"]})
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any(m.startswith("refine: alias 'PANGA' accepted:") for m in msgs)
    assert any(m.startswith("refine: flag 'GAGE Facility' accepted:") for m in msgs)
    assert any(n.startswith("refiner added: aliases [PANGA]; flags [GAGE Facility]") for n in r.notes)


def test_no_field_pattern_without_matching_papers():
    from dataset_agent.adapters.fulltext_refiner import category_patterns

    def fps():
        return [ctx_pub(f"f{i}", 10 - i, "not_mentioned", "t", [], categories=["37 Earth Sciences"])
                for i in range(5)]

    def matches(n):
        return [ctx_pub(f"g{i}", 50 - i, "used", "t", [], categories=["46 Information"]) for i in range(n)]

    def evi(m):
        return ev(fps() + m)

    assert category_patterns(evi([]), 3) == []
    assert category_patterns(evi(matches(2)), 3) == []
    out = category_patterns(evi(matches(3)), 3)
    assert len(out) == 1 and "vs 0/3 matches" in out[0]


def test_mixed_case_proper_names_pass_the_flag_name_rule():
    from dataset_agent.adapters.fulltext_refiner import _generic_word

    for name in ("dbGaP", "eICU", "gnomAD", "eBird", "eICU Collaborative Research Database",
                 "Census Bureau"):
        assert _generic_word(name) is False, name
    for name in ("state UI", "administrative data", "employer", "Gen3 data commons",
                 "data.midrc.org", "gen3"):
        assert _generic_word(name) is True, name


def _gen_pubs(quote):
    return [pub(f"g{i}", 40 - i, True, quote) for i in range(3)] + [
        pub("f0", 5, False, "NHIS nursing"), pub("f1", 4, False, "NHIS nursing")]


def _rejected_note(r, kind):
    return next(n for n in r.notes if n.startswith(f"rejected {kind} candidates"))


def test_bare_short_acronym_flag_is_rejected_when_flags_exist():
    pubs = _gen_pubs("we used NHIS data from the PMI cohort")
    out = {"exclude_terms": [], "flag_terms": ["PMI"], "new_aliases": [], "reasoning": ""}
    cand = QueryCandidate("X", safe=["NHIS"], flag_terms=["Census Bureau"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=FakeLLM(out), dataset_name="X",
                         count_fn=lambda t: 5, reference_count=100)
    assert "PMI" not in r.candidate.flag_terms
    assert "PMI (bare short acronym)" in _rejected_note(r, "flag")


def test_rhetorical_alias_is_rejected():
    pubs = _gen_pubs("we used NHIS data from All of Us")
    r = _refine_with(pubs, new_aliases=["All of Us"])
    assert not [d for d in r.diffs if d.kind == "alias" and d.action == "add"]
    assert "All of Us (rhetorical phrase)" in _rejected_note(r, "alias")


def test_flag_inside_a_flag_gated_alias_is_rejected():
    pubs = _gen_pubs("NHIS data and the Precision Medicine Initiative Cohort Program")
    out = {"exclude_terms": [], "flag_terms": ["Precision Medicine Initiative"], "new_aliases": [],
           "reasoning": ""}
    cand = QueryCandidate("X", safe=["NHIS"], risky=["Precision Medicine Initiative Cohort Program"],
                          flag_terms=["Census Bureau"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=FakeLLM(out), dataset_name="X",
                         count_fn=lambda t: 5, reference_count=100)
    assert "Precision Medicine Initiative" not in r.candidate.flag_terms
    assert "Precision Medicine Initiative (contained in a flag-gated alias)" in _rejected_note(r, "flag")


def test_flag_inside_a_new_risky_alias_accepted_in_the_same_round_is_rejected():
    pubs = _gen_pubs("NHIS data and the Precision Medicine Initiative Cohort Program")
    out = {"exclude_terms": [], "flag_terms": ["Precision Medicine Initiative"],
           "new_aliases": ["Precision Medicine Initiative Cohort Program"], "reasoning": ""}
    cand = QueryCandidate("X", safe=["NHIS"], flag_terms=["Census Bureau"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=FakeLLM(out), dataset_name="X",
                         count_fn=lambda t: 5000, reference_count=100)
    assert "Precision Medicine Initiative Cohort Program" in r.candidate.risky
    assert "Precision Medicine Initiative" not in r.candidate.flag_terms
    assert "contained in a flag-gated alias" in _rejected_note(r, "flag")


def test_new_risky_alias_is_rejected_when_the_contained_flag_was_its_only_gate():
    pubs = _gen_pubs("NHIS data and the Precision Medicine Initiative Cohort Program")
    out = {"exclude_terms": [], "flag_terms": ["Precision Medicine Initiative"],
           "new_aliases": ["Precision Medicine Initiative Cohort Program"], "reasoning": ""}
    cand = QueryCandidate("X", safe=["NHIS"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=FakeLLM(out), dataset_name="X",
                         count_fn=lambda t: 5000, reference_count=100)
    assert r.candidate.flag_terms == []
    assert r.candidate.risky == []
    assert not [d for d in r.diffs if d.kind == "alias" and d.action == "add"]
    assert "risky and there are no flag terms" in _rejected_note(r, "alias")


def test_demoted_safe_alias_returns_to_safe_when_its_only_new_flag_was_contained():
    pubs = ([pub(f"g{i}", 40 - i, True, "NHIS data and the Beta Survey Program") for i in range(3)]
            + [pub(f"b{i}", 20 - i, False, "the Beta Survey Program sample") for i in range(4)])
    out = {"exclude_terms": [], "flag_terms": ["Beta Survey"], "new_aliases": [], "reasoning": ""}
    cand = QueryCandidate("X", safe=["NHIS", "Beta Survey Program"])
    r = refine_candidate(cand, ev(pubs), settings=S, agent=FakeLLM(out), dataset_name="X",
                         count_fn=lambda t: 5, reference_count=100)
    assert sorted(r.candidate.safe) == ["Beta Survey Program", "NHIS"]
    assert r.candidate.risky == [] and r.candidate.flag_terms == []
    assert not [d for d in r.diffs if d.action == "demote"]
    rej = [n for n in r.notes if n.startswith("rejected alias candidates")]
    assert not any("Beta Survey Program" in n for n in rej)
    assert "contained in a flag-gated alias" in _rejected_note(r, "flag")

from __future__ import annotations

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
    assert "1 full-text snippet(s)/quote(s) and 1 title(s) only" in d.reason


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
    assert flags == ["wage records"]


def test_flag_needs_a_share_of_the_matching_papers():
    pubs = [ctx_pub(f"g{i}", 50 - i, "used", "t", [("UI", "UI records")], quote="UI records") for i in range(10)]
    pubs[0]["term_contexts"][0]["text"] = "UI records from the state agency"
    pubs[1]["term_contexts"][0]["text"] = "UI records from the state agency"
    c = QueryCandidate(main_dataset_name="UI", safe=["UI"], risky=[], flag_terms=[])
    llm = FakeLLM({"exclude_terms": [], "flag_terms": ["state agency"], "new_aliases": [], "reasoning": ""})
    strict = Settings(refine_min_alias_papers=3, fulltext_min_evidence=3, refine_flag_min_share=0.3)
    r = refine_candidate(c, ev(pubs), settings=strict, agent=llm, dataset_name="UI")
    assert not [d for d in r.diffs if d.kind == "flag" and d.action == "add"]   # 2/10 < 30%
    loose = Settings(refine_min_alias_papers=3, fulltext_min_evidence=3, refine_flag_min_share=0.2)
    r = refine_candidate(c, ev(pubs), settings=loose, agent=llm, dataset_name="UI")
    assert [d.term for d in r.diffs if d.kind == "flag" and d.action == "add"] == ["state agency"]


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

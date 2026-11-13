"""Recall check: candidate selection, complement shape and the promotion decision."""

from __future__ import annotations

import pytest

from dataset_agent.adapters.fulltext_evidence import Evidence, PubVerdict
from dataset_agent.adapters.fulltext_recall import judge_complement, recall_candidates
from dataset_agent.adapters.seed_mapping import (
    QueryCandidate,
    candidate_to_seed,
    complement_candidate,
    complement_clause,
)
from dataset_agent.settings import Settings

AOU_FLAGS = ["NIH", "National Institutes of Health", "All of Us Research Program", "Data and Research Center"]
AOU = QueryCandidate(main_dataset_name="All of Us", safe=["All of Us survey"], flag_terms=AOU_FLAGS)
MIDRC_FLAGS = ["MIDRC", "Medical Imaging and Data Resource Center", "NIBIB", "NCI", "National Cancer Institute"]
MIDRC = QueryCandidate(main_dataset_name="MIDRC Data Commons", safe=["MIDRC Data Commons"],
                       flag_terms=MIDRC_FLAGS)


def run(cand, names, flags=None, known=(), against=()):
    return recall_candidates(cand, main_names=list(names),
                             known_aliases=list(known), flag_terms=list(flags if flags is not None else cand.flag_terms),
                             evidence_against=set(against))


def terms(res):
    return [c.term for c in res[0]]


def test_aou_only_the_program_name_is_eligible():
    res = run(AOU, ["All of Us"])
    assert terms(res) == ["All of Us Research Program"]
    notes = "; ".join(res[1])
    assert "NIH (short acronym)" in notes
    assert "National Institutes of Health (not a name of the dataset)" in notes
    assert "Data and Research Center (generic phrase)" in notes
    assert "All of Us (rhetorical phrase)" in notes


def test_midrc_acronym_and_long_form_eligible_funders_rejected():
    res = run(MIDRC, ["MIDRC Data Commons"])
    assert terms(res) == ["MIDRC", "Medical Imaging and Data Resource Center"]
    notes = "; ".join(res[1])
    for t in ("NIBIB", "NCI", "National Cancer Institute"):
        assert t in notes


def test_head_acronym_alias_does_not_make_the_funder_related():
    cand = QueryCandidate(main_dataset_name="All of Us", safe=["All of Us NIH"],
                          flag_terms=["National Institutes of Health"])
    res = run(cand, ["All of Us"])
    assert res[0] == [] and "not a name of the dataset" in res[1][0]


def test_safety_rejections():
    cand = QueryCandidate(main_dataset_name="Plate Boundary Observatory Data",
                          safe=["Plate Boundary Observatory Data"], risky=["PBO network"],
                          flag_terms=["Plate Boundary Observatory"])
    known = ["AoU", "WR dataset", "PBO network", 'The "Big" Survey', "Refiner Dropped Name",
             "Refiner Demoted Name", "Plate Boundary Observatory Data", "lowercase generic phrase"]
    eligible, notes = run(cand, ["Plate Boundary Observatory Data"], known=known,
                          against=["Refiner Dropped Name", "refiner demoted name"])
    text = "; ".join(notes)
    assert "AoU (short acronym)" in text
    assert "WR dataset (short acronym plus generic suffix)" in text
    assert "unusable characters" in text
    assert "Refiner Dropped Name (full text already rejected it)" in text
    assert "Refiner Demoted Name (full text already rejected it)" in text
    assert "lowercase generic phrase (not a proper name)" in text
    # already safe: silently skipped
    assert "Plate Boundary Observatory Data" not in text
    assert "PBO network (not a proper name)" in text
    assert terms((eligible, notes)) == ["Plate Boundary Observatory"]


def test_risky_alias_without_flags_is_rejected():
    cand = QueryCandidate(main_dataset_name="Gamma Survey Data", safe=["Gamma Survey Data"],
                          risky=["Gamma Network Study"])
    eligible, notes = run(cand, ["Gamma Survey Data"], flags=[], known=["Gamma Network Study"])
    assert eligible == [] and "already matches without flags" in notes[0]


def test_tier_order_flags_and_names_before_risky_aliases():
    cand = QueryCandidate(main_dataset_name="Delta Cohort Study", safe=["Delta Cohort"],
                          risky=["Delta Cohort Study Group"], flag_terms=["Delta Cohort Study Center"])
    eligible, _ = run(cand, ["Delta Cohort Study"], known=["Delta Cohort Study Group"])
    assert [(c.term, c.tier) for c in eligible] == [
        ("Delta Cohort Study Center", 0), ("Delta Cohort Study", 0), ("Delta Cohort Study Group", 1)]


def test_complement_clause_shapes():
    safe_only = QueryCandidate("X", safe=["a b"], exclude_terms=["e f"])
    assert complement_clause("Term One", QueryCandidate("X", safe=["a b"])) == (
        '(\\"Term One\\") NOT (((\\"a b\\")))')
    assert complement_clause("Term One", safe_only) == (
        '(\\"Term One\\") NOT (((\\"a b\\")) OR (\\"e f\\"))')
    hybrid = QueryCandidate("X", safe=["a b"], risky=["r s"], flag_terms=["F G"], exclude_terms=["e f"])
    out = complement_clause("Term One", hybrid)
    assert out.startswith('(\\"Term One\\") NOT (')
    assert '((\\"r s\\") AND (\\"F G\\"))' in out and out.endswith('OR (\\"e f\\"))')
    assert complement_clause("Term One", hybrid, server_quoting=True) == out


def test_complement_candidate_safe_only_seed():
    cur = QueryCandidate("Main", safe=["a b"], exclude_terms=["e f"])
    seed = candidate_to_seed(complement_candidate("Term One", cur))
    assert seed["dimensions_query"] == '(\\"Term One\\") NOT (\\"a b\\" OR \\"e f\\")'
    assert seed["dataset_names"] == ["Term One"]


def test_complement_candidate_special_characters_use_structured_seed():
    cur = QueryCandidate("Main", safe=["R&D Survey"], exclude_terms=["e f"])
    seed = candidate_to_seed(complement_candidate("Term One", cur))
    assert seed["dimensions_query"] is None
    assert seed["standalone_names"] == ["Term One"]
    assert seed["exclude_terms"] == ["R&D Survey", "e f"]


def test_complement_candidate_hybrid_is_raw_and_special_chars_are_refused():
    cur = QueryCandidate("Main", safe=["a b"], risky=["r s"], flag_terms=["F G"])
    seed = candidate_to_seed(complement_candidate("Term One", cur))
    assert 'AND' in seed["dimensions_query"] and seed["dataset_names"] == ["Term One"]
    bad = QueryCandidate("Main", safe=["a b"], risky=["r&s"], flag_terms=["F G"])
    with pytest.raises(ValueError, match="not expressible"):
        complement_candidate("Term One", bad)


def ev(n_genuine, n_total, **kw):
    verdicts = [PubVerdict(publication_id=str(i), doi=None, title="t", citation_count=1,
                           verdict="used" if i < n_genuine else "not_mentioned", quote="Term One text")
                for i in range(n_total)]
    return Evidence(verdicts=verdicts, n_sampled=n_total, n_delivered=n_total, **kw)


S = Settings()


def test_judge_promotes_with_exact_reason():
    ok, reason, stats = judge_complement("Term One", ev(8, 10), 123, S)
    assert ok and reason == "recall: complement precision 0.80 on 10 papers; +123 Dimensions hits; 8 of 10 use the data"
    assert stats["added_hits"] == 123 and stats["term_in_text"] == 10


def test_judge_rejects_low_precision_and_shortfall():
    ok, reason, _ = judge_complement("Term One", ev(7, 10), 50, S)
    assert not ok and reason == "complement precision 0.70 on 10 papers (need 0.80)"
    ok, reason, _ = judge_complement(
        "Term One", ev(9, 9, target=20, stop_reason="pool_exhausted", batches=3, budget_exhausted=True), 50, S)
    assert not ok and "only 9 full-text paper(s) evaluated (need 10)" in reason
    assert "OpenAlex budget exhausted" in reason
    ok, reason, stats = judge_complement("Term One", ev(0, 0), 50, S)
    assert not ok and stats["complement_precision"] is None


def test_aou_real_main_name_with_agency_suffix_uses_known_alias():
    res = run(AOU, ["All of Us NIH"], known=["All of Us"])
    assert "All of Us Research Program" in terms(res)
    notes = "; ".join(res[1])
    assert "National Institutes of Health (not a name of the dataset)" in notes
    assert "NIH" in notes and "NIH" not in terms(res)


def test_generic_phrase_and_acronym_compound_rejected():
    cand = QueryCandidate(main_dataset_name="MIDRC Data Commons", safe=["MIDRC Data Commons"])
    res = run(cand, ["MIDRC Data Commons"], flags=[], known=["Data Commons", "PBO GPS"])
    assert terms(res) == []
    notes = "; ".join(res[1])
    assert "Data Commons (generic phrase)" in notes
    assert "PBO GPS (acronyms only)" in notes


# Real gpt-oss inputs (aou-g2 / midrc-g2 / gnss-b2): bare "All of Us" is not a known name.
AOU_G2 = QueryCandidate(
    main_dataset_name="All of Us NIH",
    safe=["All of Us NIH", "All of Us Researcher Workbench study", "All of Us Researcher Workspace",
          "All of Us Dataset"],
    risky=["All", "All of Us Data"],
    flag_terms=["All of Us Research Program", "NIH"],
)
AOU_G2_ALIASES = ["All of Us NIH", "All of Us Researcher Workbench study", "All of Us Researcher Workspace",
                  "All of Us Dataset", "All of Us Data", "All"]
MIDRC_G2_FLAGS = ["National Cancer Institute", "NCI", "Medical Imaging & Data Resource Center", "MIDRC",
                  "National Institutes of Health", "NIH", "Association of Academic Health Centers"]
GNSS_B2_FLAGS = ["NSF", "National Science Foundation", "UNAVCO", "EarthScope Consortium", "IRIS Consortium",
                 "Incorporated Research Institutions for Seismology"]


def test_aou_g2_shared_phrase_makes_research_program_eligible():
    res = run(AOU_G2, ["All of Us NIH"], known=AOU_G2_ALIASES)
    assert [c.term for c in res[0] if c.source == "flag"] == ["All of Us Research Program"]
    assert "All of Us Data" in terms(res)  # tier-1 risky alias, unchanged behavior
    assert "NIH (short acronym)" in "; ".join(res[1])


def test_midrc_g2_agency_flags_still_rejected():
    cand = QueryCandidate(main_dataset_name="MIDRC Data Commons",
                          safe=["MIDRC Data Commons", "MIDRC Commons"], flag_terms=MIDRC_G2_FLAGS)
    res = run(cand, ["MIDRC Data Commons"],
              known=["MIDRC Commons", "Medical Imaging & Data Resource Center Data Commons"])
    notes = "; ".join(res[1])
    for t in ("National Cancer Institute", "National Institutes of Health",
              "Association of Academic Health Centers"):
        assert f"{t} (not a name of the dataset)" in notes
    assert not {"NCI", "NIH", "National Cancer Institute", "National Institutes of Health",
                "Association of Academic Health Centers"} & set(terms(res))


def test_gnss_b2_agency_flags_still_rejected():
    cand = QueryCandidate(main_dataset_name="GNSS EarthScope UNAVCO Earthquakes",
                          risky=["GNSS Earthquakes", "PBO GPS Data", "Plate Boundary Observatory GPS",
                                 "GNSS Time Series dataset"], flag_terms=GNSS_B2_FLAGS)
    res = run(cand, ["GNSS EarthScope UNAVCO Earthquakes"],
              known=["GNSS Time Series", "PBO GPS Data", "Plate Boundary Observatory GPS"])
    got = set(terms(res))
    for agency in ("NSF", "National Science Foundation", "UNAVCO", "IRIS Consortium",
                   "Incorporated Research Institutions for Seismology"):
        assert agency not in got


def test_generic_shared_phrases_do_not_make_a_flag_related():
    cand = QueryCandidate(main_dataset_name="NHIS", flag_terms=["National Institutes of Health", "Research Program"])
    res = run(cand, ["National Health Interview Survey"],
              known=["Health Research Program Survey", "Research Program Data"])
    assert "National Institutes of Health" not in terms(res)
    assert "Research Program" not in terms(res)


def test_fold_fragments_do_not_make_funder_flags_related():
    # "Alzheimer's" folds to "alzheimer s" and "U.S." to "u s": single letters are not name words.
    adni = QueryCandidate(main_dataset_name="ADNI", flag_terms=["Alzheimer's Association"])
    assert "Alzheimer's Association" not in terms(run(adni, ["ADNI"], known=["Alzheimer's Disease Neuroimaging"]))
    ppmi = QueryCandidate(main_dataset_name="PPMI",
                          flag_terms=["Michael J. Fox Foundation for Parkinson's Research"])
    assert not [c for c in run(ppmi, ["PPMI"], known=["Parkinson's Progression Markers Initiative"])[0]
                if c.source == "flag"]
    acs = QueryCandidate(main_dataset_name="American Community Survey", flag_terms=["U.S. Census Bureau"])
    res = run(acs, ["American Community Survey"], known=["U.S. Census American Community Survey"])
    assert "U.S. Census Bureau" not in terms(res)


def test_flag_equal_to_shared_phrase_is_not_promoted_by_alias_overlap():
    cand = QueryCandidate(main_dataset_name="PSID", flag_terms=["University of Michigan"])
    res = run(cand, ["PSID"], known=["University of Michigan PSID"])
    assert "University of Michigan" not in terms(res)


# --- gnss-n11 follow-up: risky alias heads, large-expansion evidence bar ---

N11_RISKY = ["PBO GNSS", "Plate Boundary Observatory GNSS", "UNAVCO geodetic data",
             "Plate Boundary Observatory", "Network of the Americas"]


def test_gnss_n11_risky_alias_head_does_not_make_org_flag_a_name():
    cand = QueryCandidate(main_dataset_name="GNSS EarthScope UNAVCO Earthquakes", risky=list(N11_RISKY),
                          flag_terms=["EarthScope Consortium", "UNAVCO", "IRIS Consortium"])
    res = run(cand, ["GNSS EarthScope UNAVCO Earthquakes"], known=N11_RISKY)
    got = terms(res)
    assert "UNAVCO" not in got
    assert "UNAVCO (not a name of the dataset)" in "; ".join(res[1])
    assert "Plate Boundary Observatory" in got and "Network of the Americas" in got


def test_dataset_acronyms_head_aliases():
    from dataset_agent.adapters.fulltext_recall import dataset_acronyms

    names = ["GNSS EarthScope UNAVCO Earthquakes"]
    aliases = ["UNAVCO geodetic data", "CMIP6"]
    vetted_none = dataset_acronyms(names, aliases, head_aliases=[])
    assert {"CMIP6", "GNSS"} <= vetted_none and "UNAVCO" not in vetted_none
    assert "UNAVCO" in dataset_acronyms(names, aliases, head_aliases=["UNAVCO geodetic data"])
    assert "UNAVCO" in dataset_acronyms(names, aliases)


def test_safe_alias_head_still_makes_flag_related():
    cand = QueryCandidate(main_dataset_name="Medical Imaging Repository", safe=["MIDRC imaging collection"],
                          flag_terms=["MIDRC"])
    res = run(cand, ["Medical Imaging Repository"])
    assert "MIDRC" in terms(res)


def test_wilson_lower():
    from dataset_agent.adapters.fulltext_recall import wilson_lower

    assert 0.66 < wilson_lower(23, 27) < 0.69
    assert wilson_lower(24, 27) > 0.70
    assert wilson_lower(0, 0) == 0.0


def test_large_expansion_needs_lower_bound():
    ok, reason, stats = judge_complement("Term One", ev(23, 27), 3866, S, base_count=880)
    assert not ok and "too uncertain" in reason and "lower bound" in reason
    assert stats["base_count"] == 880 and stats["precision_lower_bound"] is not None


def test_large_expansion_passes_with_strong_evidence():
    ok, reason, _ = judge_complement("Term One", ev(26, 27), 3866, S, base_count=880)
    assert ok and reason.startswith("recall: complement precision") and "use the data" in reason


def test_small_expansion_keeps_point_rule():
    assert judge_complement("Term One", ev(23, 27), 300, S, base_count=880)[0]
    assert judge_complement("Term One", ev(23, 27), 3866, S)[0]

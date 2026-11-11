from __future__ import annotations

from dataset_agent.adapters.literature import evaluate_terms_with_llm
from dataset_agent.adapters.query_builder import build_for_clause
from dataset_agent.adapters.term_sanitizer import sanitize_term, sanitize_terms

NBSP_JUNK = "MIDRC" + "\xa0​\xa0\xa0" * 4 + "..."


def test_nbsp_zero_width_and_ellipsis_junk_is_dropped() -> None:
    assert sanitize_term(NBSP_JUNK) is None
    assert sanitize_term("MIDRC\xa0​\xa0\xa0\xa0…") is None


def test_nbsp_runs_collapse_to_single_space() -> None:
    assert sanitize_term("MIDRC\xa0\xa0Data​\xa0Commons") == "MIDRC Data Commons"


def test_unicode_dashes_become_ascii_hyphen() -> None:
    assert sanitize_term("ARPA‑H") == "ARPA-H"
    for ch in "‐‒–—―−﹘﹣－":
        assert sanitize_term(f"A{ch}B") == "A-B"


def test_curly_quotes_and_surrounding_punctuation() -> None:
    assert sanitize_term("“MIDRC Commons”") == "MIDRC Commons"
    assert sanitize_term("O’Brien Study") == "O'Brien Study"
    assert sanitize_term("  'Foo Bar',  ") == "Foo Bar"


def test_ampersand_kept_and_no_letters_dropped() -> None:
    name = "Medical Imaging & Data Resource Center Data Commons"
    assert sanitize_term(name) == name
    assert sanitize_term("2020-2021") is None
    assert sanitize_term("") is None
    assert sanitize_term(None) is None


def test_sanitize_terms_keeps_order_and_drops_bad() -> None:
    out = sanitize_terms(
        [NBSP_JUNK, "MIDRC Data Commons", "ARPA‑H", "x..."], "alias"
    )
    assert out == ["MIDRC Data Commons", "ARPA-H"]


def test_terms_evaluation_suggestions_are_sanitized() -> None:
    class Agent:
        def get_structured(self, prompt, tool):
            return {
                "is_effective": False,
                "dataset_names_score": 3,
                "flag_terms_score": 3,
                "issues": ["Rationale: x"],
                "suggested_dataset_names": [NBSP_JUNK, "MIDRC\xa0Commons"],
                "suggested_flag_terms": ["ARPA‑H"],
                "suggested_exclude_terms": ["foo...", "bar baz"],
                "reasoning": "r",
            }

    res = evaluate_terms_with_llm("MIDRC", "d", ["a"], ["b"], [], Agent())  # type: ignore[arg-type]
    assert res["suggested_dataset_names"] == ["MIDRC Commons"]
    assert res["suggested_flag_terms"] == ["ARPA-H"]
    assert res["suggested_exclude_terms"] == ["bar baz"]


def test_query_builder_never_emits_junk_terms() -> None:
    clause = build_for_clause(
        safe=["MIDRC Data Commons", NBSP_JUNK],
        risky=["MIDRC"],
        variant="V3",
        flag_terms=["ARPA‑H"],
    )
    assert "\xa0" not in clause and "​" not in clause and "‑" not in clause
    assert "..." not in clause
    assert "ARPA-H" in clause


def test_query_builder_escapes_ampersand_inside_phrase() -> None:
    clause = build_for_clause(
        safe=["Medical Imaging & Data Resource Center"], risky=[], flag_terms=[]
    )
    assert "Imaging \\& Data" in clause


def test_balanced_parentheses_and_quotes_are_preserved() -> None:
    assert (
        sanitize_term("National Health and Nutrition Examination Survey (NHANES)")
        == "National Health and Nutrition Examination Survey (NHANES)"
    )
    assert sanitize_term("Add Health (Wave I)") == "Add Health (Wave I)"
    assert sanitize_term("NIH ‘All of Us’") == "NIH 'All of Us'"
    assert sanitize_term("Yelp!") == "Yelp!"


def test_wrapping_and_unbalanced_edges_are_removed() -> None:
    assert sanitize_term('"MIDRC Data Commons".') == "MIDRC Data Commons"
    assert sanitize_term("(MIDRC)") == "MIDRC"
    assert sanitize_term("MIDRC)") == "MIDRC"
    assert sanitize_term("'MIDRC") == "MIDRC"


def test_acronym_survives_for_validation_terms() -> None:
    from dataset_agent.adapters.literature import build_dataset_validation_terms
    from dataset_agent.domain.models import DatasetRecord, Group, YearsRange

    name = sanitize_term("National Health and Nutrition Examination Survey (NHANES)")
    rec = DatasetRecord(
        engine="dimensions",
        group=Group(name="default"),
        main_dataset_name=name,
        description="x",
        dataset_names=[name],
        years_range=YearsRange(start_year=2015, end_year=2025),
    )
    assert "NHANES" in build_dataset_validation_terms(rec)


def test_curly_inner_quotes_are_removed_and_seed_accepts_term() -> None:
    from dataset_agent.adapters.seed_mapping import QueryCandidate, candidate_to_seed

    cleaned = sanitize_term("“All of Us” Research Program")
    assert cleaned == "All of Us Research Program"
    seed = candidate_to_seed(QueryCandidate(cleaned, safe=[cleaned]))
    assert cleaned in seed["standalone_names"]


def test_trailing_question_mark_is_stripped_and_logged(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="dataset_agent.adapters.term_sanitizer"):
        assert sanitize_term("MIDAS?") == "MIDAS"
    assert any("uncertainty marker" in r.message for r in caplog.records)
    assert sanitize_term("MIDAS (?)") == "MIDAS"
    assert sanitize_term("MIDAS [?].") == "MIDAS"
    assert sanitize_term("MIDAS survey?") == "MIDAS survey"
    assert sanitize_term("MIDRC Data Commons") == "MIDRC Data Commons"
    assert sanitize_terms(["MIDAS?", "MIDRC"]) == ["MIDAS", "MIDRC"]


def test_question_mark_only_terms_are_dropped() -> None:
    assert sanitize_term("?") is None
    assert sanitize_term("(?)") is None
    assert sanitize_term("???") is None
    assert sanitize_terms(["?", "MIDRC"]) == ["MIDRC"]

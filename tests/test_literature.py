"""Unit tests for literature substring prefilter (iteration v2)."""

from __future__ import annotations

from dataset_agent.adapters.literature import (
    _build_ordered_search_terms,
    _build_string_search_matched_publications,
    _find_first_substring_match,
    _match_context_window,
    _validate_publications_lexical,
)


def test_ordered_terms_main_then_aliases_then_flags() -> None:
    ordered = _build_ordered_search_terms(
        "Main Dataset",
        ["Alias One", "Main Dataset"],
        ["Org A"],
    )
    terms = [t for t, _ in ordered]
    assert terms[0] == "Main Dataset"
    assert "Alias One" in terms
    assert "Org A" in terms
    assert terms.index("Alias One") < terms.index("Org A")


def test_find_first_respects_order_dataset_before_flag() -> None:
    text = "We use Org A and the Main Dataset here."
    ordered = _build_ordered_search_terms(
        "Main Dataset",
        [],
        ["Org A"],
    )
    hit = _find_first_substring_match(text, ordered, min_term_len=3)
    assert hit is not None
    term, src, _, _ = hit
    assert term == "Main Dataset"
    assert src == "dataset"


def test_find_first_flag_when_only_flag_matches() -> None:
    text = "Sponsored by Org A only."
    ordered = _build_ordered_search_terms("Zebra", [], ["Org A"])
    hit = _find_first_substring_match(text, ordered, min_term_len=3)
    assert hit is not None
    assert hit[0] == "Org A"
    assert hit[1] == "flag"


def test_context_window_clamps() -> None:
    text = "0123456789" * 20
    mid = 50
    w = _match_context_window(text, mid, mid + 5, before=10, after=10)
    assert len(w) <= 25
    assert "01234" in w


def test_validate_publications_lexical_sets_match_context() -> None:
    pubs = [
        {
            "title": "Study",
            "abstract": "We analyzed the Main Dataset extensively.",
        }
    ]
    ordered = _build_ordered_search_terms("Main Dataset", [], [])
    out = _validate_publications_lexical(pubs, ordered)
    assert out["valid"] == 1
    d0 = out["details"][0]
    assert d0["valid"] is True
    assert d0["matched_term"] == "Main Dataset"
    assert "Main Dataset" in d0["match_context"]


def test_build_string_search_matched_publications() -> None:
    pubs = [
        {"title": "A", "abstract": "no hit", "id": "pub.1"},
        {"title": "B", "abstract": "Uses Main Dataset.", "id": "pub.2"},
    ]
    ordered = _build_ordered_search_terms("Main Dataset", [], [])
    lex = _validate_publications_lexical(pubs, ordered)
    valid_sorted = sorted(
        [d for d in lex["details"] if d["valid"]],
        key=lambda d: d["index"],
    )
    matched = _build_string_search_matched_publications(pubs, valid_sorted)
    assert len(matched) == 1
    assert matched[0]["publication_id"] == "pub.2"
    assert "Main Dataset" in matched[0]["match_snippet"]

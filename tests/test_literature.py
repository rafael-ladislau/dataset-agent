"""Unit tests for literature substring prefilter (iteration v2)."""

from __future__ import annotations

from dataset_agent.adapters.literature import (
    _build_ordered_search_terms,
    _build_string_search_matched_publications,
    _drop_suggestions_already_in_query,
    _filter_publications_by_exclude_terms,
    _find_first_substring_match,
    _match_context_window,
    _pub_index_by_publication_id,
    _terms_evaluation_prompt,
    _validate_publications_lexical,
)


def test_ordered_terms_skips_short_split_parts() -> None:
    ordered = _build_ordered_search_terms(
        "Occupational Requirements Survey - ORS",
        [],
        [],
    )
    terms = [t for t, _ in ordered]
    assert "ORS" not in terms
    assert "Occupational Requirements Survey - ORS" in terms
    assert "Occupational Requirements Survey" in terms


def test_ors_does_not_match_inside_authors() -> None:
    text = "The authors conclude that factors matter."
    ordered = [("ORS", "dataset")]
    assert _find_first_substring_match(text, ordered, min_term_len=3) is None


def test_ors_matches_whole_token() -> None:
    ordered = [("ORS", "dataset")]
    hit = _find_first_substring_match("We use ORS from BLS.", ordered, min_term_len=3)
    assert hit is not None
    assert hit[0] == "ORS"
    hit2 = _find_first_substring_match("Survey (ORS) methodology.", ordered, min_term_len=3)
    assert hit2 is not None
    assert hit2[0] == "ORS"


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
            "id": "dims.pub.1",
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
    assert d0["publication_id"] == "dims.pub.1"
    assert "index" not in d0
    assert "Main Dataset" in d0["match_context"]


def test_terms_evaluation_prompt_includes_metrics_and_recipe_rules() -> None:
    details = [
        {
            "final_score": 10,
            "mention_score": 10,
            "context_score": 10,
            "mentioned_term": "O*NET",
            "title": "Paper One",
        },
        {
            "final_score": 0,
            "mention_score": 0,
            "context_score": 5,
            "mentioned_term": "",
            "title": "Paper Two",
        },
    ]
    p = _terms_evaluation_prompt(
        "My Dataset",
        "A short description.",
        ["Alias A"],
        ["Org X"],
        details,
    )
    assert "Share with final_score>=5: 1/2" in p
    assert "avg mention_score=" in p and "avg context_score=" in p
    assert "string matching" in p
    assert "workforce skills database" in p  # negative example anchored in prompt
    assert "empty arrays" in p


def test_drop_suggestions_removes_terms_already_in_query() -> None:
    names, flags = _drop_suggestions_already_in_query(
        ["Foo Survey", "Foo Survey", "Brand New Alias"],
        ["USDA", "New Org"],
        main_dataset_name="Foo Survey",
        dataset_names=["Alias One"],
        flag_terms=["usda"],
    )
    assert names == ["Brand New Alias"]
    assert flags == ["New Org"]


def test_exclude_terms_post_filter_drops_matching_pub() -> None:
    pubs = [
        {"title": "Stem cell review", "abstract": "Methods.", "id": "p1"},
        {"title": "Farm survey", "abstract": "Uses USDA data.", "id": "p2"},
    ]
    out = _filter_publications_by_exclude_terms(pubs, ["stem cell", "xx"])
    assert len(out) == 1
    assert out[0]["id"] == "p2"


def test_build_string_search_matched_publications() -> None:
    pubs = [
        {"title": "A", "abstract": "no hit", "id": "pub.1"},
        {"title": "B", "abstract": "Uses Main Dataset.", "id": "pub.2"},
    ]
    ordered = _build_ordered_search_terms("Main Dataset", [], [])
    lex = _validate_publications_lexical(pubs, ordered)
    order = _pub_index_by_publication_id(pubs)
    valid_sorted = sorted(
        [d for d in lex["details"] if d["valid"]],
        key=lambda d: order.get(d["publication_id"], 999),
    )
    matched = _build_string_search_matched_publications(valid_sorted)
    assert len(matched) == 1
    assert matched[0]["publication_id"] == "pub.2"
    assert "Main Dataset" in matched[0]["match_snippet"]

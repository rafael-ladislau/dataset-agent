"""An inner double quote becomes a space on every path that builds a Dimensions phrase, so
the clause that is released searches the same phrase that was counted."""

from __future__ import annotations

from dataset_agent.adapters.dimensions_dsl import build_publications_count_dsl, inner_quoted_term
from dataset_agent.adapters.query_builder import build_for_clause
from dataset_agent.adapters.resource_map import cooccurrence_clause
from dataset_agent.adapters.term_sanitizer import sanitize_term


def test_sanitize_term_replaces_an_inner_quote_with_a_space():
    assert sanitize_term('ERA5"Land"') == "ERA5 Land"
    assert sanitize_term('The"All of Us"Research Program') == "The All of Us Research Program"
    assert sanitize_term('The "All of Us" Research Program') == "The All of Us Research Program"


def test_the_built_clause_and_the_alias_count_search_one_phrase():
    term = 'ERA5"Land"'
    clause = build_for_clause(safe=[term])
    assert clause == '(\\"ERA5 Land\\")'
    assert '\\"ERA5 Land\\"' in build_publications_count_dsl(term)
    assert inner_quoted_term('The "All of Us" Research Program') == (
        "The All of Us Research Program")


def test_the_cooccurrence_clause_escapes_like_every_other_phrase():
    clause = cooccurrence_clause(["All of Us"], "Medical Imaging & Data Resource Center")
    assert '\\"Medical Imaging \\& Data Resource Center\\"' in clause
    assert '\\"All of Us\\"' in clause

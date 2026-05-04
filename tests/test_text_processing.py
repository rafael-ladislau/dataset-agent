"""Tests for text_processing helpers."""

from __future__ import annotations

from dataset_agent.adapters.text_processing import (
    clean_description,
    dedupe_strings_ci_preserve_order,
    hostname_from_http_url,
    is_whole_alias_generic_token,
    remove_compound_aliases,
)


def test_clean_description_strips_search_preamble() -> None:
    raw = (
        "Based on the web search results, here is a concise description of the ORS dataset: "
        "The Occupational Requirements Survey is published by the BLS."
    )
    out = clean_description(raw)
    assert not out.lower().startswith("based on")
    assert "Occupational Requirements Survey" in out
    assert "BLS" in out


def test_clean_description_strips_here_is() -> None:
    raw = "Here's a concise description: The dataset contains job requirements."
    out = clean_description(raw)
    assert out.startswith("The dataset")


def test_remove_compound_aliases_drops_long_form() -> None:
    aliases = [
        "Occupational Information Network - O*NET",
        "O*NET",
        "Occupational Information Network",
    ]
    out = remove_compound_aliases(aliases)
    assert "Occupational Information Network - O*NET" not in out
    assert "O*NET" in out


def test_dedupe_strings_ci_preserve_order() -> None:
    assert dedupe_strings_ci_preserve_order(["A", "a", "B"]) == ["A", "B"]


def test_hostname_from_http_url() -> None:
    assert hostname_from_http_url("https://WWW.Census.Gov/foo") == "www.census.gov"
    assert hostname_from_http_url(None) is None


def test_is_whole_alias_generic_token() -> None:
    assert is_whole_alias_generic_token("data") is True
    assert is_whole_alias_generic_token("ACS data") is False

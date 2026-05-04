"""Tests for research pipeline alias filtering."""

from __future__ import annotations

from dataset_agent.application.research import _filter_alias_entries


def test_filter_drops_whole_generic_token() -> None:
    out = _filter_alias_entries(["survey", "CPS"], [])
    assert any(x.lower() == "cps" for x in out)
    assert "survey" not in out


def test_filter_drops_overlong_non_url() -> None:
    long = "x" * 90
    out = _filter_alias_entries([long, "Short"], [])
    assert "Short" in out
    assert long not in out


def test_filter_keeps_urls_even_if_long() -> None:
    u = "https://example.com/" + "p" * 100
    out = _filter_alias_entries([u, "Z"], [])
    assert u in out

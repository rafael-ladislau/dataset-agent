"""Tests for heuristic list/URL extraction."""

from __future__ import annotations

from dataset_agent.adapters.extractor import HeuristicTextExtractor


def test_extract_list_python_literal() -> None:
    ex = HeuristicTextExtractor()
    assert ex.extract_list('["a", "b"]') == ["a", "b"]


def test_extract_list_embedded_brackets() -> None:
    ex = HeuristicTextExtractor()
    text = 'Here: ["x", "y"]'
    assert ex.extract_list(text) == ["x", "y"]


def test_extract_list_bullets() -> None:
    ex = HeuristicTextExtractor()
    text = "- one\n* two\n- three"
    assert ex.extract_list(text) == ["one", "two", "three"]


def test_extract_url() -> None:
    ex = HeuristicTextExtractor()
    assert ex.extract_url('See https://example.com/path).') == "https://example.com/path"


def test_extract_section_single() -> None:
    ex = HeuristicTextExtractor()
    text = """===DESCRIPTION===
This is the description.
===HOME_URL===
https://example.com
"""
    assert ex.extract_section(text, "DESCRIPTION") == "This is the description."
    assert ex.extract_section(text, "HOME_URL") == "https://example.com"


def test_extract_section_none_value() -> None:
    ex = HeuristicTextExtractor()
    text = "===DATA_URL===\nNone\n===SCHEMA_URL===\nhttps://schema.example.com"
    assert ex.extract_section(text, "DATA_URL") == ""
    assert ex.extract_section(text, "SCHEMA_URL") == "https://schema.example.com"


def test_extract_sections_all() -> None:
    ex = HeuristicTextExtractor()
    text = """===DATA_URL===
https://data.example.com
===SCHEMA_URL===
None
===ACCESS_TYPE===
Open
"""
    sections = ex.extract_sections(text)
    assert sections["DATA_URL"] == "https://data.example.com"
    assert sections["SCHEMA_URL"] == ""
    assert sections["ACCESS_TYPE"] == "Open"


def test_extract_section_missing() -> None:
    ex = HeuristicTextExtractor()
    text = "===DESCRIPTION===\nSome text"
    assert ex.extract_section(text, "MISSING") == ""

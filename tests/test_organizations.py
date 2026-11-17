"""Tests for organization / flag_terms cleaning."""

from __future__ import annotations

from dataset_agent.adapters.organizations import process_organizations


def test_process_organizations_drops_tool_json_and_meta() -> None:
    raw = [
        "BLS",
        '{"name": "web_search", "parameters": {"query": "Occupational Requirements Survey"}}',
        'To answer the prompt, we need to call the `web_search` function',
        "United States Department of Agriculture",
    ]
    out = process_organizations(raw)
    assert "BLS" in out
    assert "United States Department of Agriculture" in out
    assert not any("web_search" in x for x in out)
    assert not any("To answer the prompt" in x for x in out)


def test_process_organizations_drops_prompt_echo() -> None:
    raw = [
        '"Occupational Requirements Survey - ORS dataset organization creator publisher funder".',
        "National Institutes of Health",
    ]
    out = process_organizations(raw)
    assert "NIH" in out or "National Institutes of Health" in out
    assert not any("organization creator publisher funder" in x for x in out)


def test_process_organizations_handles_numbered_lists() -> None:
    """LLM sometimes returns numbered lists like '1. "Name" and "ACRONYM"'."""
    raw = [
        '1. "Bureau of Labor Statistics" and "BLS"',
        '2. "US Bureau of Labor Statistics" and "United States Bureau of Labor Statistics"',
        '3. "Social Security Administration" and "SSA"',
    ]
    out = process_organizations(raw)
    assert "Bureau of Labor Statistics" in out
    assert "BLS" in out
    assert "Social Security Administration" in out
    assert "SSA" in out
    # Should NOT contain the numbered format
    assert not any(x.startswith("1.") or x.startswith("2.") for x in out)


def test_process_organizations_drops_meta_commentary() -> None:
    """LLM meta-commentary should be filtered out."""
    raw = [
        "Based on the web search results, here is the list of organizations",
        "Bureau of Labor Statistics",
        "Note: The list only includes creators, publishers, funders, hosting institutions as per the original question.",
    ]
    out = process_organizations(raw)
    assert "Bureau of Labor Statistics" in out
    assert not any("Based on" in x for x in out)
    assert not any("Note:" in x for x in out)


def test_process_organizations_no_irrelevant_acronyms() -> None:
    """Should NOT add acronyms that weren't in the input."""
    raw = [
        "Bureau of Labor Statistics",
        "BLS",
    ]
    out = process_organizations(raw)
    assert "BLS" in out
    assert "Bureau of Labor Statistics" in out
    # Should NOT have unrelated acronyms from the map
    assert "NASA" not in out
    assert "CDC" not in out
    assert "USDA" not in out
    assert "NIH" not in out

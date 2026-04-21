"""Tests for LLM artifact filtering in list fields."""

from __future__ import annotations

from dataset_agent.adapters.llm_output_cleanup import is_llm_list_entry_junk


def test_junk_tool_json_and_prose_aliases() -> None:
    assert is_llm_list_entry_junk(
        'To Answer This Question, I Will Use the `web_search` Function To Search '
        "for Information About How the Dataset Is Cited and Referenced."
    )
    assert is_llm_list_entry_junk(
        '{"name": "web_search", "parameters": {"query": "occupational Requirements Survey - Ors Citation"}}'
    )
    assert not is_llm_list_entry_junk("Occupational Requirements Survey")
    assert not is_llm_list_entry_junk("BLS ORS")


def test_junk_json_fragments() -> None:
    """JSON fragments from tool calls should be filtered."""
    assert is_llm_list_entry_junk('"parameters": {')
    assert is_llm_list_entry_junk('"query": "occupational Requirements Survey - Ors How To Cite Dataset"}}')
    assert is_llm_list_entry_junk('{"name":')
    assert is_llm_list_entry_junk('}}')


def test_junk_meta_commentary() -> None:
    """LLM meta-commentary should be filtered."""
    assert is_llm_list_entry_junk("Based on the web search results, here is the list of organizations")
    assert is_llm_list_entry_junk("Here is the list of aliases I found:")
    assert is_llm_list_entry_junk("Note: The list only includes creators, publishers")
    assert is_llm_list_entry_junk("According to the search results, the following...")


def test_junk_numbered_list_with_quotes() -> None:
    """Numbered list items with embedded quotes are parsing artifacts."""
    assert is_llm_list_entry_junk('1. "Bureau of Labor Statistics"')
    assert is_llm_list_entry_junk('2. "Some Organization"')


def test_valid_entries_not_filtered() -> None:
    """Valid organization/alias names should NOT be filtered."""
    assert not is_llm_list_entry_junk("Bureau of Labor Statistics")
    assert not is_llm_list_entry_junk("BLS")
    assert not is_llm_list_entry_junk("U.S. Department of Labor")
    assert not is_llm_list_entry_junk("https://doi.org/10.1234/example")
    assert not is_llm_list_entry_junk("Occupational Requirements Survey (ORS)")


def test_junk_citation_artifacts() -> None:
    """Citation format artifacts from LLM should be filtered."""
    # Malformed list strings with citation formats
    assert is_llm_list_entry_junk(
        '["bls", "occupational Requirements Survey", Apa: Bureau of Labor Statistics...]'
    )
    # Citation notes
    assert is_llm_list_entry_junk(
        "Also Note That Citation Styles May Vary Depending On the Field of Study"
    )
    assert is_llm_list_entry_junk(
        "Note That the Dois Provided Are for the Bls Website"
    )
    # Citation format markers
    assert is_llm_list_entry_junk(
        "Apa: Bureau of Labor Statistics (bls). (). Occupational Requirements Survey."
    )
    assert is_llm_list_entry_junk(
        "Mla: Bureau of Labor Statistics. The Occupational Requirements Survey."
    )

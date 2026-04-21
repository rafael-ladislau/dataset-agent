"""Tests for text_processing helpers."""

from __future__ import annotations

from dataset_agent.adapters.text_processing import clean_description


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

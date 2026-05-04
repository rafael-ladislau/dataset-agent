"""Prompt wiring for alias discovery."""

from __future__ import annotations

from dataset_agent.application import prompts


def test_aliases_prompt_adds_site_search_when_dataset_url() -> None:
    body = prompts.aliases_prompt(
        "Current Population Survey",
        "Monthly household survey.",
        "https://www.census.gov/cps",
        ["Census Bureau"],
        dataset_url="https://www.census.gov/programs-surveys/cps.html",
    )
    assert "site:www.census.gov" in body
    assert "Reference URL" in body

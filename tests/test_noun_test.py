"""Round 2, item 5: one noun test for the map prompt, the scope question and alias routing."""

from __future__ import annotations

from dataset_agent.adapters.resource_map import resource_map_prompt
from dataset_agent.adapters.scope_check import (
    DATA_LIST_NOUNS,
    PROGRAM_LIST_NOUNS,
    carries_data_noun,
    generated_data_phrases,
    scope_prompt,
)
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import ResourceMap


def test_map_prompt_and_scope_question_share_the_noun_lists():
    m = resource_map_prompt(dataset_name="E", description="", names=[], pages="")
    s = scope_prompt(dataset_name="E", description="", program_names=[], data_names=[],
                     title="", passages=[], quote="")
    for nouns in (DATA_LIST_NOUNS, PROGRAM_LIST_NOUNS):
        assert nouns in m and nouns in s


def test_head_noun_decides():
    assert carries_data_noun("Example Cohort") and carries_data_noun("EX cohort")
    assert not carries_data_noun("Example Cohort Program")
    assert generated_data_phrases(["Example Cohort"]) == []


def test_a_data_noun_alias_is_not_gated():
    rmap = ResourceMap(program_names=["Example Program", "Example Cohort"],
                       data_resource_names=["Example Workbench"])
    notes: list[str] = []
    safe, risky, _ = QueryOptimizationUseCase._apply_resource_map(
        rmap, ["Example Cohort", "Example Program"], [], [], notes)
    assert "Example Cohort" in safe and "Example Cohort" not in risky
    assert "Example Program" in risky and "Example Program" not in safe
    assert any("data-noun names not gated ['Example Cohort']" in n for n in notes)

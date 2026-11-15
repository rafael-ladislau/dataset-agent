"""The Dimensions search-semantics lessons doc: present, self-contained, cited by the code
that follows it, and not contradicted by the older spec."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "dimensions-search-semantics.md"
DOC_REF = "docs/dimensions-search-semantics.md"


def _doc() -> str:
    assert DOC.is_file(), f"{DOC_REF} is missing"
    return DOC.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "topic",
    [
        "hyphen", "slash", "asterisk", "U+2011", "soft hyphen", "joined", "letter and a digit",
        "two digits", "two letters", '"+"', '"&"', "case", '":"', '","', '";"', "apostrophe",
        "parentheses", "inner double quote", '"_"', "plural", "wildcard", "precedence",
        "lowercase", "NOT binds loosest",
    ],
)
def test_doc_covers_each_measured_rule(topic: str) -> None:
    assert topic.casefold() in _doc().casefold()


def test_doc_states_dates_sources_and_the_mirrored_table() -> None:
    doc = _doc()
    assert "2026-11-15" in doc
    assert "MEASURED_PHRASE_RULES" in doc
    assert "65" in doc  # the identity check over the emitted query shapes
    assert "Not measured" in doc
    # The parenthesisation the builder emits.
    assert "((S) OR ((R) AND (F))) NOT (E)" in doc


def test_doc_is_self_contained() -> None:
    doc = _doc()
    for private in ("/Volumes", "/private/", "/Users/", "scratchpad", "utah_project", "Trello"):
        assert private not in doc


@pytest.mark.parametrize(
    "module",
    ["src/dataset_agent/adapters/query_builder.py",
     "src/dataset_agent/adapters/dimensions_phrase.py"],
)
def test_code_cites_the_doc(module: str) -> None:
    assert DOC_REF in (ROOT / module).read_text(encoding="utf-8")


def _escape_sets(text: str) -> list[set[str]]:
    """Every escape-set listing in *text* (the spec's code block and table cells)."""
    out = []
    block = re.search(r"backslash-escaped inside for-clauses:\s*```\s*\n(.*?)\n```", text, re.S)
    if block:
        out.append(set(block.group(1).replace(" ", "")))
    for m in re.finditer(r"Escape DSL special characters: `([^`]*)`", text):
        out.append(set(m.group(1).replace(" ", "").replace("\\|", "|")))
    return out


@pytest.mark.parametrize("spec", ["dataset_query_agent_spec.md", "docs/spec-gap-analysis.md"])
def test_spec_escape_set_matches_the_code(spec: str) -> None:
    from dataset_agent.adapters.dimensions_dsl import _DSL_INNER_SPECIAL

    sets = _escape_sets((ROOT / spec).read_text(encoding="utf-8"))
    assert sets, f"no escape set found in {spec}"
    for found in sets:
        assert found == set(_DSL_INNER_SPECIAL) | {"\\"}


def test_spec_never_shows_an_unparenthesised_mixed_clause() -> None:
    spec = (ROOT / "dataset_query_agent_spec.md").read_text(encoding="utf-8")
    assert "OR (tier terms AND qualifier) NOT exclusions" not in spec

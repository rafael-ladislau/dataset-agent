"""The measured Dimensions same-search rules and the helpers that rely on them."""

from __future__ import annotations

import pytest

from dataset_agent.adapters.ai_model_research import HubEvidence, assemble_candidates
from dataset_agent.adapters.dimensions_phrase import same_search, search_key
from dataset_agent.adapters.fulltext_recall import recall_candidates
from dataset_agent.adapters.fulltext_refiner import _covered
from dataset_agent.adapters.query_builder import drop_flags_equal_to_aliases
from dataset_agent.adapters.seed_mapping import QueryCandidate


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("GPT-4", "GPT 4"),
        ("CPS/FSS", "CPS FSS"),
        ("O*NET", "O NET"),
        ("GPT‑4", "GPT-4"),  # non-breaking hyphen
        ("GPT–4", "GPT 4"),  # en dash
        ("C++", "C"),
        ("PhaseNet+", "PhaseNet"),
        ("Medical Imaging & Data Resource Center", "Medical Imaging Data Resource Center"),
        ("All of Us Research Program", "All of Us Research Programs"),
        ("all of us research program", "ALL OF US RESEARCH PROGRAM"),
        ("Prithvi-EO-2.0", "Prithvi EO 2.0"),
    ],
)
def test_one_search(a: str, b: str) -> None:
    assert same_search(a, b)
    assert search_key(a) == search_key(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("GPT4", "GPT 4"),
        ("ONET", "O NET"),
        ("PanguWeather", "Pangu Weather"),
        ("scikit.learn", "scikit-learn"),
        ("GPT-4.5", "GPT-4 5"),
        ("Prithvi-EO-2.0", "Prithvi-EO-2-0"),
        ("v2.0", "v2 0"),
        # "&" is dropped, it is not the word "and" (122 vs 170 papers).
        ("Medical Imaging & Data Resource Center", "Medical Imaging and Data Resource Center"),
        # Plural folding was measured for ordinary words only.
        ("CPS", "CP"),
        ("BioCLIPs", "BioCLIP"),
        ("MegaDetectorV5s", "MegaDetectorV5"),
        ("Census", "Censu"),
    ],
)
def test_two_searches(a: str, b: str) -> None:
    assert not same_search(a, b)


def test_flag_equal_to_a_risky_alias_under_dimensions_folding_is_dropped() -> None:
    assert drop_flags_equal_to_aliases(["GPT-4", "NVIDIA"], ["GPT 4"]) == ["NVIDIA"]
    assert drop_flags_equal_to_aliases(
        ["Medical Imaging & Data Resource Center", "NIBIB"],
        ["Medical Imaging Data Resource Center"],
    ) == ["NIBIB"]
    # The "and" spelling is another search: it gates, it does not satisfy itself.
    assert drop_flags_equal_to_aliases(
        ["Medical Imaging and Data Resource Center", "NIBIB"],
        ["Medical Imaging & Data Resource Center"],
    ) == ["Medical Imaging and Data Resource Center", "NIBIB"]
    assert drop_flags_equal_to_aliases(["scikit.learn", "NumPy"], ["scikit-learn"]) == [
        "scikit.learn", "NumPy",
    ]


def test_ai_model_name_expansion_keeps_dot_versions_apart() -> None:
    c = assemble_candidates(
        model_name="Prithvi-EO", family=None,
        given_aliases=["Prithvi-EO-2.0", "Prithvi-EO-2-0", "Prithvi EO 2.0", "PhaseNet+"],
        hub=HubEvidence(), llm={}, max_names=14, max_context=12, max_excludes=8,
    )
    names = [n for n, _k in c.names]
    assert "Prithvi-EO-2.0" in names and "Prithvi-EO-2-0" in names
    reasons = {t: r for k, t, r in c.rejected if k == "alias"}
    assert reasons["Prithvi EO 2.0"] == "same search as 'Prithvi-EO-2.0' for Dimensions"


def test_refiner_coverage_follows_the_measured_rules() -> None:
    assert _covered("GPT 4", ["GPT-4"])
    assert _covered("Medical Imaging Data Resource Center", ["Medical Imaging & Data Resource Center"])
    assert not _covered("Medical Imaging and Data Resource Center",
                        ["Medical Imaging & Data Resource Center"])
    assert not _covered("scikit.learn", ["scikit-learn"])


def test_recall_check_keeps_a_dot_spelling_as_another_search() -> None:
    current = QueryCandidate(main_dataset_name="Prithvi-EO-2.0", safe=["Prithvi-EO-2-0"])
    eligible, _rejected = recall_candidates(
        current, main_names=["Prithvi-EO-2.0", "Prithvi EO 2 0"], known_aliases=[],
        flag_terms=[],
    )
    assert [c.term for c in eligible] == ["Prithvi-EO-2.0"]

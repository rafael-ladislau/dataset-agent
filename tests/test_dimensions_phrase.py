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


@pytest.mark.parametrize(
    ("name", "others"),
    [
        ("Medical Imaging & Data Resource Center", ["Medical Imaging and Data Resource Center"]),
        ("Medical Imaging and Data Resource Center", ["Medical Imaging & Data Resource Center"]),
        ("Quarterly Census of Employment and Wages UI", ["Quarterly Census of Employment & Wages UI"]),
        ("R & D Survey", ["R and D Survey"]),
        # Lower-case words around "and" are description, not a proper name.
        ("data from the All of Us and the UK Biobank", []),
        ("AT&T", []),
        ("All of Us NIH", []),
    ],
)
def test_ampersand_spellings(name: str, others: list[str]) -> None:
    from dataset_agent.adapters.dimensions_phrase import ampersand_spellings

    assert ampersand_spellings(name) == others


# --------------------------------------------------------------------------- dots


@pytest.mark.parametrize(
    ("outer", "inner", "contained"),
    [
        ("Prithvi-WxC-1.0-2300M", "Prithvi WxC", True),
        ("Prithvi-EO-2.0", "Prithvi-EO", True),
        ("NVIDIA FourCastNet", "FourCastNet", True),
        # "." is part of the token: "2.0" and "4.5" are not "2" and "4".
        ("Prithvi-EO-2.0", "Prithvi-EO-2", False),
        ("GPT-4.5", "GPT-4", False),
        ("scikit.learn", "scikit", False),
        ("v2.0 release", "v2", False),
        # Joined forms are other tokens.
        ("FourCastNetv2", "FourCastNet", False),
        ("phase unwrapping network", "phase unwrapping", True),
    ],
)
def test_phrase_contains_treats_a_dot_as_part_of_the_token(
    outer: str, inner: str, contained: bool
) -> None:
    from dataset_agent.adapters.dimensions_phrase import phrase_contains

    assert phrase_contains(outer, inner) is contained


def test_a_dot_version_is_not_covered_by_its_shorter_name() -> None:
    from dataset_agent.application.query_optimization import covered_safe_names

    covered = covered_safe_names(
        ["GPT-4", "GPT-4.5", "Prithvi-EO-2", "Prithvi-EO-2.0", "Prithvi WxC",
         "Prithvi-WxC-1.0-2300M", "scikit-learn", "scikit.learn"]
    )
    assert covered == {"Prithvi-WxC-1.0-2300M": "Prithvi WxC"}

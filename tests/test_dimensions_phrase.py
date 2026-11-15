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
        # Plural folding of every token: all-caps, short, letter-and-digit, CamelCase
        # (matching is case-insensitive, so case cannot gate the stemmer).
        ("CPS", "CP"),
        ("NHANES", "NHANE"),
        ("LLMs", "LLM"),
        ("ERA5s", "ERA5"),
        ("maps", "map"),
        ("BioCLIPs", "BioCLIP"),
        ("MegaDetectorV5s", "MegaDetectorV5"),
        ("Parkinsons Progression Markers Initiative", "Parkinson Progression Markers Initiative"),
        ("BioCLIP: A Vision Foundation Model", "BioCLIP A Vision Foundation Model"),
        ("COVID-19:data", "COVID-19 data"),
        ("Aurora, Colorado", "Aurora Colorado"),
        ("Washington;DC", "Washington DC"),
        ("Alzheimer's disease", "Alzheimer disease"),
        ("Alzheimer\u2019s disease", "Alzheimers disease"),
        ("Medical Information Mart for Intensive Care (MIMIC)",
         "Medical Information Mart for Intensive Care MIMIC"),
        ('The "All of Us" Research Program', "The All of Us Research Program"),
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
        # A numeric token does not fold ("GPT-4s" 78 vs "GPT-4" 52,109).
        ("GPT-4s", "GPT-4"),
        ("GPT-4's", "GPT-4"),
        # Not measured, conservative: "ss" and "us" endings are not plurals.
        ("Census", "Censu"),
        ("GNSS", "GNS"),
        # ":" between two letters stays inside the token ("BioCLIP:A" 0 vs 32).
        ("BioCLIP:A Vision Foundation Model", "BioCLIP A Vision Foundation Model"),
        ("Alzheimer's disease", "Alzheimer s disease"),
        ("O'Brien", "O Brien"),
        ("bert_base_uncased", "bert base uncased"),
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


# --------------------------------------------------------------- mirrored table

# Measured Dimensions phrase rules: MIRRORED TABLE.
#
# The same rows, with the same expected answers, are kept in three repositories:
#   dataset-agent  tests/test_dimensions_phrase.py
#   insights       tests/unit/services/test_snippet_matching_table.py
#   extraction     tests/test_llm_results_term_contexts.py
# Change all three together. Each row reads: does the quoted Dimensions phrase
# ``term`` match a text whose words are ``text``? Counts are Dimensions full_data,
# 2020-2024, quoted phrases.
MEASURED_PHRASE_RULES: list[tuple[str, str, str, bool]] = [
    # Separators: hyphen, slash, asterisk and the Unicode dashes fold to a space.
    ("hyphen-is-space", "GPT-4", "GPT 4", True),
    ("slash-is-space", "CPS/FSS", "CPS FSS", True),
    ("asterisk-is-space", "O*NET", "O NET", True),
    ("nb-hyphen-is-hyphen", "GPT\u20114", "GPT-4", True),
    ("en-dash-is-space", "GPT\u20134", "GPT 4", True),
    # A joined form is another token ("GPT4" 12,146 vs "GPT 4" 52,109).
    ("joined-differs", "GPT4", "GPT 4", False),
    ("split-differs", "O NET", "ONET", False),
    # A dot between word characters is not a separator ("scikit.learn" 101).
    ("dot-not-separator", "scikit-learn", "scikit.learn", False),
    ("dot-version-is-one-token", "GPT-4", "GPT-4.5", False),
    # A dot is a separator between a letter and a digit ("NHANES.12" = "NHANES 12"
    # 97; "UK Biobank.23" = "UK Biobank 23" 142), which is how a GROBID citation
    # superscript reads ("NHANES.12 We"). It stays inside the token between two digits
    # ("GPT-4.5" 43 vs "GPT-4 5" 471; "ERA5.1" 113 vs "ERA5 1" 313) and between two
    # letters ("scikit.learn" above).
    ("dot-letter-digit", "NHANES 12", "NHANES.12", True),
    ("dot-letter-digit-in-term", "UK Biobank.23", "UK Biobank 23", True),
    ("dot-citation-superscript", "NHANES", "NHANES.12 We", True),
    ("dot-digits-one-token", "ERA5 1", "ERA5.1", False),
    ("dot-digits-version-split-differs", "GPT-4 5", "GPT-4.5", False),
    # Not measured, conservative: a dot between a digit and a letter stays inside.
    ("dot-digit-letter-not-separator", "ERA5 Land", "ERA5.Land", False),
    # "+" and "&" are dropped; "&" is not "and" (122 vs 170).
    ("plus-dropped", "PhaseNet", "PhaseNet+", True),
    ("ampersand-dropped", "Medical Imaging Data Resource Center",
     "Medical Imaging & Data Resource Center", True),
    ("ampersand-not-and", "Medical Imaging and Data Resource Center",
     "Medical Imaging & Data Resource Center", False),
    ("case-insensitive", "all of us", "ALL OF US", True),
    # ":" with whitespace on either side is a separator (BioCLIP title 32 = 32 = 32).
    ("colon-space-after", "BioCLIP A Vision Foundation Model",
     "BioCLIP: A Vision Foundation Model", True),
    ("colon-spaced", "BioCLIP A Vision Foundation Model",
     "BioCLIP : A Vision Foundation Model", True),
    ("colon-in-term", "BioCLIP: A Vision Foundation Model",
     "BioCLIP A Vision Foundation Model", True),
    # ":" between a digit and a letter is a separator ("COVID-19:data" 47,789).
    ("colon-digit-letter", "COVID-19 data", "COVID-19:data", True),
    ("colon-digit-letter-joined-differs", "COVID-19 data", "COVID-19data", False),
    # ":" between two letters is not a separator ("BioCLIP:A ..." 0 vs 32). Dropped
    # or kept cannot be told apart; conservative: it stays inside the token.
    ("colon-letters-not-separator", "BioCLIP A Vision", "BioCLIP:A Vision", False),
    ("colon-letters-one-token", "BioCLIP", "BioCLIP:A Vision", False),
    ("colon-letters-same-text", "BioCLIP:A Vision", "BioCLIP:A Vision", True),
    # "," and ";" are separators with or without a space (14,260 = 14,260; joined 10).
    ("comma-space", "Aurora Colorado", "Aurora, Colorado", True),
    ("comma-no-space", "Aurora Colorado", "Aurora,Colorado", True),
    ("comma-in-term", "Washington, DC", "Washington DC", True),
    ("comma-joined-differs", "Aurora Colorado", "AuroraColorado", False),
    ("semicolon-space", "Washington DC", "Washington; DC", True),
    ("semicolon-no-space", "Washington DC", "Washington;DC", True),
    # An apostrophe before a final "s" is dropped, then the plural folds:
    # "Alzheimer's" = "Alzheimer’s" = "Alzheimers" = "Alzheimer" (610,032), but
    # "Alzheimer s" is another phrase (9,904).
    ("apostrophe-s-straight", "Alzheimer disease", "Alzheimer's disease", True),
    ("apostrophe-s-curly", "Alzheimer disease", "Alzheimer\u2019s disease", True),
    ("apostrophe-s-in-term", "Alzheimer's disease", "Alzheimers disease", True),
    ("apostrophe-s-curly-term", "Alzheimer\u2019s disease", "Alzheimer's disease", True),
    ("apostrophe-s-split-differs", "Alzheimer's disease", "Alzheimer s disease", False),
    ("apostrophe-s-split-differs-singular", "Alzheimer disease", "Alzheimer s disease",
     False),
    ("apostrophe-s-ppmi", "Parkinson Progression Markers Initiative",
     "Parkinson's Progression Markers Initiative", True),
    # Any other apostrophe was not measured: conservative, it stays inside the token.
    ("apostrophe-other-not-separator", "O Brien", "O'Brien", False),
    # "4's" is "4s", and a numeric token does not fold (follows from the two rules).
    ("apostrophe-s-numeric-not-folded", "GPT-4", "GPT-4's", False),
    # Parentheses are separators (MIMIC 2,164 = 2,164, escaped or raw).
    ("parentheses-text", "Medical Information Mart for Intensive Care MIMIC",
     "Medical Information Mart for Intensive Care (MIMIC)", True),
    ("parentheses-term", "Medical Information Mart for Intensive Care (MIMIC)",
     "Medical Information Mart for Intensive Care MIMIC", True),
    # Inner double quotes break a Dimensions phrase (412,891 vs 3,876), so the query
    # never sends one; for matching they are a separator.
    ("double-quote-text", "All of Us Research Program",
     'The "All of Us" Research Program', True),
    ("double-quote-term", 'The "All of Us" Research Program',
     "The All of Us Research Program", True),
    # "_" is part of the token ("bert_base_uncased" 45 vs split 7,366 vs joined 11).
    ("underscore-not-split-text", "bert base uncased", "bert_base_uncased", False),
    ("underscore-not-split-term", "bert_base_uncased", "bert base uncased", False),
    ("underscore-not-joined", "bert_base_uncased", "bertbaseuncased", False),
    ("underscore-same-text", "ERA5_Land", "ERA5_Land", True),
    ("underscore-not-hyphen", "ERA5 Land", "ERA5_Land", False),
    # Plural folding, both ways, for ordinary words, all-caps tokens, short words and
    # letter-and-digit tokens; also inside a phrase (Parkinsons ... = Parkinson ...).
    ("plural-ordinary-word", "All of Us Research Program", "All of Us Research Programs",
     True),
    ("plural-inside-phrase", "Parkinsons Progression Markers Initiative",
     "Parkinson Progression Markers Initiative", True),
    ("plural-all-caps-singular", "NHANES", "NHANE", True),
    ("plural-all-caps-plural", "NHANE", "NHANES", True),
    ("plural-llm", "LLM", "LLMs", True),
    ("plural-llms", "LLMs", "LLM", True),
    ("plural-cnn", "CNN", "CNNs", True),
    ("plural-mri", "MRI", "MRIs", True),
    ("plural-letter-digit", "ERA5", "ERA5s", True),
    ("plural-letter-digit-singular", "ERA5s", "ERA5", True),
    ("plural-short-word", "map", "maps", True),
    ("plural-short-word-singular", "maps", "map", True),
    # A token that is all digits before the "s" does not fold ("GPT-4s" 78 vs 52,109).
    ("plural-numeric-not-folded", "GPT-4", "GPT-4s", False),
    ("plural-numeric-term-not-folded", "GPT-4s", "GPT-4", False),
    # Not measured, conservative: "ss" and "us" endings are not plurals ("is" folds:
    # MRIs = MRI above).
    ("not-plural-us-ending", "Census", "Censu", False),
    ("not-plural-ss-ending", "GNSS", "GNS", False),
]


@pytest.mark.parametrize(
    ("term", "text", "matches"),
    [pytest.param(t, x, m, id=i) for i, t, x, m in MEASURED_PHRASE_RULES],
)
def test_measured_phrase_rules(term: str, text: str, matches: bool) -> None:
    """The phrase *term* matches *text* when its tokens occur in it (containment)."""
    from dataset_agent.adapters.dimensions_phrase import phrase_contains

    assert phrase_contains(text, term) is matches


def test_ampersand_additions_dedupe_by_dimensions_search() -> None:
    from dataset_agent.adapters.dimensions_phrase import ampersand_additions

    amp = "Medical Imaging & Data Resource Center"
    assert ampersand_additions([amp]) == ["Medical Imaging and Data Resource Center"]
    assert ampersand_additions([amp, "Medical Imaging and Data Resource Centers"]) == []
    assert ampersand_additions([amp, "MEDICAL IMAGING AND DATA RESOURCE CENTER"]) == []


def test_ai_model_names_are_searched_in_both_ampersand_spellings() -> None:
    c = assemble_candidates(
        model_name="SatVision", family=None,
        given_aliases=["Earth Imaging & Vision Foundation Model"],
        hub=HubEvidence(), llm={}, max_names=14, max_context=12, max_excludes=8,
    )
    names = [n for n, _k in c.names]
    assert "Earth Imaging & Vision Foundation Model" in names
    assert "Earth Imaging and Vision Foundation Model" in names
    c = assemble_candidates(
        model_name="SatVision", family=None,
        given_aliases=["Earth Imaging & Vision Foundation Model",
                       "Earth Imaging and Vision Foundation Models"],
        hub=HubEvidence(), llm={}, max_names=14, max_context=12, max_excludes=8,
    )
    names = [n for n, _k in c.names]
    assert "Earth Imaging and Vision Foundation Model" not in names

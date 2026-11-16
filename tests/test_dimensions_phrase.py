"""The measured Dimensions same-search rules and the helpers that rely on them."""

from __future__ import annotations

import pytest

from dataset_agent.adapters.ai_model_research import HubEvidence, assemble_candidates
from dataset_agent.adapters.dimensions_phrase import plural_base, same_search, search_key
from dataset_agent.adapters.fulltext_recall import recall_candidates
from dataset_agent.adapters.fulltext_refiner import _covered
from dataset_agent.adapters.query_builder import drop_flags_equal_to_aliases, flag_gate_notes
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
        # Plural folding of every token of four or more characters: all-caps, short,
        # letter-and-digit, CamelCase (matching is case-insensitive, so case cannot gate
        # the stemmer).
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
        # "&" is a separator, it is not the word "and" (122 vs 170 papers).
        ("Medical Imaging & Data Resource Center", "Medical Imaging and Data Resource Center"),
        # A token under three characters does not fold ("GPT-4s" 78 vs "GPT-4" 52,109).
        ("GPT-4s", "GPT-4"),
        ("GPT-4's", "GPT-4"),
        # Not measured, conservative: a three-character token does not fold (every
        # measured fold has four or more characters: maps, LLMs, MRIs, ERA5s, 1990s).
        ("CPS", "CP"),
        ("GPS", "GP"),
        ("IPUMS CPS", "IPUMS CP"),
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


def test_plural_folding_needs_a_token_of_four_or_more_characters() -> None:
    """Measured folds all have four or more characters; three-character tokens are not
    measured, so they keep their final "s" (an extra search, never a lost one)."""
    assert plural_base("cps") == "cps" and plural_base("gps") == "gps"
    assert plural_base("maps") == "map" and plural_base("llms") == "llm"
    # So a flag "CP" is not taken as already inside a risky "IPUMS CPS".
    assert drop_flags_equal_to_aliases(["CP", "NIH"], ["IPUMS CPS"]) == ["CP", "NIH"]


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


def test_flag_contained_in_a_risky_alias_is_dropped() -> None:
    """(R) AND (F) satisfies itself when F's tokens occur inside R, under the measured
    folding (plurals included): every R paper would pass its own gate."""
    assert drop_flags_equal_to_aliases(
        ["Research Programs", "NIH"], ["All of Us Research Program"]) == ["NIH"]
    assert drop_flags_equal_to_aliases(
        ["UNAVCO", "GNSS"], ["UNAVCO geodetic data"]) == ["GNSS"]
    assert drop_flags_equal_to_aliases(
        ["cohort studies", "NIH"], ["Example Cohort Study"]) == ["NIH"]
    # Not contained: another token ("4.5" is one token), or a partial token.
    assert drop_flags_equal_to_aliases(["GPT-4", "OpenAI"], ["GPT-4.5"]) == ["GPT-4", "OpenAI"]
    assert drop_flags_equal_to_aliases(["Research", "NIH"], ["Researcher Workbench"]) == [
        "Research", "NIH"]
    # A flag containing the alias is a narrower gate, not a self-satisfying one.
    assert drop_flags_equal_to_aliases(["UK Biobank data", "NIH"], ["UK Biobank"]) == [
        "UK Biobank data", "NIH"]


def test_a_gate_every_flag_of_which_is_inside_the_alias_is_reported_as_ungated() -> None:
    """Every flag inside the alias: the flags are kept (they may still gate the other
    aliases), and the output says the alias is in effect searched without a gate."""
    assert drop_flags_equal_to_aliases(["UNAVCO"], ["UNAVCO geodetic data"]) == ["UNAVCO"]
    notes = flag_gate_notes(["UNAVCO"], ["UNAVCO geodetic data"])
    assert notes == [
        "flag gate: risky alias 'UNAVCO geodetic data' is in effect ungated: every flag "
        "('UNAVCO') occurs inside it, so its gate matches every 'UNAVCO geodetic data' paper"
    ]
    # The kept flag still gates "AoU"; only the alias that contains it is ungated.
    notes = flag_gate_notes(["Research Program"], ["All of Us Research Program", "AoU"])
    assert len(notes) == 1 and "'All of Us Research Program' is in effect ungated" in notes[0]
    assert "AoU" not in notes[0]


def test_a_flag_dropped_for_one_alias_is_reported_with_the_aliases_that_lose_it() -> None:
    """One flag group is shared by every risky alias (the extraction seed has a single
    gate), so a flag inside one alias leaves the gate of all of them: the output names
    the alias that contains it and the aliases that no longer match through it."""
    assert drop_flags_equal_to_aliases(
        ["Research Program", "NIH"], ["All of Us Research Program", "AoU"]) == ["NIH"]
    assert flag_gate_notes(["Research Program", "NIH"], ["All of Us Research Program", "AoU"]) == [
        "flag gate: 'Research Program' dropped from the shared gate because it occurs inside "
        "'All of Us Research Program'; 'AoU' no longer matches through it"
    ]
    # Inside every alias: nothing is lost for the others.
    assert flag_gate_notes(["UNAVCO", "GNSS"], ["UNAVCO geodetic data", "UNAVCO network"]) == [
        "flag gate: 'UNAVCO' dropped from the shared gate because it occurs inside "
        "'UNAVCO geodetic data', 'UNAVCO network'"
    ]
    assert flag_gate_notes(["NIH"], ["All of Us Research Program"]) == []
    assert flag_gate_notes([], ["AoU"]) == [] and flag_gate_notes(["NIH"], []) == []


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
    # A soft hyphen stays inside the token ("GPT\u00ad4" 11 vs "GPT-4" 52,109): it is
    # neither a separator nor removed.
    ("soft-hyphen-not-separator", "GPT-4", "GPT\u00ad4", False),
    ("soft-hyphen-not-removed", "PhaseNet", "Phase\u00adNet", False),
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
    # "+" and "&" are separators; "&" is not "and" (122 vs 170).
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
    # "4's" is "4s", and a token under three characters does not fold (follows from
    # the two rules).
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
    # A token under three characters does not fold ("GPT-4s" 78 vs 52,109: "4s").
    ("plural-numeric-not-folded", "GPT-4", "GPT-4s", False),
    ("plural-numeric-term-not-folded", "GPT-4s", "GPT-4", False),
    # Not measured, conservative: every measured fold has a plural token of four
    # or more characters (maps, LLMs, MRIs, CNNs, ERA5s, 1990s), so a three-character
    # token stays as written ("CPS" is not "CP").
    ("plural-three-char-not-folded", "CP", "CPS", False),
    ("plural-three-char-term-not-folded", "CPS", "CP", False),
    # Not measured, conservative: "ss" and "us" endings are not plurals ("is" folds:
    # MRIs = MRI above).
    ("not-plural-us-ending", "Census", "Censu", False),
    ("not-plural-ss-ending", "GNSS", "GNS", False),
    # A numeric token of four or more characters folds like any other ("1990s" =
    # "1990" 4,611,963; "2010s" = "2010" 12,627,656; "the 1990s" = "the 1990" 754,282).
    # "GPT-4s" above does not fold because its token "4s" is under three characters.
    ("plural-numeric-decade", "1990", "1990s", True),
    ("plural-numeric-decade-singular", "1990s", "1990", True),
    ("plural-numeric-2010s", "2010", "2010s", True),
    ("plural-numeric-in-phrase", "the 1990", "the 1990s", True),
    # "&" and "+" between two word characters are separators, not deleted: "AT&T" =
    # "AT T" 659,644 vs "ATT" 182,335 vs "AT and T" 1,094; "R&D" = "R D" 2,550,427 vs
    # "RD" 1,833,482; "Q&A" = "Q A" 460,362 vs "QA" 244,734; "Pfizer+BioNTech" =
    # "Pfizer BioNTech" 38,344 vs joined 5,247.
    ("ampersand-inside-separator", "AT T", "AT&T", True),
    ("ampersand-inside-not-joined", "ATT", "AT&T", False),
    ("ampersand-inside-not-and", "AT and T", "AT&T", False),
    ("ampersand-inside-rd", "R D", "R&D", True),
    ("ampersand-inside-rd-not-joined", "RD", "R&D", False),
    ("ampersand-inside-qa", "Q A", "Q&A", True),
    ("plus-inside-separator", "Pfizer BioNTech", "Pfizer+BioNTech", True),
    ("plus-inside-not-joined", "PfizerBioNTech", "Pfizer+BioNTech", False),
    # A dot at the end of a word (before whitespace or the end) is a separator:
    # "Pfizer Inc." = "Pfizer Inc" 31,650; "U.S. Census Bureau" = "U.S Census Bureau"
    # 58,506. Between two letters the dot stays inside the token: "US Census Bureau"
    # 46,326 and "U S Census Bureau" 2,247 are other searches.
    ("dot-word-end", "Pfizer Inc", "Pfizer Inc.", True),
    ("dot-word-end-in-term", "Pfizer Inc.", "Pfizer Inc", True),
    ("dot-word-end-before-space", "U.S Census Bureau", "U.S. Census Bureau", True),
    ("dot-word-end-in-term-before-space", "U.S. Census Bureau", "U.S Census Bureau", True),
    ("dot-abbreviation-not-joined", "US Census Bureau", "U.S. Census Bureau", False),
    ("dot-abbreviation-not-split", "U S Census Bureau", "U.S. Census Bureau", False),
    # "-ies" folds to "-y" ("cancer registries" = "cancer registry" 83,433, while
    # "cancer registrie" is 0; "cohort studies" = "cohort study" 1,664,062). "-es" does
    # not fold to the bare stem ("bounding boxes" 116,770 vs "box" 157,583; "meta
    # analyses" 3,185,482 vs "meta analysis" 3,184,312).
    ("plural-ies", "cancer registry", "cancer registries", True),
    ("plural-ies-singular", "cancer registries", "cancer registry", True),
    ("plural-ies-not-e-stem", "cancer registrie", "cancer registries", False),
    ("plural-ies-study", "cohort study", "cohort studies", True),
    ("plural-es-not-folded", "bounding box", "bounding boxes", False),
    ("plural-es-analyses-not-folded", "meta analysis", "meta analyses", False),
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


def _phrase_matchers():
    from dataset_agent.adapters.dimensions_phrase import ampersand_spellings, phrase_contains
    from dataset_agent.adapters.fulltext_evidence import quote_allows_references_only
    from dataset_agent.adapters.query_builder import build_for_clause
    from dataset_agent.adapters.term_sanitizer import sanitize_term

    return {
        "search_key": search_key,
        "phrase_contains": lambda t: phrase_contains(t, "All of Us"),
        "ampersand_spellings": ampersand_spellings,
        "sanitize_term": sanitize_term,
        "build_for_clause": lambda t: build_for_clause(
            safe=[t], risky=[t], flag_terms=[t], exclusion_terms=[t], variant="V4"),
        "quote_allows_references_only": lambda t: quote_allows_references_only(
            t, [{"term": "All of Us", "part": "references", "text": t}]),
    }


@pytest.mark.parametrize("unit", [",", ":", "'", "’", ".", "A.", "A'", "A-", "A and "])
def test_phrase_matchers_stay_linear_on_long_punctuation_runs(unit: str) -> None:
    """A 300,000-character run of punctuation (or of capitalised words joined by it)
    must not make a matcher backtrack: each one finishes in well under 2 seconds."""
    import time

    text = (unit * (300_000 // len(unit) + 1))[:300_000]
    for name, fn in _phrase_matchers().items():
        start = time.perf_counter()
        fn(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"{name} took {elapsed:.1f}s on {unit!r} x 300,000"

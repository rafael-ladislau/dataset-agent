"""How Dimensions full_data matches a quoted phrase: when two names are one search.

Measured on Dimensions full_data, quoted phrases, 2020-2024 (2026-11-15); the evidence
(count pairs), the dates and what is still unmeasured are in
docs/dimensions-search-semantics.md. The same rule cases are a mirrored table in the tests of dataset-agent, insights (snippet matching)
and extraction (term contexts): ``MEASURED_PHRASE_RULES``.

* hyphen, slash and asterisk fold to a space: "GPT-4" = "GPT 4", "CPS/FSS" = "CPS FSS",
  "O*NET" = "O NET"; the Unicode dashes (U+2011, U+2013) equal the hyphen;
* "," and ";" are separators, with or without a space ("Aurora,Colorado" =
  "Aurora Colorado"); so are parentheses and double quotes inside a phrase;
* ":" is a separator with whitespace on either side ("BioCLIP: A ...") or between a
  digit and a letter ("COVID-19:data"); between two letters it is not ("BioCLIP:A"
  0 vs 32 papers) and stays inside the token;
* an apostrophe (straight or curly) before a final "s" is dropped and the plural then
  folds: "Alzheimer's" = "Alzheimers" = "Alzheimer", but "Alzheimer s" differs;
* "_" is part of the token ("bert_base_uncased" 45 vs "bert base uncased" 7,366);
* a joined form is another token: "GPT4" != "GPT 4", "ONET" != "O NET";
* a dot is a separator between a letter and a digit: "NHANES.12" = "NHANES 12" (97),
  "UK Biobank.23" = "UK Biobank 23" (142), as in a citation superscript; and after a
  letter at the end of a word (before whitespace or the end): "Pfizer Inc." = "Pfizer
  Inc", "U.S. Census Bureau" = "U.S Census Bureau" (58,506). It stays inside the token
  between two letters, "scikit.learn" (101) != "scikit-learn" (153,737), "U.S" !=
  "US" != "U S", and between two digits, "GPT-4.5" (43) != "GPT-4 5" (471);
* "+" and "&" are separators, also between two word characters: "C++" = "C";
  "AT&T" = "AT T" (659,644) != "ATT" (182,335); "Pfizer+BioNTech" = "Pfizer BioNTech";
  "Medical Imaging & Data Resource Center" = "Medical Imaging Data Resource Center"
  (122), which is NOT the "and" spelling (170);
* every token of three or more characters folds singular and plural both ways, inside
  a phrase too: ordinary words ("Program" = "Programs", "Parkinsons ..." =
  "Parkinson ..."), all-caps tokens (NHANES = NHANE, LLM = LLMs, CPS = CP), short words
  (map = maps), letter-and-digit tokens (ERA5 = ERA5s) and numbers ("1990s" = "1990");
  "-ies" folds to "-y" ("registries" = "registry", "studies" = "study"). A token under
  three characters does not fold ("GPT-4s" 78 vs "GPT-4" 52,109: the token is "4s"),
  and "-es" does not fold to the bare stem ("boxes" != "box", "analyses" != "analysis");
* matching is case-insensitive.

The plural rule is the "s-stemmer" of Lucene's English minimal stemmer, which reproduces
every measured case. Its branches that were not measured only keep tokens apart (an
"-ies" after "a" or "e", and "-ies", "-aes", "-oes", "-ees" kept as written).

Conservative where not measured (an extra search, never a lost one): characters that
were not measured stay part of the token; any apostrophe other than one before a final
"s" stays inside the token; ":" between a letter and a digit, between two digits, or at
the edge of the phrase stays inside the token; a dot between a digit and a letter,
after a digit at the end of a word, or before a word stays inside the token; "," or ";"
between two digits stays
inside the token ("1,000" may be one token, as in Unicode word segmentation); the
endings "ss" and "us" and plural tokens shorter than three characters do not fold. An
"is" ending does fold: "MRIs" = "MRI" was measured, and case cannot tell it from
"analysis" because matching is case-insensitive.
"""

from __future__ import annotations

import re
import unicodedata

_DASHES = dict.fromkeys([*range(0x2010, 0x2016), 0x2212, 0xFE58, 0xFE63, 0xFF0D], "-")
#: Characters that fold to a space: "-", "/", "*", "(", ")", '"', "&", "+" (measured:
#: "AT&T" = "AT T", "Pfizer+BioNTech" = "Pfizer BioNTech"). "," and ";" are handled
#: apart (kept between two digits).
_SEPARATORS = dict.fromkeys(map(ord, '-/*&+()"'), " ")
#: An apostrophe before a final "s" is dropped: "Alzheimer's" is "Alzheimers".
_POSSESSIVE = re.compile(r"(?<=\w)['\u2019](?=[sS](?!\w))")
#: "," or ";" except between two digits (not measured there: kept, conservative).
_COMMA = re.compile(r"(?<!\d)[,;]|[,;](?!\d)")
#: A dot between a letter and a digit (measured: "NHANES.12" = "NHANES 12"), or after a
#: letter at the end of a word (measured: "Pfizer Inc." = "Pfizer Inc", "U.S. Census" =
#: "U.S Census").
_DOT = re.compile(r"(?<=[^\W\d_])\.(?=\d|\s|$)")
#: ":" next to whitespace (measured) or between a digit and a letter (measured).
_COLON = re.compile(r"(?<=\s):|:(?=\s)|(?<=\d):(?=[^\W\d_])")


def _folded(term: str) -> str:
    s = unicodedata.normalize("NFKC", term or "").translate(_DASHES)
    s = _POSSESSIVE.sub("", s)
    s = _COMMA.sub(" ", s.translate(_SEPARATORS))
    return _DOT.sub(" ", _COLON.sub(" ", s))


def phrase_tokens(term: str) -> list[str]:
    """The tokens Dimensions matches for the quoted phrase *term*, case-folded."""
    return _folded(term).casefold().split()


def plural_base(token: str) -> str:
    """*token* without its plural: "nhanes" -> "nhane", "era5s" -> "era5", "1990s" ->
    "1990", "registries" -> "registry".

    Lucene's English minimal ("s") stemmer. Measured: tokens under three characters
    do not fold ("4s" of "GPT-4s"); "-ies" becomes "-y"; "-es" loses only the "s"
    ("boxes" -> "boxe", so it stays apart from "box"); "is" folds ("MRIs" = "MRI").
    Not measured, and they only keep tokens apart: the endings "ss" and "us" ("gnss",
    "census"), an "-ies" after "a" or "e", and "-ies", "-aes", "-oes", "-ees" kept."""
    n = len(token)
    if n < 3 or token[-1] != "s" or token[-2] in "us":
        return token
    if token[-2] == "e":
        if n > 3 and token[-3] == "i" and token[-4] not in "ae":
            return token[:-3] + "y"
        if token[-3] in "iaoe":
            return token
    return token[:-1]


def _search_tokens(term: str) -> list[str]:
    return [plural_base(t) for t in phrase_tokens(term)]


def search_key(term: str) -> str:
    """Equal for two names exactly when Dimensions searches them as one phrase
    (within the measured rules): the tokens, each with its plural folded."""
    return " ".join(_search_tokens(term))


def same_search(a: str, b: str) -> bool:
    """True when the quoted phrases *a* and *b* are one Dimensions search."""
    ka = search_key(a)
    return bool(ka) and ka == search_key(b)


def phrase_contains(outer: str, inner: str) -> bool:
    """True when every match of the phrase *outer* is also a match of *inner*: the
    tokens of *inner* occur contiguously in *outer*, each with its plural folded (as in
    :func:`search_key`). A dot between two digits is part of its token, so "GPT-4.5"
    does not contain "GPT-4" and "Prithvi-EO-2.0" does not contain "Prithvi-EO-2"."""
    o, i = _search_tokens(outer), _search_tokens(inner)
    if not i or len(i) > len(o):
        return False
    return any(o[k:k + len(i)] == i for k in range(len(o) - len(i) + 1))


_AMP = re.compile(r"(?<=\S) +& +(?=\S)")
_AND = re.compile(r"(\b[A-Z][\w.'-]*) +and +(?=[A-Z])")


def ampersand_spellings(name: str) -> list[str]:
    """The other spellings of a proper name written with "&" or "and" between words.

    Dimensions reads "&" as a separator ("Medical Imaging & Data Resource Center"
    searches as "Medical Imaging Data Resource Center", 122 papers) and that is not the "and"
    spelling (170 papers), so both are searched. "and" counts only between two
    capitalised words: a lower-case neighbour makes it description, not a name."""
    out: list[str] = []
    if _AMP.search(name):
        out.append(_AMP.sub(" and ", name))
    if _AND.search(name):
        out.append(_AND.sub(r"\1 & ", name))
    return [o for o in dict.fromkeys(out) if o != name]


def ampersand_additions(names: list[str]) -> list[str]:
    """The "&" / "and" spellings of *names* that are not already one of their searches.

    Deduplicated with :func:`same_search`, not by case: "Medical Imaging and Data
    Resource Centers" already searches "... and Data Resource Center"."""
    added: list[str] = []
    for name in names:
        for alt in ampersand_spellings(name):
            if not any(same_search(alt, other) for other in (*names, *added)):
                added.append(alt)
    return added

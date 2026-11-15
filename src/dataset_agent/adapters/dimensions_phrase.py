"""How Dimensions full_data matches a quoted phrase: when two names are one search.

Measured on Dimensions full_data, quoted phrases, 2020-2024 (2026-11-15):

* hyphen, slash and asterisk fold to a space: "GPT-4" = "GPT 4", "CPS/FSS" = "CPS FSS",
  "O*NET" = "O NET"; the Unicode dashes (U+2011, U+2013) equal the hyphen;
* a joined form is another token: "GPT4" != "GPT 4", "ONET" != "O NET";
* a dot is not a separator: "scikit.learn" (101) != "scikit-learn" (153,737), so
  "2.0" is one token and "GPT-4.5" != "GPT-4 5";
* "+" and "&" are dropped: "C++" = "C"; "Medical Imaging & Data Resource Center" =
  "Medical Imaging Data Resource Center" (122), which is NOT the "and" spelling (170);
* a plural matches inside quotes for ordinary words ("Program" = "Programs"); this was
  not measured for all-caps or digit tokens;
* matching is case-insensitive.

Characters that were not measured are kept as part of the token, so two names that
differ only in them stay two searches: an extra search, never a lost one.
"""

from __future__ import annotations

import re
import unicodedata

_DASHES = dict.fromkeys([*range(0x2010, 0x2016), 0x2212, 0xFE58, 0xFE63, 0xFF0D], "-")
#: Characters that fold to a space ("-", "/", "*") or are dropped ("&", "+").
_SEPARATORS = dict.fromkeys(map(ord, "-/*&+"), " ")


def phrase_tokens(term: str) -> list[str]:
    """The tokens Dimensions matches for the quoted phrase *term*, case-folded."""
    s = unicodedata.normalize("NFKC", term or "").translate(_DASHES).translate(_SEPARATORS)
    return s.casefold().split()


def _plural_base(token: str, original: str) -> str:
    """*token* without a plural "s" when it is an ordinary word (lower case or
    capitalised, letters only); acronyms, coined names and digits stay as they are."""
    if (
        original.isalpha()
        and (original.islower() or original.istitle())
        and len(token) >= 4
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]
    return token


def search_key(term: str) -> str:
    """Equal for two names exactly when Dimensions searches them as one phrase
    (within the measured rules): the tokens, with a plural last word folded."""
    tokens = phrase_tokens(term)
    if not tokens:
        return ""
    originals = (
        unicodedata.normalize("NFKC", term or "").translate(_DASHES).translate(_SEPARATORS)
    ).split()
    tokens[-1] = _plural_base(tokens[-1], originals[-1])
    return " ".join(tokens)


def same_search(a: str, b: str) -> bool:
    """True when the quoted phrases *a* and *b* are one Dimensions search."""
    ka = search_key(a)
    return bool(ka) and ka == search_key(b)


def phrase_contains(outer: str, inner: str) -> bool:
    """True when every match of the phrase *outer* is also a match of *inner*: the
    tokens of *inner* occur contiguously in *outer*. A dot is part of its token, so
    "GPT-4.5" does not contain "GPT-4" and "Prithvi-EO-2.0" does not contain
    "Prithvi-EO-2"; no plural folding (it was measured for a whole phrase only)."""
    o, i = phrase_tokens(outer), phrase_tokens(inner)
    if not i or len(i) > len(o):
        return False
    return any(o[k:k + len(i)] == i for k in range(len(o) - len(i) + 1))


_AMP = re.compile(r"(?<=\S) +& +(?=\S)")
_AND = re.compile(r"(\b[A-Z][\w.'-]*) +and +(?=[A-Z])")


def ampersand_spellings(name: str) -> list[str]:
    """The other spellings of a proper name written with "&" or "and" between words.

    Dimensions drops "&" ("Medical Imaging & Data Resource Center" searches as
    "Medical Imaging Data Resource Center", 122 papers) and that is not the "and"
    spelling (170 papers), so both are searched. "and" counts only between two
    capitalised words: a lower-case neighbour makes it description, not a name."""
    out: list[str] = []
    if _AMP.search(name):
        out.append(_AMP.sub(" and ", name))
    if _AND.search(name):
        out.append(_AND.sub(r"\1 & ", name))
    return [o for o in dict.fromkeys(out) if o != name]

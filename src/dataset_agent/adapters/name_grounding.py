"""Ground a descriptive dataset title in the proper names papers actually cite.

Some catalog titles are labels, not product names: "GNSS EarthScope UNAVCO Earthquakes"
describes a data collection, but papers cite "Plate Boundary Observatory", "Network of the
Americas" or "GAGE Facility". Alias research then only rearranges the title's words
("GNSS Earthquake Data"), and the query finds almost nothing.

When the title itself has no Dimensions hits and its aliases barely match, one structured
LLM call (no tool loop) names the products, networks or archives the description refers
to. Names the web results or description define with an acronym ("Network of the
Americas (NOTA)") are candidates too, so an LLM that returns nothing does not end the
search. Candidates are screened here; the caller keeps only those Dimensions finds.
"""

from __future__ import annotations

import re

from dataset_agent.adapters.query_heuristics import is_short_acronym
from dataset_agent.adapters.term_sanitizer import sanitize_terms

GROUNDING_TOOL: dict = {
    "name": "emit_named_products",
    "description": "Proper names of the data products the dataset description refers to.",
    "input_schema": {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Proper names as authors write them in papers; may be empty.",
            }
        },
        "required": ["names"],
    },
}

#: Words that never make a name distinctive on their own.
_GENERIC = frozenset(
    """
    a an and the of for in on at to by with from
    data dataset datasets database databases archive archives catalog catalogue collection
    collections network networks observation observations product products record records
    series time survey surveys study studies program programme project projects portal
    repository service services system systems facility file files set sets resource
    resources information info center centre initiative
    """.split()
)
_WORD = re.compile(r"[\W_]+")


def _fold(word: str) -> str:
    # "earthquakes" and "earthquake" are the same word for the rearrangement test.
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def content_tokens(text: str) -> set[str]:
    """Case-folded, plural-folded words of *text* that are not generic data words."""
    words = (w for w in _WORD.split(text.casefold()) if len(w) > 1 and w not in _GENERIC)
    return {_fold(w) for w in words if _fold(w) not in _GENERIC}


_LINK = frozenset({"of", "the", "and", "for", "on", "in", "de", "&"})
_DEFINED = re.compile(
    r"((?:[A-Z][\w'-]*|of|the|and|for|on|in|de|&)(?:\s+(?:[A-Z][\w'-]*|of|the|and|for|on|in|de|&)){1,7})"
    r"\s*\(([A-Z][A-Za-z]{1,9})\)"
)


def defined_names(text: str) -> list[tuple[str, str]]:
    """``(name, acronym)`` for proper names written with their acronym in *text*.

    ``"Network of the Americas (NOTA)"`` gives ``("Network of the Americas", "NOTA")``.
    The acronym must spell the initials of the name's words (capitalized words only, or
    all words), so a phrase that merely precedes a parenthesis ("Data Portal (beta)") is
    not taken."""
    out: list[tuple[str, str]] = []
    for m in _DEFINED.finditer(text or ""):
        words = m.group(1).split()
        acronym = m.group(2)
        key = acronym.upper()
        # The shortest tail of the phrase whose initials spell the acronym.
        for start in range(len(words) - 1, -1, -1):
            tail = words[start:]
            if tail[0].casefold() in _LINK:
                continue
            caps = "".join(w[0] for w in tail if w.casefold() not in _LINK).upper()
            every = "".join(w[0] for w in tail if w != "&").upper()
            if key in (caps, every):
                name = " ".join(tail)
                if len(tail) >= 2 and name.casefold() not in {n.casefold() for n, _ in out}:
                    out.append((name, acronym))
                break
            if len(caps) > len(key):
                break
    return out


def needs_grounding(c_full: int, alias_counts: list[int], min_hits: int) -> bool:
    """The title has no hits and all aliases together find fewer than *min_hits*."""
    return c_full <= 0 and sum(max(0, c) for c in alias_counts) < min_hits


def grounding_prompt(
    dataset_name: str,
    description: str,
    alias_counts: list[tuple[str, int]],
    flag_terms: list[str],
    web_results: str,
) -> str:
    counts = "\n".join(f"  - {a!r}: {n} papers" for a, n in alias_counts[:15]) or "  (none)"
    flags = ", ".join(flag_terms[:10]) or "(none)"
    return f"""The catalog title {dataset_name!r} returns no papers in a literature search, and the
aliases below find almost none. The title is probably a descriptive label for a data
collection, not the name authors cite.

=== DESCRIPTION ===
{(description or "(none)")[:2500]}

=== ALIASES TRIED (exact-phrase hits in papers) ===
{counts}

=== ORGANIZATIONS (already searched separately) ===
{flags}

=== WEB RESULTS ===
{(web_results or "(none)")[:4000]}

=== YOUR TASK ===
Call emit_named_products with the proper names of the specific datasets, observatories,
networks, archives or data products that this collection consists of or is distributed
as, written the way authors cite them in papers (for example "Plate Boundary Observatory"
for a GNSS station collection). Rules:
- Only names supported by the description or the web results; never invent a name.
- No generic phrases ("GNSS data", "earthquake catalog") and no rearrangement of the
  title's own words.
- No organization name on its own (those are listed above).
- At most 8 names; an empty list if none is supported."""


def screen_candidates(
    candidates: list[str],
    *,
    dataset_name: str,
    known_aliases: list[str],
    flag_terms: list[str],
    limit: int,
) -> tuple[list[str], list[str]]:
    """Candidates worth counting, plus ``"<name> (<reason>)"`` notes for the rest."""
    title_tokens = content_tokens(dataset_name)
    taken = {t.casefold() for t in [*known_aliases, *flag_terms, dataset_name]}
    kept: list[str] = []
    rejected: list[str] = []
    for name in sanitize_terms([c for c in candidates if isinstance(c, str)], "grounded name"):
        key = name.casefold()
        if key in taken:
            continue
        taken.add(key)
        tokens = content_tokens(name)
        if is_short_acronym(name):
            reason = "bare short acronym"
        elif not any(ch.isupper() for ch in name):
            reason = "not a proper name"
        elif not tokens:
            reason = "generic words only"
        elif tokens <= title_tokens:
            reason = "rearranges the title"
        elif len(kept) >= limit:
            reason = "over the candidate limit"
        else:
            kept.append(name)
            continue
        rejected.append(f"{name} ({reason})")
    return kept, rejected

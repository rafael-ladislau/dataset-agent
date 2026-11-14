"""Resource map: does the dataset name a *program* that is separate from its *data*?

Some catalog entries name a research program (it recruits participants, runs a study,
funds work) whose data is reached through separately named resources (an access
platform, a repository, tiers). Papers about the program are not papers about its data,
so the query and the full-text judgment must tell them apart (``scope_check``).

One structured LLM step proposes a kind for each research name (``program``,
``data_resource``, ``organization``, ``other``), data resource names it read in the
fetched pages, former or parent names of the program, and the year the data was first
released. The split itself is deterministic; there is a map only when ALL hold:

* the dataset's main name is classified ``program`` and is not itself a data noun
  (``scope_check.name_denotes_data``: commons, survey, database, data, cohort ...);
* at least one data resource name occurs literally in the fetched pages and is not
  contained in a program name;
* that data resource name has Dimensions hits together with the program names that it
  does not contain (``min_cooccurrence``): "<program> Workbench" AND "<program>" would
  count the data name alone.

A data name that starts with a program name also brings its remainder ("<program>
Researcher Workbench" -> "Researcher Workbench") when the remainder has co-occurrence
hits; the map keeps the union of spellings. A data name is *specific* when it is not
generic, has at most ``max_hits`` hits on its own, and at least ``specific_min_share`` of
them also carry a program name: it may then stand alone as a safe alias (subject to the
probe). A generic one ("Data Browser") only ever gates the program names.

Otherwise there is no map and nothing changes. Former and parent names are kept only
when they occur literally in the pages; the release year only with a quote found in the
pages. General rules only: no dataset names live in this module.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import httpx

from dataset_agent.adapters.fulltext_evidence import _phrase_in
from dataset_agent.adapters.scope_check import (
    DATA_LIST_NOUNS,
    PROGRAM_LIST_NOUNS,
    name_denotes_data,
)
from dataset_agent.domain.models import ResourceMap

logger = logging.getLogger(__name__)

KINDS = ("program", "data_resource", "organization", "other")

EMIT_RESOURCE_MAP = {
    "name": "emit_resource_map",
    "description": "Classify the dataset's names as program or data resource.",
    "input_schema": {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"},
                                   "kind": {"type": "string", "enum": list(KINDS)}},
                    "required": ["name", "kind"],
                },
            },
            "data_resource_names_in_pages": {"type": "array", "items": {"type": "string"}},
            "former_or_parent_program_names": {"type": "array", "items": {"type": "string"}},
            "first_data_release_year": {"type": ["integer", "null"]},
            "first_data_release_quote": {"type": "string"},
        },
        "required": ["names"],
    },
}

_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)


def fetch_page_text(url: str | None, *, max_chars: int = 60_000) -> str:
    """Visible text of one page (best effort; empty on any error)."""
    if not url:
        return ""
    try:
        r = httpx.get(url, timeout=30.0, follow_redirects=True)
        if not r.is_success:
            logger.warning("resource map: fetch %s returned %s", url, r.status_code)
            return ""
        return " ".join(_TAGS.sub(" ", r.text or "").split())[:max_chars]
    except Exception as exc:
        logger.warning("resource map: fetch %s failed: %s", url, exc)
        return ""


def resource_map_prompt(*, dataset_name: str, description: str, names: list[str],
                        pages: str, max_page_chars: int = 12_000) -> str:
    listed = "\n".join(f"- {n}" for n in names)
    return f"""Dataset (catalog entry): {dataset_name!r}.
Description: {" ".join((description or "").split())}
Names found by research:
{listed}

Fetched pages (excerpt):
{pages[:max_page_chars]}

Classify each name:
- program: a research program, initiative or study that recruits participants, collects
  data, funds or runs research; its data is reached through resources with their own names.
- data_resource: a data set, database, survey, data commons, repository, data center,
  observation network, access platform, data browser, tier or release.
- organization: an agency, institute, consortium, company or funder.
- other: anything else.
If the catalog entry's own name already denotes the data (a survey, a database, a data
center or commons), classify it data_resource, not program.
Noun test (the same one the paper-level check uses): a name whose head noun is one of
{DATA_LIST_NOUNS} is data_resource; one of {PROGRAM_LIST_NOUNS} is program.

Also give: data_resource_names_in_pages (names of this program's data access platform,
repository, tiers or data browser exactly as written in the pages); former_or_parent_program_names
(earlier names or the parent initiative of the program, exactly as written in the pages);
first_data_release_year (the earliest year ANY of the program's data was made available
to researchers: any data type, including a beta, a demonstration release or a public data
browser; null if the pages do not say) with first_data_release_quote (the sentence from the
pages that says so).

Use emit_resource_map exactly once."""


def propose(agent, *, dataset_name: str, description: str, names: list[str],
            pages: str) -> dict:
    """The LLM proposal (empty dict when the call failed)."""
    try:
        raw = agent.get_structured(
            resource_map_prompt(dataset_name=dataset_name, description=description,
                                names=names, pages=pages),
            EMIT_RESOURCE_MAP,
        )
    except Exception as exc:
        logger.warning("resource map: LLM call failed: %s", exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def _in_pages(term: str, pages: str) -> bool:
    return bool(term.strip()) and _phrase_in(term, pages, plural_ok=False)


def _contained(inner: str, outer: str) -> bool:
    return _phrase_in(inner, outer, plural_ok=False)


def _quoted(term: str) -> str:
    return '\\"' + " ".join(term.replace('"', " ").split()) + '\\"'


def cooccurrence_clause(program_names: list[str], data_name: str) -> str:
    """Dimensions FOR clause: any program name AND the data resource name."""
    group = " OR ".join(_quoted(p) for p in program_names)
    return f"({group}) AND ({_quoted(data_name)})"


def candidates(raw: dict, *, dataset_name: str, pages: str) -> tuple[list[str], list[str]] | None:
    """Program names and the data resource names still to count, or None (no map)."""
    rows = [r for r in (raw.get("names") or []) if isinstance(r, dict)]
    kind = {" ".join(str(r.get("name") or "").split()).lower(): r.get("kind") for r in rows}
    if kind.get(" ".join(dataset_name.split()).lower()) != "program":
        return None
    if name_denotes_data(dataset_name):
        logger.info("resource map: %r is itself a data name; no split", dataset_name)
        return None
    programs = [str(r["name"]).strip() for r in rows if r.get("kind") == "program"]
    programs += [n.strip() for n in (raw.get("former_or_parent_program_names") or [])
                 if isinstance(n, str) and _in_pages(n, pages)]
    programs = list(dict.fromkeys(p for p in programs if p))
    data = [str(r["name"]).strip() for r in rows if r.get("kind") == "data_resource"]
    data += [n.strip() for n in (raw.get("data_resource_names_in_pages") or []) if isinstance(n, str)]
    data = [d for d in dict.fromkeys(data) if d and _in_pages(d, pages)
            and not any(_contained(d, p) for p in programs)]
    return (programs, data) if programs and data else None


def remainders(data: list[str], programs: list[str]) -> list[str]:
    """For each data name that starts with a program name, the rest of it when it has two
    or more words ("<program> Researcher Workbench" -> "Researcher Workbench"); longest
    program first."""
    from dataset_agent.adapters.query_heuristics import is_short_acronym

    out: list[str] = []
    have = {d.casefold() for d in data}
    for d in data:
        for p in sorted(programs, key=len, reverse=True):
            if d.casefold().startswith(p.casefold() + " "):
                rest = d[len(p):].strip(" -:,")
                # One word left ("Workbench") is a kind of thing, not a name.
                if (len(rest.split()) >= 2 and rest.casefold() not in have
                        and re.search(r"[A-Za-z]", rest) and not is_short_acronym(rest)):
                    out.append(rest)
                    have.add(rest.casefold())
                break
    return out


# Words that describe a kind of data resource without naming one.
_EXTRA_GENERIC = frozenset({"browser", "tier", "tiers", "public", "controlled", "registered",
                            "release", "version", "access"})


def is_generic_data_name(name: str) -> bool:
    """Every word describes a kind of resource ("Data Browser", "Public Tier dataset")."""
    from dataset_agent.adapters.fulltext_recall import _GENERIC_NAME_WORDS

    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in {"of", "the", "and"}]
    return bool(words) and all(w in _GENERIC_NAME_WORDS or w in _EXTRA_GENERIC for w in words)


def specific_names(hits: dict[str, int], alone: dict[str, int], *, min_share: float,
                   max_hits: int) -> list[str]:
    """Data names that may stand alone: not generic, not a short acronym, at most
    *max_hits* hits on their own, and at least *min_share* of them with a program name."""
    from dataset_agent.adapters.query_heuristics import is_short_acronym

    out = []
    for d, n in hits.items():
        total = alone.get(d)
        if (not total or total > max_hits or is_generic_data_name(d) or is_short_acronym(d)):
            continue
        if n / total >= min_share:
            out.append(d)
    return out


def release_year(raw: dict, pages: str, *, lo: int = 1950, hi: int = 2100) -> int | None:
    """The first data release year, only with a quote found in the pages that holds it."""
    year, quote = raw.get("first_data_release_year"), str(raw.get("first_data_release_quote") or "")
    if not isinstance(year, int) or not (lo <= year <= hi) or str(year) not in quote:
        return None
    head = " ".join(quote.split()[:12])
    return year if head and _in_pages(head, pages) else None


def data_clause(programs: list[str], data_names: list[str]) -> str:
    """Any program name AND any data resource name."""
    return (f"({' OR '.join(_quoted(p) for p in programs)}) AND "
            f"({' OR '.join(_quoted(d) for d in data_names)})")


def confirm_release_year(year: int | None, programs: list[str], data_names: list[str], *,
                         count: Callable[[str, str], int | None], max_share_before: float
                         ) -> tuple[int | None, float | None]:
    """Keep *year* only when few program-and-data papers come before it. *count* takes a
    clause and a ``where`` filter. Returns (year or None, share of hits before it)."""
    if year is None:
        return None, None
    clause = data_clause(programs, data_names)
    total, before = count(clause, ""), count(clause, f"year < {year}")
    if not total or before is None:
        return None, None
    share = before / total
    return (year if share <= max_share_before else None), share


def build_map(raw: dict, *, dataset_name: str, pages: str,
              count: Callable[[str], int | None], min_cooccurrence: int = 1,
              count_where: Callable[[str, str], int | None] | None = None,
              max_share_before: float = 0.05, specific_min_share: float = 0.5,
              max_hits: int = 500_000) -> ResourceMap | None:
    """Apply the deterministic split rule; *count* returns Dimensions hits of a clause and
    *count_where* (optional) the hits under a ``where`` filter, to confirm the release year."""
    cand = candidates(raw, dataset_name=dataset_name, pages=pages)
    if cand is None:
        return None
    programs, data = cand
    data = [*data, *remainders(data, programs)]
    hits: dict[str, int] = {}
    for d in data:
        others = [p for p in programs if not _contained(p, d)]
        if not others:
            logger.info("resource map: %r contains every program name; co-occurrence not "
                        "measurable", d)
            continue
        n = count(cooccurrence_clause(others, d))
        if n is not None and n >= min_cooccurrence:
            hits[d] = n
    if not hits:
        return None
    alone: dict[str, int] = {}
    for d in hits:
        n = count(_quoted(d))
        if n is not None:
            alone[d] = n
    specific = specific_names(hits, alone, min_share=specific_min_share, max_hits=max_hits)
    former = [n for n in (raw.get("former_or_parent_program_names") or [])
              if isinstance(n, str) and n.strip() in programs]
    proposed = release_year(raw, pages)
    year, share = (proposed, None)
    if count_where is not None:
        year, share = confirm_release_year(proposed, programs, list(hits), count=count_where,
                                           max_share_before=max_share_before)
    return ResourceMap(program_names=programs, data_resource_names=list(hits),
                       former_or_parent_names=former, first_release_year=year,
                       proposed_release_year=proposed, share_before_release=share,
                       cooccurrence=hits, data_name_hits=alone, specific_data_names=specific)


def gather_pages_with_sources(dataset_name: str, home_url: str | None, *, search=None,
                              fetch=None) -> tuple[str, list[str]]:
    """The home page text plus three web searches (data access, history, first data
    release); when the home page cannot be fetched, two more searches (data portal and
    documentation pages) stand in for it. Returns (text, sources)."""
    from dataset_agent.adapters.tools import web_search

    search = search or web_search
    fetch = fetch or fetch_page_text
    parts: list[str] = []
    sources: list[str] = []
    home = fetch(home_url) if home_url else ""
    if home:
        parts.append(home)
        sources.append(str(home_url))
    elif home_url:
        sources.append(f"failed: {home_url}")
    queries = [f"{dataset_name} data access researchers",
               f"{dataset_name} history formerly known as launched by initiative",
               f"{dataset_name} first data release to researchers year"]
    if not home:
        queries += [f"{dataset_name} data portal researcher platform",
                    f"{dataset_name} data documentation support pages"]
    for q in queries:
        try:
            text = search(q)
        except Exception as exc:  # web context is best effort
            logger.warning("resource map: web search failed: %s", exc)
            continue
        if text:
            parts.append(text)
            sources.append(f"search: {q}")
    logger.info("resource map: pages for %r from %s", dataset_name, sources)
    return "\n".join(parts), sources


def gather_pages(dataset_name: str, home_url: str | None, *, search=None, fetch=None) -> str:
    """The text of :func:`gather_pages_with_sources`."""
    return gather_pages_with_sources(dataset_name, home_url, search=search, fetch=fetch)[0]


def map_options(settings) -> dict:
    """The map keyword arguments taken from Settings."""
    return {"min_cooccurrence": settings.resource_map_min_cooccurrence,
            "max_share_before": settings.resource_map_max_share_before_release,
            "specific_min_share": settings.resource_map_specific_min_share,
            "max_hits": settings.optimize_flag_max_hits}


def map_for_record(agent, record, *, count: Callable[[str], int | None],
                   min_cooccurrence: int = 1, pages: str | None = None,
                   count_where: Callable[[str, str], int | None] | None = None,
                   max_share_before: float = 0.05, specific_min_share: float = 0.5,
                   max_hits: int = 500_000, sources: list[str] | None = None
                   ) -> ResourceMap | None:
    """Research the resource map of a DatasetRecord (pages fetched unless given)."""
    name = record.main_dataset_name
    if pages is None:
        pages, sources = gather_pages_with_sources(name, record.home_url)
    names = list(dict.fromkeys([name, *record.dataset_names, *record.flag_terms]))
    raw = propose(agent, dataset_name=name, description=record.description, names=names,
                  pages=pages)
    m = build_map(raw, dataset_name=name, pages=pages, count=count,
                  min_cooccurrence=min_cooccurrence, count_where=count_where,
                  max_share_before=max_share_before, specific_min_share=specific_min_share,
                  max_hits=max_hits)
    if m is not None:
        m.page_sources = list(sources or [])
    logger.info("resource map for %r: %s", name, m.model_dump() if m else "none")
    return m

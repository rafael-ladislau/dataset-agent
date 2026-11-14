"""Resource map: does the dataset name a *program* that is separate from its *data*?

Some catalog entries name a research program (it recruits participants, runs a study,
funds work) whose data is reached through separately named resources (an access
platform, a repository, tiers). Papers about the program are not papers about its data,
so the query and the full-text judgment must tell them apart (``scope_check``).

One structured LLM step proposes a kind for each research name (``program``,
``data_resource``, ``organization``, ``other``), data resource names it read in the
fetched pages, former or parent names of the program, and the year the data was first
released. The split itself is deterministic; there is a map only when ALL hold:

* the dataset's main name is classified ``program``;
* at least one data resource name occurs literally in the fetched pages and is not
  contained in a program name;
* that data resource name has Dimensions hits together with the program names
  (``min_cooccurrence``).

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

Also give: data_resource_names_in_pages (names of this program's data access platform,
repository, tiers or data browser exactly as written in the pages); former_or_parent_program_names
(earlier names or the parent initiative of the program, exactly as written in the pages);
first_data_release_year (the year the program's data was first released to researchers,
or null) with first_data_release_quote (the sentence from the pages that says so).

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
    programs = [str(r["name"]).strip() for r in rows if r.get("kind") == "program"]
    programs += [n.strip() for n in (raw.get("former_or_parent_program_names") or [])
                 if isinstance(n, str) and _in_pages(n, pages)]
    programs = list(dict.fromkeys(p for p in programs if p))
    data = [str(r["name"]).strip() for r in rows if r.get("kind") == "data_resource"]
    data += [n.strip() for n in (raw.get("data_resource_names_in_pages") or []) if isinstance(n, str)]
    data = [d for d in dict.fromkeys(data) if d and _in_pages(d, pages)
            and not any(_contained(d, p) for p in programs)]
    return (programs, data) if programs and data else None


def release_year(raw: dict, pages: str, *, lo: int = 1950, hi: int = 2100) -> int | None:
    """The first data release year, only with a quote found in the pages that holds it."""
    year, quote = raw.get("first_data_release_year"), str(raw.get("first_data_release_quote") or "")
    if not isinstance(year, int) or not (lo <= year <= hi) or str(year) not in quote:
        return None
    head = " ".join(quote.split()[:12])
    return year if head and _in_pages(head, pages) else None


def build_map(raw: dict, *, dataset_name: str, pages: str,
              count: Callable[[str], int | None], min_cooccurrence: int = 1) -> ResourceMap | None:
    """Apply the deterministic split rule; *count* returns Dimensions hits of a clause."""
    cand = candidates(raw, dataset_name=dataset_name, pages=pages)
    if cand is None:
        return None
    programs, data = cand
    hits: dict[str, int] = {}
    for d in data:
        n = count(cooccurrence_clause(programs, d))
        if n is not None and n >= min_cooccurrence:
            hits[d] = n
    if not hits:
        return None
    former = [n for n in (raw.get("former_or_parent_program_names") or [])
              if isinstance(n, str) and n.strip() in programs]
    return ResourceMap(program_names=programs, data_resource_names=list(hits),
                       former_or_parent_names=former,
                       first_release_year=release_year(raw, pages), cooccurrence=hits)


def gather_pages(dataset_name: str, home_url: str | None, *, search=None, fetch=None) -> str:
    """The home page text plus three web searches: data access, history (former and
    parent names), first data release."""
    from dataset_agent.adapters.tools import web_search

    search = search or web_search
    fetch = fetch or fetch_page_text
    parts = [fetch(home_url)]
    for q in (f"{dataset_name} data access researchers",
              f"{dataset_name} history formerly known as launched by initiative",
              f"{dataset_name} first data release to researchers year"):
        try:
            parts.append(search(q))
        except Exception as exc:  # web context is best effort
            logger.warning("resource map: web search failed: %s", exc)
    return "\n".join(p for p in parts if p)


def map_for_record(agent, record, *, count: Callable[[str], int | None],
                   min_cooccurrence: int = 1, pages: str | None = None) -> ResourceMap | None:
    """Research the resource map of a DatasetRecord (pages fetched unless given)."""
    name = record.main_dataset_name
    pages = gather_pages(name, record.home_url) if pages is None else pages
    names = list(dict.fromkeys([name, *record.dataset_names, *record.flag_terms]))
    raw = propose(agent, dataset_name=name, description=record.description, names=names,
                  pages=pages)
    m = build_map(raw, dataset_name=name, pages=pages, count=count,
                  min_cooccurrence=min_cooccurrence)
    logger.info("resource map for %r: %s", name, m.model_dump() if m else "none")
    return m

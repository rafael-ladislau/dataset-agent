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

class ResourceMapUndecided(Exception):
    """The map could not be decided (the LLM call failed or gave no answer, or no page
    could be read). Never a firm "no split": the caller records it and may draw again."""


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


#: Window of page text the map LLM reads (search snippets are always kept whole).
MAX_PAGE_CHARS = 12_000
#: Floor of each fetched page's slice when the snippets fill the window.
MIN_PAGE_SLICE = 1_500


def prompt_excerpt(parts: list[tuple[str, str]], *, max_chars: int = MAX_PAGE_CHARS) -> str:
    """The page text the LLM reads, budgeted per source: every search's snippets first and
    whole (one search per section), then a slice of each fetched page sharing the rest of
    the window evenly (at least ``MIN_PAGE_SLICE`` each). One long page can no longer push
    a later search out of the window."""
    searches = [(s, t) for s, t in parts if str(s).startswith("search:") and t]
    pages = [(s, t) for s, t in parts if not str(s).startswith("search:") and t]
    out = [f"[{s}]\n{t}" for s, t in searches]
    if pages:
        used = sum(len(x) for x in out)
        share = max((max_chars - used) // len(pages), MIN_PAGE_SLICE)
        out += [f"[page: {s}]\n{t[:share]}" for s, t in pages]
    return "\n".join(out)


def resource_map_prompt(*, dataset_name: str, description: str, names: list[str],
                        pages: str, max_page_chars: int = MAX_PAGE_CHARS,
                        parts: list[tuple[str, str]] | None = None) -> str:
    """With *parts* (source, text), the window is budgeted per source (``prompt_excerpt``);
    otherwise it is the first *max_page_chars* of *pages*."""
    listed = "\n".join(f"- {n}" for n in names)
    excerpt = (prompt_excerpt(parts, max_chars=max_page_chars) if parts
               else pages[:max_page_chars])
    return f"""Dataset (catalog entry): {dataset_name!r}.
Description: {" ".join((description or "").split())}
Names found by research:
{listed}

Fetched pages (excerpt):
{excerpt}

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
            pages: str, parts: list[tuple[str, str]] | None = None) -> dict:
    """The LLM proposal (empty dict when the call failed)."""
    try:
        raw = agent.get_structured(
            resource_map_prompt(dataset_name=dataset_name, description=description,
                                names=names, pages=pages, parts=parts),
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


def _note(found: dict | None, name: str, **kw) -> None:
    """Record what happened to a data-name candidate (first source wins, later facts add)."""
    if found is None or not name:
        return
    row = found.setdefault(name.casefold(), {"name": name})
    for k, v in kw.items():
        if k in ("source", "where") and row.get(k):
            continue
        row[k] = v


def candidates(raw: dict, *, dataset_name: str, pages: str,
               found: dict | None = None) -> tuple[list[str], list[str]] | None:
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
    data = _literal_data(data, programs, pages, found, source="pages")
    return (programs, data) if programs and data else None


def _literal_data(data: list[str], programs: list[str], pages: str, found: dict | None, *,
                  source: str) -> list[str]:
    """Candidates written in *pages* and not inside a program name (others are recorded)."""
    kept = []
    for d in dict.fromkeys(x for x in data if x):
        if not _in_pages(d, pages):
            _note(found, d, source=source, kept=False, reason="not written in the pages read")
        elif any(_contained(d, p) for p in programs):
            _note(found, d, source=source, kept=False, reason="inside a program name")
        else:
            _note(found, d, source=source)
            kept.append(d)
    return kept


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


def check_data_names(data: list[str], programs: list[str], *,
                     count: Callable[[str], int | None], min_cooccurrence: int = 1,
                     min_share: float | None = None, found: dict | None = None,
                     source: str = "") -> tuple[dict[str, int], dict[str, int]]:
    """Co-occurrence and own hits of data-name candidates (plus remainders of names that
    start with a program name). A candidate is kept with at least *min_cooccurrence* hits
    together with the program names it does not contain and, when *min_share* is given,
    at least that share of its own hits. Returns (co-occurrence hits, own hits)."""
    rests = remainders(data, programs)
    for r in rests:
        _note(found, r, source=source, where="remainder of a program-prefixed name")
    hits: dict[str, int] = {}
    alone: dict[str, int] = {}
    for d in dict.fromkeys([*data, *rests]):
        others = [p for p in programs if not _contained(p, d)]
        if not others:
            logger.info("resource map: %r contains every program name; co-occurrence not "
                        "measurable", d)
            _note(found, d, kept=False, reason="contains every program name: co-occurrence "
                  "not measurable (its remainder is checked instead)")
            continue
        n = count(cooccurrence_clause(others, d))
        if n is None or n < min_cooccurrence:
            _note(found, d, hits_with_program=n, kept=False,
                  reason="count failed" if n is None else f"{n} hits with the program names")
            continue
        a = count(_quoted(d))
        # A remainder ("Research Hub" from "<program> Research Hub") is derived, not
        # read: it must carry a program name in at least half of its own hits.
        share = min_share if d not in rests else max(min_share or 0.0, 0.5)
        if share and (not a or n / a < share):
            logger.info("resource map: %r rejected: %s of %s hits with the program", d, n, a)
            _note(found, d, hits_with_program=n, hits_alone=a, kept=False,
                  reason=f"only {n} of {a} hits carry a program name")
            continue
        hits[d] = n
        if a is not None:
            alone[d] = a
        _note(found, d, hits_with_program=n, hits_alone=a, kept=True, reason="")
    return hits, alone


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
    found: dict = {}
    cand = candidates(raw, dataset_name=dataset_name, pages=pages, found=found)
    if cand is None:
        return None
    programs, data = cand
    hits, alone = check_data_names(data, programs, count=count, min_cooccurrence=min_cooccurrence,
                                   found=found, source="pages")
    if not hits:
        return None
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
                       cooccurrence=hits, data_name_hits=alone, specific_data_names=specific,
                       name_sources={d: "pages" for d in hits},
                       found_names=list(found.values()))


def gather_pages_with_sources(dataset_name: str, home_url: str | None, *, search=None,
                              fetch=None, parts_out: list | None = None) -> tuple[str, list[str]]:
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
        if parts_out is not None:
            parts_out.append((str(home_url), home))
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
            if parts_out is not None:
                parts_out.append((f"search: {q}", text))
    logger.info("resource map: pages for %r from %s", dataset_name, sources)
    return "\n".join(parts), sources


def gather_pages(dataset_name: str, home_url: str | None, *, search=None, fetch=None) -> str:
    """The text of :func:`gather_pages_with_sources`."""
    return gather_pages_with_sources(dataset_name, home_url, search=search, fetch=fetch)[0]


# Data documentation a split program usually publishes (general wording, no dataset names).
DISCOVERY_QUERIES = (
    "{name} data access for researchers",
    "{name} data dictionary codebook",
    "{name} user guide support site data",
    "{name} data tiers data releases",
    "{name} data portal",
)
_URL_LINE = re.compile(r"^\s+(https?://\S+)\s*$", re.M)


def discovery_pages(dataset_name: str, *, search=None, fetch=None, per_query: int = 2,
                    max_pages: int = 6, parts: list | None = None) -> tuple[str, list[str]]:
    """Targeted searches for the program's data documentation; the top result pages
    themselves are fetched, not only the snippets. Returns (text, sources)."""
    from dataset_agent.adapters.tools import web_search

    search = search or web_search
    fetch = fetch or fetch_page_text
    texts: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()
    fetched = 0  # successful fetches only: a failed URL does not use up the cap
    for tpl in DISCOVERY_QUERIES:
        q = tpl.format(name=dataset_name)
        try:
            res = search(q) or ""
        except Exception as exc:  # web context is best effort
            logger.warning("resource map: discovery search failed: %s", exc)
            continue
        texts.append(res)
        sources.append(f"search: {q}")
        if parts is not None:
            parts.append((f"search: {q}", res))
        for url in _URL_LINE.findall(res)[:per_query]:
            if url in seen or fetched >= max_pages:
                continue
            seen.add(url)
            text = fetch(url)
            if text:
                fetched += 1
                texts.append(text)
                sources.append(url)
                if parts is not None:
                    parts.append((url, text))
            else:
                sources.append(f"failed: {url}")
    logger.info("resource map: discovery pages for %r from %s", dataset_name, sources)
    return "\n".join(p for p in texts if p), sources


def merge_names(rmap: ResourceMap, hits: dict[str, int], alone: dict[str, int], *, source: str,
                specific_min_share: float, max_hits: int) -> list[str]:
    """Add newly checked data names to *rmap* in place; returns the names added."""
    have = {d.casefold() for d in rmap.data_resource_names}
    new = {d: n for d, n in hits.items() if d.casefold() not in have}
    if not new:
        return []
    rmap.data_resource_names = [*rmap.data_resource_names, *new]
    rmap.cooccurrence.update(new)
    rmap.data_name_hits.update({d: alone[d] for d in new if d in alone})
    rmap.specific_data_names = [
        *rmap.specific_data_names,
        *specific_names(new, alone, min_share=specific_min_share, max_hits=max_hits),
    ]
    rmap.name_sources.update({d: source for d in new})
    return list(new)


def discover_from_web(agent, rmap: ResourceMap, *, dataset_name: str, description: str,
                      count: Callable[[str], int | None], min_cooccurrence: int = 1,
                      specific_min_share: float = 0.5, max_hits: int = 500_000,
                      search=None, fetch=None) -> list[str]:
    """Only for a dataset already split: more data names from documentation pages. The
    LLM proposes; a name is kept only when it is written in those pages, is not inside a
    program name and co-occurs with the program names (the same checks as the map)."""
    parts: list = []
    pages, sources = discovery_pages(dataset_name, search=search, fetch=fetch, parts=parts)
    rmap.page_sources = [*rmap.page_sources, *sources]
    if not pages:
        return []
    raw = propose(agent, dataset_name=dataset_name, description=description,
                  names=[*rmap.program_names, *rmap.data_resource_names], pages=pages,
                  parts=parts)
    rows = [r for r in (raw.get("names") or []) if isinstance(r, dict)]
    cands = [str(r.get("name") or "").strip() for r in rows if r.get("kind") == "data_resource"]
    cands += [n.strip() for n in (raw.get("data_resource_names_in_pages") or [])
              if isinstance(n, str)]
    programs = list(rmap.program_names)
    found = {r["name"].casefold(): r for r in rmap.found_names}
    have = {d.casefold() for d in rmap.data_resource_names}
    cands = _literal_data([c for c in cands if c.casefold() not in have], programs, pages,
                          found, source="discovery")
    for c in cands:
        _note(found, c, where=locate(c, parts))
    hits, alone = check_data_names(cands, programs, count=count,
                                   min_cooccurrence=min_cooccurrence, found=found,
                                   source="discovery")
    rmap.found_names = list(found.values())
    return merge_names(rmap, hits, alone, source="discovery",
                       specific_min_share=specific_min_share, max_hits=max_hits)


_CAP_NAME = re.compile(
    r"\b[A-Z][\w'&-]*(?:\s+(?:of|the|and|for|[A-Z][\w'&-]*))*\s+[A-Z][\w'&-]*")
_LEADING = frozenset({"in", "the", "we", "our", "this", "these", "from", "using", "with", "a",
                      "an", "by", "for", "to", "of", "at", "on", "as", "and", "data", "via"})


def _trim(words: list[str]) -> list[str]:
    while words and words[0].lower() in _LEADING:
        words = words[1:]
    while words and words[-1].lower() in {"of", "the", "and", "for"}:
        words = words[:-1]
    return words


def _name_segments(name: str, program_names: list[str]) -> list[str]:
    """The parts of a capitalised run around any program name in it ("Controlled Tier
    Dataset of the <program>" -> "Controlled Tier Dataset")."""
    segments = [name]
    for p in sorted(program_names, key=len, reverse=True):
        pat = re.compile(r"(?<!\w)" + re.escape(p) + r"(?!\w)", re.I)
        segments = [s for seg in segments for s in pat.split(seg)]
    return [" ".join(w) for w in (_trim(s.split()) for s in segments) if w]


def literature_candidates(passages_by_paper: dict[str, list[str]], program_names: list[str],
                          known: list[str], *, min_papers: int = 2) -> list[str]:
    """Capitalised names in passages that also carry a program name, found in at least
    *min_papers* papers. A name that starts with a program name gives its remainder."""
    papers: dict[str, set[str]] = {}
    shown: dict[str, str] = {}
    known_f = {k.casefold() for k in [*known, *program_names]}
    for pid, passages in passages_by_paper.items():
        for text in passages:
            if not any(_phrase_in(p, text, plural_ok=False) for p in program_names):
                continue
            for m in _CAP_NAME.finditer(text):
                for name in _name_segments(m.group(0), program_names):
                    if len(name.split()) < 2 or name.casefold() in known_f:
                        continue
                    if any(_contained(name, p) for p in program_names):
                        continue
                    papers.setdefault(name.casefold(), set()).add(pid)
                    shown.setdefault(name.casefold(), name)
    return [shown[k] for k, ps in papers.items() if len(ps) >= min_papers]


def locate(name: str, parts: list) -> str:
    """The first page or search whose text holds *name* literally ("" when unknown)."""
    for src, text in parts or []:
        if _in_pages(name, text):
            return src
    return ""


_KIND_WORDS = (
    ("access_tier", {"tier", "tiers"}),
    ("release", {"release", "releases", "version", "versions"}),
    ("platform", {"workbench", "platform", "hub", "portal", "browser", "enclave", "gateway"}),
    ("repository", {"repository", "repositories", "commons", "archive", "archives", "bank"}),
    ("data_type", {"data", "dataset", "datasets", "survey", "surveys", "cohort", "records",
                   "biobank"}),
)


def data_kind(name: str) -> str:
    """platform | repository | access_tier | release | data_type | other, from the words."""
    words = set(re.findall(r"[a-z0-9]+", name.lower()))
    if re.search(r"\bv\d+(\.\d+)*\b", name.lower()):
        return "release"
    for kind, keys in _KIND_WORDS:
        if words & keys:
            return kind
    return "other"


_CLAUSE_TOKEN = re.compile(r'\\?"((?:[^"\\]|\\(?!"))*)\\?"|\(|\)|\bAND\b|\bOR\b|\bNOT\b')


def clause_roles(clause: str) -> dict[str, str]:
    """Role of each quoted term in a FOR clause: ``standalone`` (matches on its own),
    ``gated`` (needs the AND-ed group), ``gate`` (the last operand of an AND), ``exclude``
    (after NOT). Keys are case-folded terms."""
    toks = [(m.group(1) is not None and "phrase" or m.group(0),
             " ".join((m.group(1) or "").split())) for m in _CLAUSE_TOKEN.finditer(clause or "")]
    pos = 0

    def group():
        nonlocal pos
        ors, nots = [[operand()]], []
        while pos < len(toks) and toks[pos][0] in ("AND", "OR", "NOT"):
            op = toks[pos][0]
            pos += 1
            node = operand()
            (ors[-1].append(node) if op == "AND" else ors.append([node]) if op == "OR"
             else nots.append(node))
        return ("g", ors, nots)

    def operand():
        nonlocal pos
        kind, val = toks[pos]
        pos += 1
        if kind == "phrase":
            return ("p", val)
        node = group()
        pos += 1  # ")"
        return node

    roles: dict[str, str] = {}

    def walk(node, role):
        if node[0] == "p":
            roles.setdefault(node[1].casefold(), role)
            return
        _, ors, nots = node
        for conj in ors:
            if len(conj) == 1:
                walk(conj[0], role)
            else:
                for n in conj[:-1]:
                    walk(n, "gated" if role == "standalone" else role)
                walk(conj[-1], "gate" if role == "standalone" else role)
        for n in nots:
            walk(n, "exclude")

    try:
        walk(group(), "standalone")
    except (IndexError, ValueError):
        return {}
    return roles


#: Reason for a non-specific data name missing from the final query under the broad query.
NOT_SEARCHED_BY_DESIGN = ("not searched by design: the query is broad on the program name "
                          "and the full-text judge decides")


def program_datasets(rmap: ResourceMap | None, *, safe: list[str] | None = None,
                     flags: list[str] | None = None, built: bool = True,
                     clause: str | None = None):
    """The output list: the program and every data name examined, with its status in the
    final query (``standalone``, ``gated``, ``gate`` or ``not_used`` with the reason); statuses are
    None when no query was built (research record). A non-specific data name missing from
    the query is "not searched by design"; each program name's own role is reported in
    ``program_name_status``."""
    from dataset_agent.domain.models import ProgramDataset, ProgramDatasets

    if rmap is None:
        return None
    safe_f = {s.casefold() for s in (safe or [])}
    flag_f = {f.casefold() for f in (flags or [])}
    gated_f: set[str] = set()
    if clause is not None:
        roles = clause_roles(clause)
        safe_f = {k for k, r in roles.items() if r == "standalone"}
        flag_f = {k for k, r in roles.items() if r == "gate"}
        gated_f = {k for k, r in roles.items() if r == "gated"}
    specific_f = {d.casefold() for d in rmap.specific_data_names}
    rows = {r["name"].casefold(): dict(r) for r in rmap.found_names}
    for d in rmap.data_resource_names:  # names that entered without a record
        rows.setdefault(d.casefold(), {"name": d, "kept": True,
                                       "source": rmap.name_sources.get(d, ""),
                                       "hits_with_program": rmap.cooccurrence.get(d),
                                       "hits_alone": rmap.data_name_hits.get(d)})
    out = []
    for key, r in rows.items():
        status, reason = None, ""
        if built:
            if key in safe_f:
                status = "standalone"
            elif key in flag_f:
                status = "gate"
            elif key in gated_f:
                status = "gated"
            else:
                status = "not_used"
                reason = r.get("reason") or (
                    NOT_SEARCHED_BY_DESIGN if key not in specific_f
                    else "kept by the map, not in the final query (screened, or dropped "
                         "after a probe)")
        elif not r.get("kept"):
            reason = r.get("reason") or ""
        out.append(ProgramDataset(
            name=r["name"], kind=data_kind(r["name"]),
            source=r.get("source") or rmap.name_sources.get(r["name"], ""),
            where=r.get("where") or "", hits_alone=r.get("hits_alone"),
            hits_with_program=r.get("hits_with_program"), status=status, reason=reason))
    program_status: dict[str, str] = {}
    if built:
        for p in rmap.program_names:
            k = p.casefold()
            program_status[p] = ("standalone" if k in safe_f else "gate" if k in flag_f
                                 else "gated" if k in gated_f else "not_used")
    return ProgramDatasets(program_names=list(rmap.program_names),
                           former_or_parent_names=list(rmap.former_or_parent_names),
                           program_name_status=program_status, datasets=out)


DATA_WORDING_NOTE = ("informational: the data-wording clause (data phrases and specific data "
                     "names alone, program names only together with a data name); not the "
                     "released query")


def data_wording_clause(rmap: ResourceMap | None, *, max_hits: int = 500_000) -> str | None:
    """The data-wording alternative for a split dataset (the round-2 shape), reported next
    to the released broad query: non-acronym data phrases (``<program> data``) and specific
    data names stand alone; program names are AND-ed with the data resource names (not a
    bare short acronym, at most *max_hits* hits). With no such data name the program names
    are left out. None without a map, or when nothing is left."""
    from dataset_agent.adapters.query_builder import build_for_clause
    from dataset_agent.adapters.query_heuristics import is_short_acronym
    from dataset_agent.adapters.scope_check import acronym_form, generated_data_phrases

    if rmap is None:
        return None
    programs = list(rmap.program_names)
    phrases = [p for p in generated_data_phrases(programs) if not acronym_form(p)]
    specific = [d for d in rmap.specific_data_names if not is_short_acronym(d)
                and (rmap.data_name_hits.get(d) or 0) <= max_hits]
    gate = [d for d in rmap.data_resource_names if not is_short_acronym(d)
            and (rmap.data_name_hits.get(d) or 0) <= max_hits]
    safe = list(dict.fromkeys([*phrases, *specific]))
    # With no gate, program names are left out (never OR-ed bare): the alternative is the
    # safe data phrases only, and with nothing at all there is no alternative.
    return build_for_clause(safe=safe, variant="V4", risky=programs if gate else [],
                            flag_terms=gate) or None


def program_datasets_summary(pd) -> str:
    """Human-readable lines for a run summary ("" when not split)."""
    if pd is None:
        return ""
    roles = getattr(pd, "program_name_status", None) or {}
    lines = [f"Program: {', '.join(f'{p} [{roles[p]}]' if p in roles else p for p in pd.program_names)}"]
    if pd.former_or_parent_names:
        lines.append(f"  former or parent names: {', '.join(pd.former_or_parent_names)}")
    lines.append(f"Datasets of the program ({len(pd.datasets)}):")
    for d in pd.datasets:
        hits = f"{d.hits_alone if d.hits_alone is not None else '?'} alone, " \
               f"{d.hits_with_program if d.hits_with_program is not None else '?'} with the program"
        status = f" [{d.status}]" if d.status else ""
        why = f" — {d.reason}" if d.reason else ""
        lines.append(f"  - {d.name} ({d.kind}; {d.source}"
                     f"{': ' + d.where if d.where else ''}; {hits}){status}{why}")
    if getattr(pd, "data_wording_clause", None):
        lines.append(f"Data-wording alternative ({pd.data_wording_note}): "
                     f"{pd.data_wording_count if pd.data_wording_count is not None else '?'} "
                     f"papers: {pd.data_wording_clause}")
    return "\n".join(lines)


def map_options(settings) -> dict:
    """The map keyword arguments taken from Settings."""
    return {"min_cooccurrence": settings.resource_map_min_cooccurrence,
            "max_share_before": settings.resource_map_max_share_before_release,
            "specific_min_share": settings.resource_map_specific_min_share,
            "max_hits": settings.optimize_flag_max_hits,
            "discover": settings.resource_map_discovery}


def map_for_record(agent, record, *, count: Callable[[str], int | None],
                   min_cooccurrence: int = 1, pages: str | None = None,
                   count_where: Callable[[str, str], int | None] | None = None,
                   max_share_before: float = 0.05, specific_min_share: float = 0.5,
                   max_hits: int = 500_000, sources: list[str] | None = None,
                   discover: bool = False, search=None, fetch=None,
                   parts: list | None = None) -> ResourceMap | None:
    """Research the resource map of a DatasetRecord (pages fetched unless given). With
    *discover*, a split dataset also gets the web discovery step (``discover_from_web``).
    Returns None for a firm "no split"; raises :class:`ResourceMapUndecided` when there
    are no pages or the LLM call failed."""
    name = record.main_dataset_name
    if pages is None:
        parts = []
        pages, sources = gather_pages_with_sources(name, record.home_url, search=search,
                                                   fetch=fetch, parts_out=parts)
    elif parts is None:
        parts = [("pages given", pages)]
    if not (pages or "").strip():
        raise ResourceMapUndecided("no pages could be read")
    names = list(dict.fromkeys([name, *record.dataset_names, *record.flag_terms]))
    raw = propose(agent, dataset_name=name, description=record.description, names=names,
                  pages=pages, parts=parts)
    if not isinstance(raw.get("names"), list):
        raise ResourceMapUndecided("the map LLM call failed or returned no answer")
    m = build_map(raw, dataset_name=name, pages=pages, count=count,
                  min_cooccurrence=min_cooccurrence, count_where=count_where,
                  max_share_before=max_share_before, specific_min_share=specific_min_share,
                  max_hits=max_hits)
    if m is not None:
        m.page_sources = list(sources or [])
        for r in m.found_names:
            if r.get("source") == "pages" and not r.get("where"):
                r["where"] = locate(r["name"], parts)
        if discover:
            added = discover_from_web(agent, m, dataset_name=name, description=record.description,
                                      count=count, min_cooccurrence=min_cooccurrence,
                                      specific_min_share=specific_min_share, max_hits=max_hits,
                                      search=search, fetch=fetch)
            logger.info("resource map: web discovery added %s", added)
    logger.info("resource map for %r: %s", name, m.model_dump() if m else "none")
    return m

"""Former and successor names of a dataset and its organizations.

Data collections get renamed and merged: the Plate Boundary Observatory became part of the
Network of the Americas in 2018, and UNAVCO merged with IRIS into EarthScope Consortium in
2023. Papers in the search window cite whichever name was current when they were written,
so a query with only today's names misses the older papers (and the reverse).

One web search per leading name plus one structured LLM call name the history; phrases
such as "formerly the Plate Boundary Observatory" in the web results are candidates too,
so an LLM that returns nothing does not end the search. The caller screens the names with
``name_grounding.screen_candidates`` and keeps only those Dimensions finds.
"""

from __future__ import annotations

import re

HISTORY_TOOL: dict = {
    "name": "emit_name_history",
    "description": "Former, successor and merged names of a data collection and its organizations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "dataset_names": {
                "type": "array",
                "description": "Other names the same data collection has had or now has.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "relation": {
                            "type": "string",
                            "enum": ["former", "successor", "merged_into", "merged_from"],
                        },
                        "year": {"type": "integer"},
                    },
                    "required": ["name", "relation"],
                },
            },
            "organizations": {
                "type": "array",
                "description": "Former or successor names of the listed organizations.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "of": {"type": "string", "description": "The listed organization it renames."},
                        "relation": {
                            "type": "string",
                            "enum": ["former", "successor", "merged_into", "merged_from"],
                        },
                        "year": {"type": "integer"},
                    },
                    "required": ["name", "of", "relation"],
                },
            },
        },
        "required": ["dataset_names", "organizations"],
    },
}

_NAME = r"(?:the\s+)?([A-Z][\w'&-]*(?:\s+(?:[A-Z][\w'&-]*|of|the|and|for|on|in|de)){1,7})"
_HISTORY_PHRASES = re.compile(
    r"(?:formerly(?:\s+(?:known\s+as|called|named))?|previously\s+(?:known\s+as|called|named)"
    r"|originally\s+(?:known\s+as|called|named)|renamed\s+(?:to|as|from)"
    r"|successor\s+(?:to|of)|merged\s+(?:into|with))\s*,?\s+" + _NAME
)
_LINK_TAIL = re.compile(r"(?:\s+(?:of|the|and|for|on|in|de))+$")


def history_phrases(text: str, anchors: list[str] | None = None) -> list[str]:
    """Names introduced by "formerly", "renamed to", "successor to" and similar phrases.

    With *anchors*, only web results (blocks starting ``"- "``) that mention one of them
    count: a search for "Network of the Americas" also returns a broadcaster's history,
    whose "merged with United Paramount Theatres" says nothing about this collection."""
    keys = [a.casefold() for a in (anchors or []) if a and a.strip()]
    blocks = re.split(r"\n(?=- )", text or "")
    out: list[str] = []
    for block in blocks:
        if keys and not any(k in block.casefold() for k in keys):
            continue
        for m in _HISTORY_PHRASES.finditer(block):
            name = _LINK_TAIL.sub("", m.group(1)).strip()
            if len(name.split()) >= 2 and name.casefold() not in {n.casefold() for n in out}:
                out.append(name)
    return out


def history_prompt(
    dataset_name: str,
    leading_names: list[tuple[str, int]],
    flag_terms: list[str],
    years: str,
    web_results: str,
) -> str:
    names = "\n".join(f"  - {a!r}: {n} papers" for a, n in leading_names) or "  (none)"
    flags = "\n".join(f"  - {f}" for f in flag_terms[:10]) or "  (none)"
    return f"""Data collection {dataset_name!r} is searched in the literature of {years} by these
names (exact-phrase hits in papers):
{names}

Organizations used as context terms:
{flags}

=== WEB RESULTS ===
{(web_results or "(none)")[:6000]}

=== YOUR TASK ===
Call emit_name_history with the renames and merges of this collection and of the listed
organizations, so papers written under any of its names are found:
- dataset_names: other proper names of the SAME data collection: what it was called
  before (relation "former"), what it is called now ("successor"), or a collection it was
  merged into or absorbed ("merged_into"/"merged_from").
- organizations: former or successor names of a LISTED organization; set "of" to the
  listed name it renames (for example an organization that merged into a consortium).
Rules: only renames and merges the web results or well-established history support.
Networks or archives that were combined into this collection count ("merged_from"), as
does a larger collection it now belongs to ("merged_into"). Never a sibling product that
is still separate, a parent agency or a funder; names as authors write them in papers;
empty lists if there is no history."""

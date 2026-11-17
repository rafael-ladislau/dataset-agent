"""Former and successor names of a dataset.

Data collections get renamed and merged: the Plate Boundary Observatory became part of the
Network of the Americas in 2018, and UNAVCO merged with IRIS into EarthScope Consortium in
2023. Papers in the search window cite whichever name was current when they were written,
so a query with only today's names misses the older papers (and the reverse).

One web search per leading name plus one structured LLM call name the history; the caller
screens the names with ``name_grounding.screen_candidates`` and keeps only those Dimensions
finds.
"""

from __future__ import annotations

HISTORY_TOOL: dict = {
    "name": "emit_name_history",
    "description": "Former, successor and merged names of a data collection.",
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
        },
        "required": ["dataset_names"],
    },
}

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
Call emit_name_history with the renames and merges of this collection, so papers written
under any of its names are found:
- dataset_names: other proper names of the SAME data collection: what it was called
  before (relation "former"), what it is called now ("successor"), or a collection it was
  merged into or absorbed ("merged_into"/"merged_from").
Rules: only renames and merges the web results or well-established history support.
Networks or archives that were combined into this collection count ("merged_from"), as
does a larger collection it now belongs to ("merged_into"). Never a sibling product that
is still separate, a parent agency or a funder; names as authors write them in papers;
empty lists if there is no history."""

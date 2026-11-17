"""Detect LLM agent artifacts in string lists (aliases, organizations, etc.)."""

from __future__ import annotations

import json
import re

_TOOL_NAME_RE = re.compile(
    r'["\']name["\']\s*:\s*["\'](?:web_search|make_request)["\']',
    re.IGNORECASE,
)

# Patterns for JSON fragments from tool calls
_JSON_FRAGMENT_PATTERNS = (
    r'^["\']?parameters["\']?\s*:',
    r'^["\']?query["\']?\s*:',
    r'^["\']?name["\']?\s*:',
    r'^["\']?url["\']?\s*:',
    r'^\{["\']',
    r'^}+$',
)
_JSON_FRAGMENT_RE = re.compile("|".join(_JSON_FRAGMENT_PATTERNS), re.IGNORECASE)

# Patterns for numbered/bulleted list items that weren't properly parsed
_NUMBERED_LIST_RE = re.compile(r'^\d+\.\s*["\']')


def is_llm_list_entry_junk(text: str) -> bool:
    """
    Entries that are not real names: tool-call JSON, prose instructions,
    prompt echoes, etc.
    """
    t = text.strip()
    if not t:
        return True
    low = t.lower()

    # JSON fragments from tool calls
    if _JSON_FRAGMENT_RE.match(t):
        return True

    # Numbered list items with quotes (LLM formatting artifacts)
    if _NUMBERED_LIST_RE.match(t):
        return True
    
    # Malformed Python list strings (LLM tried to return a list as string)
    # e.g., '["bls", "ors", Apa: Bureau of Labor Statistics...]'
    if t.startswith('["') or t.startswith("['"):
        return True

    if _TOOL_NAME_RE.search(t):
        return True
    if '"parameters"' in low and '"query"' in low:
        return True
    if '"query"' in low and ('how to cite' in low or 'dataset' in low):
        return True
    if t.startswith("{") and ("web_search" in low or "make_request" in low):
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and obj.get("name") in ("web_search", "make_request"):
                return True
        except json.JSONDecodeError:
            return True

    if t.startswith('"') and t.endswith('"'):
        try:
            inner = json.loads(t)
            if isinstance(inner, str) and inner.strip().startswith("{"):
                return is_llm_list_entry_junk(inner)
        except json.JSONDecodeError:
            pass

    # Meta-commentary from LLM
    meta_markers = (
        "to answer the prompt",
        "to answer this question",
        "to answer the question",
        "we need to call",
        "we will format the response",
        "here's a possible search query",
        "the search query should include",
        "parameters for that function",
        "dictionary with the name of the function",
        "i will use the",
        "function to search for",
        "information about how the dataset is cited",
        "based on the web search",
        "based on my search",
        "based on the search",
        "here is the list",
        "here are the",
        "the list only includes",
        "note: the list",
        "note: i ",
        "note: this ",
        "note that the",
        "note that citation",
        "also note that",
        "as per the original",
        "according to the search",
        "i found the following",
        "the following organizations",
        "citation styles may vary",
        "depending on the field",
        "follow apa and mla",
        "the above citations",
    )
    if any(m in low for m in meta_markers):
        return True
    
    # Citation format artifacts (APA:, MLA:, DOI:, Retrieved From)
    citation_markers = (
        "apa:",
        "mla:",
        "doi:",
        "retrieved from",
        "citation form",
    )
    if any(m in low for m in citation_markers):
        return True

    if "`web_search`" in t or "`make_request`" in t:
        return True

    if "use the web_search tool" in low and "search for" in low:
        return True

    if "dataset organization creator publisher funder" in low:
        return True

    # Entries that are just punctuation or brackets
    if re.match(r'^[\[\]{}(),.:;"\'\s]+$', t):
        return True

    return False

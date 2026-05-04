"""Shared tool implementations for the agent (web search + HTTP GET)."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


def web_search(query: str) -> str:
    """Search the public web via DuckDuckGo and return formatted results."""
    q = (query or "").strip()
    if not q:
        return (
            "web_search: the model sent an empty or invalid query. "
            "Call again with a single non-empty search string in the 'query' argument."
        )
    qprev = (q[:120] + "…") if len(q) > 120 else q
    logger.info("Tool web_search: query=%r", qprev)
    try:
        from ddgs import DDGS
    except ImportError:
        return "web_search: ddgs not installed (pip install ddgs)"
    try:
        hits = list(DDGS().text(q, max_results=6))
        if not hits:
            logger.info("Tool web_search: 0 results")
            return "web_search: no results"
        logger.info("Tool web_search: %s results", len(hits))
        parts = []
        for h in hits:
            body = (h.get("body") or "")[:400]
            title = h.get("title") or ""
            href = h.get("href") or ""
            parts.append(f"- {title}\n  {href}\n  {body}")
        return "\n".join(parts)
    except Exception as e:
        logger.warning("web_search failed: %s", e)
        return f"web_search error: {e}"


def make_request(url: str) -> str:
    """HTTP GET a URL and return status code + short text preview."""
    u = (url or "").strip()
    if not u:
        return "request error: empty URL"
    logger.info("Tool make_request: GET %s", u[:200] + ("…" if len(u) > 200 else ""))
    try:
        r = httpx.get(u, timeout=30.0, follow_redirects=True)
        preview = (r.text or "")[:800]
        logger.info(
            "Tool make_request: status=%s bytes=%s",
            r.status_code,
            len(r.content),
        )
        return f"status_code={r.status_code} content_length={len(r.content)} preview={preview!r}"
    except Exception as e:
        return f"request error: {e}"


# Anthropic tool definitions (JSON Schema for the messages API)
TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the public web for up-to-date information about a dataset, "
            "organization, or topic. Use this to find official websites, dataset "
            "descriptions, publishers, and alternative names. "
            "Example: query='Current Population Survey official website census bureau'. "
            "Always use a specific, non-empty search string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Non-empty search query string. Be specific: include the "
                        "dataset name plus context terms (e.g. 'dataset creator "
                        "publisher organization')."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "make_request",
        "description": (
            "HTTP GET a URL and return the HTTP status code plus a short text preview. "
            "Use this to validate that a URL is live (status 200) and inspect page "
            "content for data download links, schema definitions, or documentation. "
            "Only call with full http(s) URLs. Returns an error string on network failure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "Full http(s) URL to fetch with GET, e.g. "
                        "'https://www.census.gov/programs-surveys/cps.html'."
                    ),
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
]

# ---------------------------------------------------------------------------
# Result / emit tools — used with tool_choice={"type":"tool"} to get
# guaranteed structured output without regex parsing.
# ---------------------------------------------------------------------------

EMIT_DATASET_NAMES: dict = {
    "name": "emit_dataset_names",
    "description": (
        "Return the cleaned list of dataset name aliases suitable for literature "
        "string search. Call this tool exactly once with the final list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "dataset_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Cleaned dataset name aliases (product titles and acronyms only).",
            },
        },
        "required": ["dataset_names"],
        "additionalProperties": False,
    },
}

EMIT_FLAG_TERMS: dict = {
    "name": "emit_flag_terms",
    "description": (
        "Return the cleaned list of organization / sponsor flag terms for literature "
        "search. Call this tool exactly once with the final list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "flag_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Organization and sponsor terms that identify the dataset publisher.",
            },
        },
        "required": ["flag_terms"],
        "additionalProperties": False,
    },
}

EMIT_SUBDATASET_ALIASES: dict = {
    "name": "emit_subdataset_aliases",
    "description": (
        "Split dataset aliases into those that refer to this dataset product versus "
        "versions/sub-products that should be separate records. Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keep_aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Aliases that still refer to this dataset (max 40 strings).",
            },
            "remove_aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Aliases to drop (different cohort/version/sub-product).",
            },
        },
        "required": ["keep_aliases", "remove_aliases"],
        "additionalProperties": False,
    },
}

EMIT_RELEVANCE_SCORE: dict = {
    "name": "emit_relevance_score",
    "description": (
        "Return the relevance assessment scores for the publication. "
        "Call this tool exactly once with the evaluation results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mention_score": {
                "type": "integer",
                "description": "0–10: how clearly the title/abstract mentions this specific dataset.",
            },
            "context_score": {
                "type": "integer",
                "description": "0–10: how well the publication's domain matches this dataset's purpose.",
            },
            "mentioned_term": {
                "type": "string",
                "description": "The best-matched alias or term found, or empty string.",
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of the scores (max 10 words).",
            },
        },
        "required": ["mention_score", "context_score", "mentioned_term", "reason"],
        "additionalProperties": False,
    },
}

EMIT_TERMS_EVALUATION: dict = {
    "name": "emit_terms_evaluation",
    "description": (
        "Return the effectiveness evaluation of the dataset search terms. "
        "Call this tool exactly once with the full evaluation results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_effective": {
                "type": "boolean",
                "description": "True if current terms are adequate for literature search.",
            },
            "dataset_names_score": {
                "type": "integer",
                "description": "0–10 quality score for dataset name aliases.",
            },
            "flag_terms_score": {
                "type": "integer",
                "description": "0–10 quality score for organization flag terms.",
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of identified issues with current terms.",
            },
            "suggested_dataset_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New dataset name aliases to add (not already in current terms).",
            },
            "suggested_flag_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New organization terms to add (not already in current terms).",
            },
            "suggested_exclude_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Phrases to exclude from future searches to reduce off-domain noise.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the evaluation (max 50 words).",
            },
        },
        "required": [
            "is_effective",
            "dataset_names_score",
            "flag_terms_score",
            "issues",
            "suggested_dataset_names",
            "suggested_flag_terms",
            "suggested_exclude_terms",
            "reasoning",
        ],
        "additionalProperties": False,
    },
}

# Dispatch table mapping tool name → callable (only applies to action tools)
TOOL_DISPATCH: dict[str, callable] = {
    "web_search": lambda args: web_search(args.get("query", "")),
    "make_request": lambda args: make_request(args.get("url", "")),
}

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

EMIT_FP_ANALYSIS: dict = {
    "name": "emit_fp_analysis",
    "description": (
        "Return false-positive domains for a risky alias in scientific literature. "
        "Call exactly once with structured domains and indicator keywords per domain."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "domains": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "domain_name": {
                            "type": "string",
                            "description": "Short FP domain label (e.g. biochemistry).",
                        },
                        "indicator_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Terms that signal this FP domain in titles/abstracts (max 20).",
                        },
                    },
                    "required": ["domain_name", "indicator_terms"],
                    "additionalProperties": False,
                },
                "description": "Non-empty list of FP collision domains for the alias.",
            },
        },
        "required": ["domains"],
        "additionalProperties": False,
    },
}

EMIT_FLAG_TERMS_FOR_ALIAS: dict = {
    "name": "emit_flag_terms_for_alias",
    "description": (
        "Return context flag terms that indicate the paper is about the target dataset "
        "when a risky alias appears. Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "flag_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Domain-positive terms (max 20 strings).",
            },
        },
        "required": ["flag_terms"],
        "additionalProperties": False,
    },
}

EMIT_DOMAIN_POSITIVE_KEYWORDS: dict = {
    "name": "emit_domain_positive_keywords",
    "description": (
        "Return short override keywords: if they appear in a title, treat as dataset-relevant. "
        "Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Override keywords (max 15).",
            },
        },
        "required": ["keywords"],
        "additionalProperties": False,
    },
}

EMIT_TITLE_RELEVANCE: dict = {
    "name": "emit_title_relevance",
    "description": (
        "Signal 3: score how relevant a single publication title is to the dataset. "
        "Call exactly once per title."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mention_score": {
                "type": "integer",
                "description": "0–10: clear mention of this dataset vs unrelated use of terms.",
            },
            "context_score": {
                "type": "integer",
                "description": "0–10: domain match (uses/analyzes/references this dataset's data).",
            },
            "mentioned_term": {
                "type": "string",
                "description": "Best matching term or empty string.",
            },
            "reason": {
                "type": "string",
                "description": "Brief justification (max 25 words).",
            },
        },
        "required": ["mention_score", "context_score", "mentioned_term", "reason"],
        "additionalProperties": False,
    },
}

EMIT_QUERY_VARIANTS: dict = {
    "name": "emit_query_variants",
    "description": (
        "Summarize tested Dimensions query variants (counts, FP rates). Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "expected_count": {"type": "integer"},
                        "fp_rate_pct": {"type": "number"},
                        "notes": {"type": "string"},
                    },
                    "required": ["label", "expected_count", "fp_rate_pct", "notes"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["variants"],
        "additionalProperties": False,
    },
}

EMIT_ALIAS_MENTION_CLASSIFICATION: dict = {
    "name": "emit_alias_mention_classification",
    "description": (
        "Decide whether an alias mention in an abstract refers to the target dataset "
        "or a false positive. Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_genuine_dataset_reference": {
                "type": "boolean",
                "description": "True if the mention is about the target dataset product.",
            },
            "confidence": {
                "type": "integer",
                "description": "0–10 self-rated confidence in the classification.",
            },
            "reason": {
                "type": "string",
                "description": "One short sentence explaining the decision.",
            },
        },
        "required": ["is_genuine_dataset_reference", "confidence", "reason"],
        "additionalProperties": False,
    },
}

EMIT_EXCLUDE_TERMS: dict = {
    "name": "emit_exclude_terms",
    "description": (
        "Return NOT-clause candidate phrases derived from confirmed false-positive contexts. "
        "Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "exclude_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Phrases for Dimensions post-filter or DSL NOT (max 25).",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief summary of how terms were chosen (max 40 words).",
            },
        },
        "required": ["exclude_terms", "reasoning"],
        "additionalProperties": False,
    },
}

EMIT_WEB_ALIAS_MEANINGS: dict = {
    "name": "emit_web_alias_meanings",
    "description": (
        "After reviewing web search hits, list alternative meanings of an ambiguous term "
        "as domains with candidate exclude terms. Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "domains": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "domain_name": {"type": "string"},
                        "candidate_exclude_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["domain_name", "candidate_exclude_terms"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["domains"],
        "additionalProperties": False,
    },
}

EMIT_SUB_PRODUCT_NAMES: dict = {
    "name": "emit_sub_product_names",
    "description": (
        "Return distinct sub-product or module names found on the dataset homepage "
        "(e.g. QWI, LODES under LEHD). Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short product/module names only (max 20).",
            },
        },
        "required": ["names"],
        "additionalProperties": False,
    },
}

EMIT_MULTILINGUAL_ALIASES: dict = {
    "name": "emit_multilingual_aliases",
    "description": (
        "Return alternate spellings or translations of the dataset title for search "
        "(accented/unaccented, other languages). Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional citation forms (max 25).",
            },
        },
        "required": ["aliases"],
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

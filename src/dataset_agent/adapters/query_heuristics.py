"""Dataset-specific query heuristics (spec §9, gap doc §2.8).

Used during query optimization Phase 1/2 before or alongside ``build_for_clause``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Sequence

from dataset_agent.adapters.text_processing import dedupe_strings_ci_preserve_order
from dataset_agent.adapters.tools import (
    EMIT_MULTILINGUAL_ALIASES,
    EMIT_SUB_PRODUCT_NAMES,
    make_request,
)

if TYPE_CHECKING:
    from dataset_agent.domain.ports import AgentPort, DimensionsDslPort

logger = logging.getLogger(__name__)

# Spec / gap: aliases shorter than 5 characters must use hybrid (AND flags), not bare OR.
SHORT_ACRONYM_MAX_LEN = 4

_SUFFIX_CANDIDATES: tuple[str, ...] = ("dataset", "survey", "program", "data", "study")

# Minimal English function-word set: if every token is in this set and there are <4 tokens,
# the dataset title is likely a rhetorical/common phrase (e.g. "All of Us").
_COMMON_ENGLISH_WORDS: frozenset[str] = frozenset(
    """
    a an the and or of to in for on at by from as is are was were been being
    be have has had do does did will would could should may might must can need
    this that these those it we you he she they i me my our your their us one
    all any both each few more most other some such no nor not only own same so
    than too very just also now here there when where why how who what which
    then once ever never every into onto upon out off over under again further
    once twice
    """.split()
)

# Extra rhetorical collocations (All of Us style) from dataset_query_agent_spec.md.
_RHETORICAL_FIXED_EXTRAS: tuple[str, ...] = (
    "affects all of us",
    "benefits all of us",
    "for all of us",
    "concerns all of us",
    "impacts all of us",
    "all of us need",
    "all of us can",
    "all of us should",
    "teach all of us",
    "among all of us",
    "all of us together",
    "matters to all of us",
)


def is_short_acronym(alias: str) -> bool:
    """True if *alias* must not appear as a bare OR term (length ≤ 4 characters)."""
    s = (alias or "").strip()
    if not s:
        return False
    return len(s) <= SHORT_ACRONYM_MAX_LEN


def apply_short_acronym_heuristic(
    safe: Sequence[str],
    risky: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Move short acronyms from *safe* into *risky* so V1 never ORs them alone."""
    rk: list[str] = [x.strip() for x in (risky or []) if isinstance(x, str) and x.strip()]
    rk_lower = {x.lower() for x in rk}
    st: list[str] = []
    for x in safe:
        if not isinstance(x, str) or not x.strip():
            continue
        s = x.strip()
        if is_short_acronym(s):
            if s.lower() not in rk_lower:
                rk.append(s)
                rk_lower.add(s.lower())
        else:
            st.append(s)
    return dedupe_strings_ci_preserve_order(st), dedupe_strings_ci_preserve_order(rk)


def detect_rhetorical_name(dataset_name: str) -> bool:
    """Heuristic: very short titles made only of ultra-common English words."""
    raw = (dataset_name or "").strip()
    if not raw:
        return False
    words = re.findall(r"[A-Za-z]+", raw)
    if not words or len(words) >= 4:
        return False
    lowered = [w.lower() for w in words]
    if any(len(w) > 10 for w in lowered):
        return False
    return all(w in _COMMON_ENGLISH_WORDS for w in lowered)


def rhetorical_exclusions(dataset_name: str) -> list[str]:
    """Suggested NOT phrases when :func:`detect_rhetorical_name` is true."""
    core = " ".join(re.findall(r"[A-Za-z]+", dataset_name)).strip().lower()
    if not core:
        return list(_RHETORICAL_FIXED_EXTRAS)
    dynamic = [
        f"affects {core}",
        f"benefits {core}",
        f"for {core}",
        f"concerns {core}",
        f"impacts {core}",
        f"among {core}",
        f"matters to {core}",
    ]
    merged = dedupe_strings_ci_preserve_order(list(_RHETORICAL_FIXED_EXTRAS) + dynamic)
    return merged


def platform_exclude_suggestions(dataset_url: str | None) -> list[str]:
    """Extra NOT candidates when the dataset lives on a noisy consumer-tech domain."""
    if not dataset_url or not isinstance(dataset_url, str):
        return []
    u = dataset_url.lower()
    if "linkedin.com" in u:
        return [
            "social media marketing",
            "sentiment analysis",
            "social network analysis",
            "linkedin profile",
            "recruitment analytics",
        ]
    if "twitter.com" in u or "x.com" in u:
        return ["tweet", "twitter api", "social media", "hashtag"]
    if "facebook.com" in u:
        return ["facebook users", "social network", "privacy concerns"]
    return []


def detect_sub_products(url: str | None, agent: AgentPort) -> list[str]:
    """LLM over homepage HTML preview: short names of sub-products (e.g. QWI, LODES)."""
    if not url or not str(url).strip().startswith(("http://", "https://")):
        return []
    preview = make_request(str(url).strip())
    prompt = f"""Dataset homepage URL: {url!r}

HTTP fetch preview (truncated):
{preview[:6000]}

List **distinct data product or module names** hosted under the same program
(short tokens like QWI, LODES, PUMS — not generic words like \"data\" or \"documentation\").

Use emit_sub_product_names exactly once. If the page is unusable or there are no sub-products, return an empty list."""
    try:
        structured = agent.get_structured(prompt, EMIT_SUB_PRODUCT_NAMES)
    except Exception as e:
        logger.warning("detect_sub_products failed: %s", e)
        return []
    names = structured.get("names")
    if not isinstance(names, list):
        return []
    return [str(x).strip() for x in names if isinstance(x, str) and x.strip()][:20]


def generate_multilingual_aliases(dataset_name: str, agent: AgentPort) -> list[str]:
    """LLM: alternate spellings / translations useful for Dimensions search."""
    prompt = f"""Dataset title: {dataset_name!r}.

Produce alternate citation forms: unaccented variants, common translations,
and official non-English titles if they exist.

Use emit_multilingual_aliases exactly once."""
    try:
        raw = agent.get_structured(prompt, EMIT_MULTILINGUAL_ALIASES)
    except Exception as e:
        logger.warning("generate_multilingual_aliases failed: %s", e)
        return []
    aliases = raw.get("aliases")
    if not isinstance(aliases, list):
        return []
    out = [str(x).strip() for x in aliases if isinstance(x, str) and x.strip()][:25]
    return dedupe_strings_ci_preserve_order(out)


def strip_accents(text: str) -> str:
    """ASCII fold for simple multilingual dedupe (helper for callers)."""
    if not text:
        return ""
    nk = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nk if not unicodedata.combining(ch))


async def check_suffix_necessity(
    dsl_port: DimensionsDslPort,
    alias: str,
    *,
    search_in: str = "full_data",
    ratio_threshold: float = 100.0,
    bare_count: int | None = None,
) -> str | None:
    """If count(bare)/count(bare + suffix) > *ratio_threshold*, return best suffixed form."""
    from dataset_agent.adapters.dimensions_dsl import run_alias_count

    bare = (alias or "").strip()
    if not bare or len(bare) > 120:
        return None
    if bare_count is not None:
        c_bare = bare_count
    else:
        c_bare = await run_alias_count(dsl_port, bare, search_in=search_in)
    if c_bare <= 0:
        return None

    best: str | None = None
    best_ratio = 0.0
    for suf in _SUFFIX_CANDIDATES:
        suffixed = f"{bare} {suf}".strip()
        c_s = await run_alias_count(dsl_port, suffixed, search_in=search_in)
        if c_s <= 0:
            continue
        ratio = float(c_bare) / float(c_s)
        if ratio > ratio_threshold and ratio > best_ratio:
            best_ratio = ratio
            best = suffixed
    return best

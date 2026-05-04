"""False-positive identification helpers (dataset_query_agent_spec §6, gap doc §2.5)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dataset_agent.adapters.dimensions_dsl import (
    build_for_clause_count_dsl,
    parse_total_count,
)
from dataset_agent.adapters.tools import (
    EMIT_ALIAS_MENTION_CLASSIFICATION,
    EMIT_DOMAIN_POSITIVE_KEYWORDS,
    EMIT_EXCLUDE_TERMS,
    EMIT_FLAG_TERMS_FOR_ALIAS,
    EMIT_FP_ANALYSIS,
    EMIT_TITLE_RELEVANCE,
    EMIT_WEB_ALIAS_MEANINGS,
    web_search,
)

if TYPE_CHECKING:
    from dataset_agent.domain.ports import AgentPort, DimensionsDslPort

logger = logging.getLogger(__name__)


def classify_title(title: str, fp_keywords: dict[str, list[str]]) -> dict[str, list[str]]:
    """Scan *title* (lowercased) for FP indicator keywords grouped by domain."""
    title_lower = (title or "").lower()
    matches: dict[str, list[str]] = {}
    for category, keywords in fp_keywords.items():
        if not isinstance(keywords, list):
            continue
        matched = [kw for kw in keywords if isinstance(kw, str) and kw.lower() in title_lower]
        if matched:
            matches[str(category)] = matched
    return matches


def scan_abstracts_for_aliases(
    publications: list[dict],
    aliases: list[str],
    *,
    snippet_before: int = 40,
    snippet_after: int = 60,
) -> list[dict[str, str]]:
    """Find substring hits for each alias in publication abstracts (case-insensitive)."""
    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    alias_list = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
    for pub in publications:
        if not isinstance(pub, dict):
            continue
        pid = str(pub.get("id") or "").strip()
        abstract = pub.get("abstract")
        if abstract is None and isinstance(pub.get("basics"), dict):
            abstract = pub["basics"].get("abstract")
        if not isinstance(abstract, str):
            abstract = ""
        low = abstract.lower()
        for alias in alias_list:
            al = alias.strip()
            pos = low.find(al.lower())
            if pos == -1:
                continue
            key = (pid, al.lower())
            if key in seen:
                continue
            seen.add(key)
            sn = abstract[max(0, pos - snippet_before) : pos + len(al) + snippet_after]
            hits.append(
                {
                    "publication_id": pid,
                    "matched_alias": al,
                    "snippet": sn.replace("\n", " ").strip(),
                }
            )
    return hits


async def run_scope_comparison(
    dsl_port: DimensionsDslPort,
    for_clause: str,
) -> float:
    """Return ``count(full_data) / max(1, count(title_abstract_only))`` for the same for-clause."""
    q_full = build_for_clause_count_dsl(for_clause, search_in="full_data", limit=1)
    q_ta = build_for_clause_count_dsl(for_clause, search_in="title_abstract_only", limit=1)
    r_full = await dsl_port.execute_dsl(q_full)
    r_ta = await dsl_port.execute_dsl(q_ta)
    c_full = max(0, parse_total_count(r_full))
    c_ta = max(0, parse_total_count(r_ta))
    return float(c_full) / float(max(1, c_ta))


def _fp_domains_prompt(dataset_name: str, risky_alias: str) -> str:
    return f"""The dataset is {dataset_name!r}. The ambiguous search alias is {risky_alias!r}.

List scientific-literature collision domains where this term/acronym usually means something
OTHER than this dataset. For each domain, give short indicator terms that appear in titles or
abstracts when the paper is off-topic (false positives for our dataset).

Use the emit_fp_analysis tool exactly once."""


def identify_fp_domains(agent: AgentPort, dataset_name: str, risky_alias: str) -> dict[str, list[str]]:
    """LLM: map FP domain label → indicator keywords for *risky_alias*."""
    prompt = _fp_domains_prompt(dataset_name, risky_alias)
    try:
        raw = agent.get_structured(prompt, EMIT_FP_ANALYSIS)
    except Exception as e:
        logger.warning("identify_fp_domains failed: %s", e)
        return {}
    domains = raw.get("domains")
    if not isinstance(domains, list):
        return {}
    out: dict[str, list[str]] = {}
    for row in domains:
        if not isinstance(row, dict):
            continue
        name = str(row.get("domain_name") or "").strip()
        inds = row.get("indicator_terms")
        if not name or not isinstance(inds, list):
            continue
        out[name] = [str(x).strip() for x in inds if isinstance(x, str) and x.strip()][:25]
    return out


def generate_flag_terms_for_alias(
    agent: AgentPort,
    dataset_name: str,
    risky_alias: str,
) -> list[str]:
    """LLM: domain-positive context terms when *risky_alias* truly refers to *dataset_name*."""
    prompt = f"""Dataset: {dataset_name!r}. Risky alias: {risky_alias!r}.

Which short context terms strongly suggest the paper is genuinely about THIS dataset product
(and not another meaning of the alias)?

Use emit_flag_terms_for_alias exactly once."""
    try:
        raw = agent.get_structured(prompt, EMIT_FLAG_TERMS_FOR_ALIAS)
    except Exception as e:
        logger.warning("generate_flag_terms_for_alias failed: %s", e)
        return []
    terms = raw.get("flag_terms")
    if not isinstance(terms, list):
        return []
    return [str(x).strip() for x in terms if isinstance(x, str) and x.strip()][:20]


def generate_domain_positive_keywords(agent: AgentPort, dataset_name: str) -> list[str]:
    """LLM: title override keywords that strongly signal the target dataset."""
    prompt = f"""For the dataset {dataset_name!r}, list a few short phrases that almost always
indicate the publication uses or analyzes THIS dataset (not a homonym).

Use emit_domain_positive_keywords exactly once."""
    try:
        raw = agent.get_structured(prompt, EMIT_DOMAIN_POSITIVE_KEYWORDS)
    except Exception as e:
        logger.warning("generate_domain_positive_keywords failed: %s", e)
        return []
    kw = raw.get("keywords")
    if not isinstance(kw, list):
        return []
    return [str(x).strip() for x in kw if isinstance(x, str) and x.strip()][:15]


def classify_alias_mention(
    agent: AgentPort,
    *,
    dataset_name: str,
    alias: str,
    snippet: str,
    title: str = "",
) -> bool:
    """LLM: whether the alias mention in *snippet* refers to the dataset (True) or an FP (False)."""
    prompt = f"""Dataset product: {dataset_name!r}.
Alias matched: {alias!r}.
Publication title: {title!r}
Abstract snippet: {snippet!r}

Does this mention refer to the dataset product (same data program), or a false positive / other domain?

Use emit_alias_mention_classification exactly once."""
    try:
        raw = agent.get_structured(prompt, EMIT_ALIAS_MENTION_CLASSIFICATION)
    except Exception as e:
        logger.warning("classify_alias_mention failed: %s", e)
        return True
    val = raw.get("is_genuine_dataset_reference")
    return bool(val)


def derive_exclude_terms_from_fps(
    agent: AgentPort,
    *,
    dataset_name: str,
    fp_hits: list[dict[str, str]],
) -> list[str]:
    """LLM: propose exclude terms from confirmed false-positive hit snippets."""
    if not fp_hits:
        return []
    lines = "\n".join(
        f"- alias={h.get('matched_alias')!r} snippet={h.get('snippet', '')[:200]!r}"
        for h in fp_hits[:30]
        if isinstance(h, dict)
    )
    prompt = f"""Dataset: {dataset_name!r}.

These abstract snippets were classified as FALSE POSITIVE mentions of the dataset aliases:
{lines}

Propose short exclude phrases (NOT clause style) that would help drop similar off-domain hits.

Use emit_exclude_terms exactly once."""
    try:
        raw = agent.get_structured(prompt, EMIT_EXCLUDE_TERMS)
    except Exception as e:
        logger.warning("derive_exclude_terms_from_fps failed: %s", e)
        return []
    terms = raw.get("exclude_terms")
    if not isinstance(terms, list):
        return []
    return [str(x).strip() for x in terms if isinstance(x, str) and x.strip()][:25]


def web_search_alias_meanings(agent: AgentPort, alias: str) -> dict[str, list[str]]:
    """Run web search, then LLM-structured parse of alternative meanings → exclude candidates."""
    q = f'"{alias}" acronym meaning disambiguation science'
    body = web_search(q)
    prompt = f"""Web search results for the term {alias!r}:

{body}

Summarize alternative real-world/scientific meanings as domains with short candidate exclude terms
for a literature query filter.

Use emit_web_alias_meanings exactly once."""
    try:
        raw = agent.get_structured(prompt, EMIT_WEB_ALIAS_MEANINGS)
    except Exception as e:
        logger.warning("web_search_alias_meanings failed: %s", e)
        return {}
    domains = raw.get("domains")
    if not isinstance(domains, list):
        return {}
    out: dict[str, list[str]] = {}
    for row in domains:
        if not isinstance(row, dict):
            continue
        name = str(row.get("domain_name") or "").strip()
        cands = row.get("candidate_exclude_terms")
        if not name or not isinstance(cands, list):
            continue
        out[name] = [str(x).strip() for x in cands if isinstance(x, str) and x.strip()][:15]
    return out


def score_title_signal3(
    agent: AgentPort,
    *,
    dataset_name: str,
    dataset_description: str,
    title: str,
    top_context: str = "",
) -> dict[str, Any]:
    """Signal 3: structured relevance scores for one title (0–10 mention + context)."""
    prompt = f"""Dataset: {dataset_name!r}.
Description (short): {dataset_description[:600]!r}
Publication title: {title!r}
Extra context: {top_context[:500]!r}

Does this publication plausibly **use, analyze, or directly reference data from** this dataset?

Use emit_title_relevance exactly once with integer scores 0–10."""
    try:
        return agent.get_structured(prompt, EMIT_TITLE_RELEVANCE)
    except Exception as e:
        logger.warning("score_title_signal3 failed: %s", e)
        return {
            "mention_score": 0,
            "context_score": 0,
            "mentioned_term": "",
            "reason": f"error: {e}",
        }

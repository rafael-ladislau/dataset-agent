"""Literature screening gate (noop or Dimensions)."""

from __future__ import annotations

from collections import Counter
import logging
import re
from typing import TYPE_CHECKING

from dataset_agent.domain.ports import AgentPort, LiteratureGatePort, LiteratureGateResult

if TYPE_CHECKING:
    from dataset_agent.domain.models import DatasetRecord
    from dataset_agent.settings import Settings

logger = logging.getLogger(__name__)


def _extract_concepts_list(raw: object) -> list:
    """Normalize Dimensions `concepts_scores` to a list of entries (dicts or similar)."""
    import json

    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [x for x in raw if x]
    return []


def _concept_relevance(row: object) -> float:
    if not isinstance(row, dict):
        return 0.0
    r = row.get("relevance")
    if r is None:
        r = row.get("score")
    try:
        return float(r)
    except (TypeError, ValueError):
        return 0.0


def _format_concepts_for_prompt(
    raw: object,
    max_items: int = 25,
    min_relevance: float = 0.6,
) -> str:
    """Human-readable concepts block for LLM, filtered by relevance threshold."""
    concepts = _extract_concepts_list(raw)
    if not concepts:
        return ""

    scored = sorted(concepts, key=_concept_relevance, reverse=True)
    lines: list[str] = []
    for row in scored:
        if not isinstance(row, dict):
            continue
        label = row.get("concept") or row.get("label") or row.get("name") or ""
        if not label:
            continue
        rel = _concept_relevance(row)
        if rel <= min_relevance:
            continue
        lines.append(f"- {label} (relevance: {rel:.3f})")
        if len(lines) >= max_items:
            break
    return "\n".join(lines)


def _llm_relevance_prompt(
    dataset_name: str,
    dataset_terms: list[str],
    description: str,
    organizations: list[str],
    title: str,
    abstract: str,
    concepts_block: str,
    match_context: str = "",
) -> str:
    """Prompt for LLM to assess if the publication concerns the dataset (abstract + Dimensions concepts)."""
    terms_str = ", ".join(f'"{t}"' for t in dataset_terms[:5])
    orgs_str = ", ".join(organizations[:4]) if organizations else "Not specified"
    desc_short = description[:400] if description else "Not available"
    title_line = (title or "").strip()[:500] or "Not available"
    concepts_section = (
        concepts_block
        if concepts_block.strip()
        else "Not available for this publication."
    )
    match_section = (
        match_context.strip()
        if match_context.strip()
        else "Not available (no substring hit in title+abstract for configured terms)."
    )

    return f"""Analyze if this academic publication uses or discusses the dataset "{dataset_name}".

=== DATASET CONTEXT ===
Description: {desc_short}
Organizations: {orgs_str}
Alternative names/acronyms: {terms_str}

=== PUBLICATION TITLE ===
{title_line}

=== LEXICAL MATCH CONTEXT (title + abstract, ±500 chars around first substring hit) ===
This is the primary local evidence for a literal mention. Use it together with the abstract and concepts.

{match_section}

=== ABSTRACT TO ANALYZE ===
{abstract}

=== DIMENSIONS INDEXED CONCEPTS ===
Dimensions assigns topical concepts with relevance scores (higher = stronger association with the full text).
Use them together with the title and abstract: they help disambiguate generic mentions and confirm domain alignment.

{concepts_section}

=== SCORING ===
Provide TWO scores from 0-10:

1. MENTION SCORE: Does the title/abstract (or a highly related concept) indicate this specific dataset or its standard names/aliases?
   - 0: No usable signal
   - 5: Related wording or concepts but ambiguous
   - 10: Clear mention or unmistakable conceptual match for this dataset

2. CONTEXT SCORE: Is the publication's domain and intent consistent with this dataset (description, organizations, field)?
   - 0: Wrong field or unrelated work
   - 5: Adjacent topic; could be a false positive
   - 10: Concepts + text clearly match this dataset's purpose and sponsors

Respond ONLY with JSON:
{{"mention_score": <0-10>, "context_score": <0-10>, "mentioned_term": "<best matched alias/term or empty>", "reason": "<10 words max>"}}"""


def _parse_llm_scores(response: str) -> dict:
    """Extract mention_score, context_score, mentioned_term and reason from LLM response."""
    import json
    
    result = {
        "mention_score": 0,
        "context_score": 0,
        "mentioned_term": "",
        "reason": "could not parse",
    }
    
    # Try to parse as JSON
    try:
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group())
            result["mention_score"] = min(10, max(0, int(data.get("mention_score", 0))))
            result["context_score"] = min(10, max(0, int(data.get("context_score", 0))))
            result["mentioned_term"] = str(data.get("mentioned_term", ""))[:120]
            result["reason"] = str(data.get("reason", ""))[:100]
            return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Fallback: try to find numbers
    numbers = re.findall(r'\b(\d+)\b', response)
    if len(numbers) >= 2:
        result["mention_score"] = min(10, max(0, int(numbers[0])))
        result["context_score"] = min(10, max(0, int(numbers[1])))
        result["reason"] = "parsed from text"
    elif len(numbers) == 1:
        result["mention_score"] = min(10, max(0, int(numbers[0])))
        result["reason"] = "single score parsed"
    
    return result


def _calculate_final_score(mention: int, context: int) -> float:
    """Calculate final score from mention and context scores.
    
    Uses minimum of both scores - both must be high for validation.
    """
    return min(mention, context)


def _validate_publications_with_llm(
    publications: list[dict],
    dataset_name: str,
    dataset_terms: list[str],
    description: str,
    organizations: list[str],
    agent: AgentPort,
    match_contexts: list[str] | None = None,
) -> dict:
    """
    Validate each publication using LLM to assess relevance.

    Uses abstract, title, and Dimensions ``concepts_scores`` when present.

    Uses two scores:
    - mention_score: Does the text/concepts indicate this dataset?
    - context_score: Is the thematic context correct for this dataset?

    Final score = min(mention, context) - both must be high.

    Returns validation summary with scores.
    """
    if not publications:
        return {
            "total": 0,
            "final_scores": [],
            "average_score": 0.0,
            "details": [],
        }
    
    final_scores = []
    details = []
    
    for i, pub in enumerate(publications):
        title = pub.get("title", "") or ""
        abstract = pub.get("abstract", "") or ""
        concepts_raw = pub.get("concepts_scores")
        concepts_block = _format_concepts_for_prompt(concepts_raw)
        abstract_ok = bool(abstract and len(abstract.strip()) >= 50)
        concepts_ok = bool(concepts_block.strip())
        match_ctx = ""
        if match_contexts is not None and i < len(match_contexts):
            match_ctx = match_contexts[i] or ""
        match_ctx_ok = bool(match_ctx.strip())

        if not match_ctx_ok and not abstract_ok and not concepts_ok:
            logger.debug("  [%s] Skipping (no match context, abstract, or concepts): %s", i + 1, title[:60])
            final_scores.append(0)
            details.append({
                "title": title[:100] + "..." if len(title) > 100 else title,
                "mention_score": 0,
                "context_score": 0,
                "mentioned_term": "",
                "final_score": 0,
                "reason": "no match context, abstract, or concepts",
                "concepts_preview": "",
                "match_context_preview": "",
            })
            continue

        abstract_for_prompt = (abstract or "")[:2000]
        if concepts_ok and not abstract_ok and not match_ctx_ok:
            if abstract_for_prompt.strip():
                abstract_for_prompt = (
                    f"{abstract_for_prompt}\n\n"
                    "(Note: abstract is short; weight Dimensions concepts and title heavily.)"
                )
            else:
                abstract_for_prompt = (
                    "(No or very short abstract; base judgment on title and Dimensions concepts.)"
                )
        elif match_ctx_ok and not abstract_ok:
            if not abstract_for_prompt.strip():
                abstract_for_prompt = (
                    "(Abstract missing or very short; rely on LEXICAL MATCH CONTEXT and concepts.)"
                )

        prompt = _llm_relevance_prompt(
            dataset_name,
            dataset_terms,
            description,
            organizations,
            title,
            abstract_for_prompt,
            concepts_block,
            match_context=match_ctx,
        )
        logger.info("  [%s/%s] LLM analyzing: %s", i + 1, len(publications), title[:60])
        
        try:
            response = agent.get_information(prompt)
            scores = _parse_llm_scores(response)
            final = _calculate_final_score(scores["mention_score"], scores["context_score"])
            logger.info(
                "  [%s] mention=%s term=%r context=%s final=%s - %s",
                i + 1,
                scores["mention_score"],
                scores["mentioned_term"],
                scores["context_score"],
                final,
                scores["reason"],
            )
        except Exception as e:
            logger.warning("  [%s] LLM error: %s", i + 1, e)
            scores = {
                "mention_score": 0,
                "context_score": 0,
                "mentioned_term": "",
                "reason": f"error: {str(e)[:30]}",
            }
            final = 0
        
        final_scores.append(final)
        details.append({
            "title": title[:100] + "..." if len(title) > 100 else title,
            "mention_score": scores["mention_score"],
            "context_score": scores["context_score"],
            "mentioned_term": scores["mentioned_term"],
            "final_score": final,
            "reason": scores["reason"],
            "concepts_preview": concepts_block[:500] + ("..." if len(concepts_block) > 500 else ""),
            "match_context_preview": match_ctx[:500] + ("..." if len(match_ctx) > 500 else ""),
        })
    
    avg_score = sum(final_scores) / len(final_scores) if final_scores else 0.0
    
    return {
        "total": len(publications),
        "final_scores": final_scores,
        "average_score": avg_score,
        "details": details,
    }


def _terms_evaluation_prompt(
    dataset_name: str,
    description: str,
    dataset_names: list[str],
    flag_terms: list[str],
    validation_details: list[dict],
) -> str:
    """Prompt for LLM to evaluate if dataset_names and flag_terms are effective."""
    names_str = ", ".join(f'"{n}"' for n in dataset_names[:10])
    terms_str = ", ".join(f'"{t}"' for t in flag_terms[:10])
    
    # Summarize validation results
    if validation_details:
        valid_count = sum(1 for d in validation_details if d.get("final_score", 0) >= 5)
        low_context = [d for d in validation_details if d.get("context_score", 0) < 5 and d.get("mention_score", 0) >= 5]
        mentioned_terms = [
            str(d.get("mentioned_term", "")).strip()
            for d in validation_details
            if d.get("mention_score", 0) > 0 and str(d.get("mentioned_term", "")).strip()
        ]
        mentioned_counts = Counter(mentioned_terms)
        top_mentioned_terms = "\n".join(
            f"  - {term}: {count}x"
            for term, count in mentioned_counts.most_common(5)
        )
        if not top_mentioned_terms:
            top_mentioned_terms = "  - None"
        low_context_summary = "\n".join(
            f"  - '{d.get('title', '')[:60]}...' (mention={d.get('mention_score')}, context={d.get('context_score')})"
            for d in low_context[:3]
        )
    else:
        valid_count = 0
        low_context_summary = "No details available"
        top_mentioned_terms = "No semantic term match details available"
    
    return f"""Evaluate the effectiveness of search terms for the dataset "{dataset_name}".

=== DATASET INFO ===
Description: {description}

=== CURRENT TERMS ===
Dataset names/aliases: {names_str}
Organizations (flag_terms): {terms_str}

=== VALIDATION RESULTS ===
Publications found with valid context: {valid_count}/{len(validation_details)}

Publications with HIGH mention but LOW context score (likely false positives):
{low_context_summary}

Most frequently matched semantic terms (from mention_score analysis):
{top_mentioned_terms}

=== YOUR TASK ===
Evaluate if the current terms are effective for finding relevant literature.

Consider:
1. Are dataset names specific enough? (e.g., "ORS" alone is too generic)
2. Are the organizations correct and complete?
3. What terms would better identify this specific dataset?

Respond with ONLY a JSON object:
{{
  "is_effective": <true/false>,
  "dataset_names_score": <0-10>,
  "flag_terms_score": <0-10>,
  "issues": ["issue1", "issue2"],
  "suggested_dataset_names": ["better term 1", "better term 2"],
  "suggested_flag_terms": ["org1", "org2"],
  "reasoning": "<brief explanation, max 50 words>"
}}"""


def _parse_terms_evaluation(response: str) -> dict:
    """Parse LLM response for terms evaluation."""
    import json
    
    default = {
        "is_effective": True,
        "dataset_names_score": 5,
        "flag_terms_score": 5,
        "issues": [],
        "suggested_dataset_names": [],
        "suggested_flag_terms": [],
        "reasoning": "Could not parse LLM response",
    }
    
    try:
        # Find JSON in response
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            data = json.loads(match.group())
            return {
                "is_effective": bool(data.get("is_effective", True)),
                "dataset_names_score": min(10, max(0, int(data.get("dataset_names_score", 5)))),
                "flag_terms_score": min(10, max(0, int(data.get("flag_terms_score", 5)))),
                "issues": list(data.get("issues", []))[:5],
                "suggested_dataset_names": list(data.get("suggested_dataset_names", []))[:5],
                "suggested_flag_terms": list(data.get("suggested_flag_terms", []))[:5],
                "reasoning": str(data.get("reasoning", ""))[:200],
            }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Failed to parse terms evaluation: %s", e)
    
    return default


def evaluate_terms_with_llm(
    dataset_name: str,
    description: str,
    dataset_names: list[str],
    flag_terms: list[str],
    validation_details: list[dict],
    agent: AgentPort,
) -> dict:
    """
    Evaluate if dataset_names and flag_terms are effective for literature search.
    
    Called when validation score is low to provide feedback and suggestions.
    """
    logger.info("Evaluating terms effectiveness with LLM...")
    
    prompt = _terms_evaluation_prompt(
        dataset_name=dataset_name,
        description=description,
        dataset_names=dataset_names,
        flag_terms=flag_terms,
        validation_details=validation_details,
    )
    
    try:
        response = agent.get_information(prompt)
        result = _parse_terms_evaluation(response)
        logger.info(
            "Terms evaluation: effective=%s names_score=%s terms_score=%s",
            result["is_effective"],
            result["dataset_names_score"],
            result["flag_terms_score"],
        )
        if result["issues"]:
            logger.info("Issues found: %s", result["issues"])
        if result["suggested_dataset_names"]:
            logger.info("Suggested dataset names: %s", result["suggested_dataset_names"])
        if result["suggested_flag_terms"]:
            logger.info("Suggested flag terms: %s", result["suggested_flag_terms"])
        return result
    except Exception as e:
        logger.warning("Terms evaluation failed: %s", e)
        return {
            "is_effective": True,
            "dataset_names_score": 5,
            "flag_terms_score": 5,
            "issues": [f"Evaluation error: {str(e)[:50]}"],
            "suggested_dataset_names": [],
            "suggested_flag_terms": [],
            "reasoning": "Could not complete evaluation",
        }


def _build_ordered_search_terms(
    main_dataset_name: str,
    dataset_names: list[str] | None,
    flag_terms: list[str] | None,
) -> list[tuple[str, str]]:
    """
    Ordered (term, source) with source in {\"dataset\", \"flag\"}.
    Order: main name, aliases, expansions from names, then flag_terms.
    Deduplication is case-insensitive (only .lower()). No whitespace normalization.
    """
    ordered: list[tuple[str, str]] = []
    seen_lower: set[str] = set()

    def add(term: str, src: str) -> None:
        t = (term or "").strip()
        if len(t) < 2:
            return
        k = t.lower()
        if k in seen_lower:
            return
        seen_lower.add(k)
        ordered.append((t, src))

    add(main_dataset_name.strip() if main_dataset_name else "", "dataset")
    for x in dataset_names or []:
        if isinstance(x, str):
            add(x, "dataset")

    raw_names = [main_dataset_name] + list(dataset_names or [])
    for term in raw_names:
        if not term:
            continue
        for ac in re.findall(r"\(([A-Z]{2,})\)", term):
            add(ac, "dataset")
        if " - " in term:
            for p in term.split(" - "):
                p = p.strip()
                if len(p) > 1:
                    add(p, "dataset")

    for f in flag_terms or []:
        if isinstance(f, str):
            add(f, "flag")

    return ordered


def _find_first_substring_match(
    text: str,
    ordered_terms: list[tuple[str, str]],
    min_term_len: int = 3,
) -> tuple[str, str, int, int] | None:
    """
    First hit in ordered_terms using case fold .lower() only (no whitespace collapse).
    Returns (matched_term, source, start, end_exclusive) in original ``text`` indices.
    """
    if not text or not ordered_terms:
        return None
    hay = text.lower()
    for term, src in ordered_terms:
        if len(term) < min_term_len:
            continue
        needle = term.lower()
        pos = hay.find(needle)
        if pos != -1:
            return (term, src, pos, pos + len(needle))
    return None


def _match_context_window(
    text: str, start: int, end: int, before: int = 500, after: int = 500
) -> str:
    if not text or start < 0 or end < 0 or start > end or end > len(text):
        return ""
    w0 = max(0, start - before)
    w1 = min(len(text), end + after)
    return text[w0:w1]


def _validate_publications_lexical(
    publications: list[dict],
    ordered_terms: list[tuple[str, str]],
    min_term_len: int = 3,
    context_before: int = 500,
    context_after: int = 500,
) -> dict:
    """
    Prefilter: first substring match (case: .lower() only) in title+abstract.
    valid=True if any term matches. Provides match_context window for LLM.
    """
    if not publications:
        return {
            "total": 0,
            "valid": 0,
            "indicator": 0.0,
            "details": [],
        }

    valid_count = 0
    details: list[dict] = []

    for idx, pub in enumerate(publications):
        title = pub.get("title", "") or ""
        abstract = pub.get("abstract", "") or ""
        text = f"{title} {abstract}"

        hit = _find_first_substring_match(text, ordered_terms, min_term_len=min_term_len)
        if hit:
            matched_term, src, start, end_excl = hit
            window = _match_context_window(
                text, start, end_excl, before=context_before, after=context_after
            )
            is_valid = True
            valid_count += 1
            dataset_found = [matched_term] if src == "dataset" else []
            flag_found = [matched_term] if src == "flag" else []
            lexical_score = float(len(matched_term))
        else:
            matched_term, src = "", ""
            start, end_excl = -1, -1
            window = ""
            is_valid = False
            dataset_found, flag_found = [], []
            lexical_score = 0.0

        details.append({
            "index": idx,
            "title": title[:100] + "..." if len(title) > 100 else title,
            "valid": is_valid,
            "lexical_score": lexical_score,
            "matched_term": matched_term,
            "match_source": src,
            "match_span": [start, end_excl] if hit else None,
            "match_context": window,
            "dataset_terms_found": dataset_found,
            "flag_terms_found": flag_found,
        })

    indicator = valid_count / len(publications) if publications else 0.0

    return {
        "total": len(publications),
        "valid": valid_count,
        "indicator": indicator,
        "details": details,
    }


def _dimensions_publication_id(pub: object) -> str:
    """Dimensions publication id from a record dict (field ``id``)."""
    if isinstance(pub, dict):
        pid = pub.get("id")
        if pid is not None and str(pid).strip():
            return str(pid).strip()
    return ""


def _build_string_search_matched_publications(
    pubs: list[dict],
    lexical_match_details: list[dict],
) -> list[dict]:
    """
    Publications that matched the substring prefilter, in sort order of lexical_match_details.
    Each item: ``publication_id`` and ``match_snippet`` (±500 window around the hit).
    """
    matched: list[dict] = []
    for d in lexical_match_details:
        if not d.get("valid"):
            continue
        idx = int(d.get("index", -1))
        if not (0 <= idx < len(pubs)):
            continue
        snippet = str(d.get("match_context", "") or "")
        matched.append(
            {
                "publication_id": _dimensions_publication_id(pubs[idx]),
                "match_snippet": snippet,
            }
        )
    return matched


class NoOpLiteratureGate(LiteratureGatePort):
    def assess(
        self,
        record: DatasetRecord,
        sample_size: int = 10,
        llm_batch_size: int = 25,
    ) -> LiteratureGateResult:
        _ = (sample_size, llm_batch_size)
        logger.info(
            "Literature gate noop: accepted without lookup (dataset=%r)",
            record.main_dataset_name,
        )
        return LiteratureGateResult(True, 1.0, {"mode": "noop"})


class DimensionsLiteratureGate(LiteratureGatePort):
    """
    Dimensions check with lexical + LLM validation.
    
    1. Query Dimensions for publications
    2. LLM validation: analyze each abstract for dataset relevance (score 0-10)
    3. Calculate overall score and pass/fail
    
    Requires optional dependency dimcli and credentials in Settings.
    """

    def __init__(self, settings: Settings, agent: AgentPort | None = None):
        self._settings = settings
        self._agent = agent

    def assess(
        self,
        record: DatasetRecord,
        sample_size: int = 10,
        llm_batch_size: int = 25,
    ) -> LiteratureGateResult:
        logger.info(
            "Literature gate Dimensions: assessing dataset=%r (sample_size=%s llm_batch_size=%s)",
            record.main_dataset_name,
            sample_size,
            llm_batch_size,
        )
        try:
            import dimcli  # type: ignore
        except ImportError:
            logger.warning("dimcli not installed; gate accepts without API call")
            return LiteratureGateResult(True, 1.0, {"skip": "dimcli_not_installed"})

        api_key = self._settings.dimensions_api_key
        if not api_key:
            logger.warning("Dimensions API key missing; gate accepts")
            return LiteratureGateResult(True, 1.0, {"skip": "no_api_key"})

        try:
            logger.info("Dimensions: authenticating with API key...")
            dimcli.login(key=api_key)
            dsl = dimcli.Dsl()
            
            # Build dataset names group: main name + all aliases
            dataset_terms_raw = [record.main_dataset_name]
            if record.dataset_names:
                dataset_terms_raw.extend(record.dataset_names)
            
            # For query: escape quotes, filter empty/short
            dataset_terms_query = [t.replace('"', '\\"') for t in dataset_terms_raw if t and len(t) > 2]
            
            # For lexical validation: include original terms + extracted acronyms
            dataset_terms_validation = list(dataset_terms_raw)
            for term in dataset_terms_raw:
                # Extract acronyms from parentheses: "Name (ABC)" -> add "ABC"
                acronyms = re.findall(r'\(([A-Z]{2,})\)', term)
                dataset_terms_validation.extend(acronyms)
                # Split "Name - ACRONYM" patterns
                if ' - ' in term:
                    parts = term.split(' - ')
                    dataset_terms_validation.extend(p.strip() for p in parts if len(p.strip()) > 1)
            dataset_terms_validation = list(set(t for t in dataset_terms_validation if t and len(t) > 1))
            
            # Build flag_terms group (organizations) - all terms
            flag_terms_raw = record.flag_terms if record.flag_terms else []
            flag_terms_query = [t.replace('"', '\\"') for t in flag_terms_raw if t and len(t) > 2]
            flag_terms_validation = list(set(t for t in flag_terms_raw if t and len(t) > 1))
            
            logger.debug("Validation terms - dataset: %s", dataset_terms_validation)
            logger.debug("Validation terms - flag: %s", flag_terms_validation)
            
            # Build query: (dataset_name1 OR dataset_name2) AND (flag_term1 OR flag_term2)
            dataset_clause = " OR ".join(f'\\"{t}\\"' for t in dataset_terms_query)
            
            if flag_terms_query:
                flag_clause = " OR ".join(f'\\"{t}\\"' for t in flag_terms_query)
                q = (
                    'search publications in full_data for '
                    f'"(({dataset_clause}) AND ({flag_clause}))" '
                    "return publications[basics + abstract + concepts_scores + times_cited] "
                    f"sort by times_cited limit {sample_size}"
                )
            else:
                # Fallback: only dataset names if no flag_terms
                q = (
                    f'search publications in full_data for "({dataset_clause})" '
                    "return publications[basics + abstract + concepts_scores + times_cited] "
                    f"sort by times_cited limit {sample_size}"
                )
            
            logger.info("Dimensions: running query: %s", q)
            
            result = dsl.query(q)
            
            # Get total count from dimcli
            total_count = 0
            if hasattr(result, 'count_total'):
                total_count = result.count_total
            elif hasattr(result, '_stats') and isinstance(result._stats, dict):
                total_count = result._stats.get('total_count', 0)
            
            # Extract publications from result
            pubs = result.get("publications", []) if hasattr(result, "get") else []
            if not pubs and hasattr(result, "publications"):
                pubs = result.publications or []
            
            logger.info("Dimensions: %s publications analyzed (total available: %s)", len(pubs), total_count)
            
            if not pubs:
                logger.warning("Dimensions: no publications found")
                return LiteratureGateResult(False, 0.0, {"publications_analyzed": 0, "publications_total": total_count, "query": q})
            
            logger.info("Dimensions: running lexical prefilter on %s publications...", len(pubs))
            ordered_search_terms = _build_ordered_search_terms(
                record.main_dataset_name,
                record.dataset_names,
                record.flag_terms,
            )
            lexical_validation = _validate_publications_lexical(
                publications=pubs,
                ordered_terms=ordered_search_terms,
            )
            lexical_candidates = [d for d in lexical_validation["details"] if d.get("valid")]
            lexical_candidates_sorted = sorted(
                lexical_candidates,
                key=lambda d: int(d.get("index", 0)),
            )
            string_search_matched_publications = _build_string_search_matched_publications(
                pubs, lexical_candidates_sorted
            )
            selected_details = lexical_candidates_sorted[: max(0, llm_batch_size)]
            selected_indices = [int(d["index"]) for d in selected_details]
            selected_pubs = [pubs[i] for i in selected_indices if 0 <= i < len(pubs)]
            selected_match_contexts = [str(d.get("match_context", "") or "") for d in selected_details]

            # LLM validation: analyze lexical top-N candidates only
            if self._agent and llm_batch_size > 0 and selected_pubs:
                logger.info(
                    "Dimensions: running LLM validation on %s/%s lexical candidates...",
                    len(selected_pubs),
                    len(lexical_candidates),
                )
                llm_validation = _validate_publications_with_llm(
                    publications=selected_pubs,
                    dataset_name=record.main_dataset_name,
                    dataset_terms=dataset_terms_validation,
                    description=record.description,
                    organizations=record.flag_terms,
                    agent=self._agent,
                    match_contexts=selected_match_contexts,
                )
                
                # Score: average of final scores (min of mention, context) normalized to 0-1
                avg_score = llm_validation["average_score"]
                indicator = avg_score / 10.0  # Convert 0-10 to 0-1
                
                # Count publications with final_score >= 5 as "valid"
                valid_count = sum(1 for s in llm_validation["final_scores"] if s >= 5)
                
                thr = self._settings.literature_threshold
                passed = indicator >= thr
                
                logger.info(
                    "Dimensions: LLM validation complete - total=%s valid=%s (score>=5) avg_score=%.1f indicator=%.2f threshold=%.2f passed=%s",
                    llm_validation["total"],
                    valid_count,
                    avg_score,
                    indicator,
                    thr,
                    passed,
                )
                
                return LiteratureGateResult(
                    passed=passed,
                    indicator=indicator,
                    detail={
                        "query": q,
                        "publications_total": total_count,
                        "publications_retrieved": len(pubs),
                        "lexical_prefilter_total": lexical_validation["total"],
                        "lexical_prefilter_valid": lexical_validation["valid"],
                        "llm_analyzed": llm_validation["total"],
                        "publications_analyzed": llm_validation["total"],
                        "llm_valid": valid_count,
                        "publications_valid": valid_count,
                        "average_score": avg_score,
                        "validation_type": "lexical_prefilter_llm",
                        "string_search_matched_publications": string_search_matched_publications,
                        "lexical_details": selected_details,
                        "validation_details": llm_validation["details"],
                    },
                )

            # Fallback to lexical-only validation if no agent, no candidates, or llm_batch_size disabled
            if self._agent and llm_batch_size > 0 and not selected_pubs:
                logger.info("Dimensions: lexical prefilter returned no valid candidates for LLM")

            logger.info(
                "Dimensions: running lexical-only validation (agent=%s llm_batch_size=%s selected=%s)",
                bool(self._agent),
                llm_batch_size,
                len(selected_pubs),
            )
            indicator = lexical_validation["indicator"]
            thr = self._settings.literature_threshold
            passed = indicator >= thr

            logger.info(
                "Dimensions: lexical validation complete - total=%s valid=%s indicator=%.2f threshold=%.2f passed=%s",
                lexical_validation["total"],
                lexical_validation["valid"],
                indicator,
                thr,
                passed,
            )

            return LiteratureGateResult(
                passed=passed,
                indicator=indicator,
                detail={
                    "query": q,
                    "publications_total": total_count,
                    "publications_retrieved": len(pubs),
                    "lexical_prefilter_total": lexical_validation["total"],
                    "lexical_prefilter_valid": lexical_validation["valid"],
                    "publications_analyzed": lexical_validation["total"],
                    "publications_valid": lexical_validation["valid"],
                    "llm_analyzed": 0,
                    "llm_valid": 0,
                    "validation_type": "lexical",
                    "string_search_matched_publications": string_search_matched_publications,
                    "validation_details": lexical_validation["details"][:5],
                },
            )
        except Exception as e:
            logger.exception("Dimensions gate error")
            return LiteratureGateResult(False, 0.0, {"error": str(e)})


def literature_gate_from_settings(
    settings: Settings, agent: AgentPort | None = None
) -> LiteratureGatePort:
    if settings.literature_gate == "dimensions":
        return DimensionsLiteratureGate(settings, agent=agent)
    return NoOpLiteratureGate()

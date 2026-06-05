"""Literature screening gate (noop or Dimensions)."""

from __future__ import annotations

from collections import Counter
import logging
import re
import time
from typing import Any
from dataset_agent.adapters.dimensions_dsl import parse_total_count, publication_title, publications_from_result
from dataset_agent.adapters.fp_analysis import classify_alias_mention
from dataset_agent.adapters.tools import EMIT_RELEVANCE_SCORE, EMIT_TERMS_EVALUATION
from dataset_agent.domain.ports import AgentPort, LiteratureGatePort, LiteratureGateResult

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

Call the emit_relevance_score tool with your scores."""


def _default_scores() -> dict:
    """Return zero-valued score dict used on parse failure."""
    return {
        "mention_score": 0,
        "context_score": 0,
        "mentioned_term": "",
        "reason": "could not parse",
    }


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
        publication_id = _dimensions_publication_id(pub)
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
                "publication_id": publication_id,
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
            raw = agent.get_structured(prompt, EMIT_RELEVANCE_SCORE)
            scores = {
                "mention_score": min(10, max(0, int(raw.get("mention_score", 0)))),
                "context_score": min(10, max(0, int(raw.get("context_score", 0)))),
                "mentioned_term": str(raw.get("mentioned_term", ""))[:120],
                "reason": str(raw.get("reason", ""))[:100],
            } if raw else _default_scores()
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
            scores = _default_scores()
            scores["reason"] = f"error: {str(e)[:30]}"
            final = 0
        
        final_scores.append(final)
        details.append({
            "publication_id": publication_id,
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
        n_val = len(validation_details)
        pct_ok = round(100.0 * valid_count / n_val, 1) if n_val else 0.0
        mention_scores = [int(d.get("mention_score", 0) or 0) for d in validation_details]
        context_scores = [int(d.get("context_score", 0) or 0) for d in validation_details]
        avg_mention = round(sum(mention_scores) / len(mention_scores), 2) if mention_scores else 0.0
        avg_context = round(sum(context_scores) / len(context_scores), 2) if context_scores else 0.0
        metrics_line = (
            f"Share with final_score>=5: {valid_count}/{n_val} ({pct_ok}%); "
            f"avg mention_score={avg_mention}, avg context_score={avg_context}."
        )
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
        metrics_line = "No per-publication validation details."
        low_context_summary = "No details available"
        top_mentioned_terms = "No semantic term match details available"

    return f"""Evaluate the effectiveness of search terms for the dataset "{dataset_name}".

=== DATASET INFO ===
Description: {description}

=== CURRENT TERMS ===
Dataset names/aliases: {names_str}
Organizations (flag_terms): {terms_str}

=== VALIDATION RESULTS ===
Publications with final_score>=5: {valid_count}/{len(validation_details)}
{metrics_line}

Publications with HIGH mention but LOW context score (likely false positives):
{low_context_summary}

Most frequently matched semantic terms (from mention_score analysis):
{top_mentioned_terms}

=== YOUR TASK ===
Literature search uses **exact / substring string matching** on titles and abstracts. Suggested terms must be
**real phrases that appear in papers**, not abstract domain descriptions.

**Rules for suggested_dataset_names and suggested_flag_terms:**
- Fill them **only** if there is a **concrete gap**: many high-mention/low-context false positives, terms too generic
  for this dataset, or a **documented alternate official name / spelling** (product name, acronym variant) that is
  missing from CURRENT TERMS and would improve recall.
- If validation is **strong** (most publications relevant, high average scores), use **empty arrays** ``[]`` for
  both and state in issues/reasoning that current terms are adequate. **Do not** invent "improvements" for a system
  that already works.
- **Do not** suggest vague conceptual paraphrases (e.g. "workforce skills database", "occupational taxonomy",
  "labor market information") unless that exact phrase is the dataset's official name. Prefer **spelling/branding
  variants** (e.g. alternate acronym or full title) aligned with DATASET INFO.
- Do **not** repeat any string already listed under CURRENT TERMS (see duplicate policy below).

**suggested_exclude_terms:** If validation shows clear **off-domain** false positives, propose specific phrases to
exclude; otherwise ``[]``. Do not use ultra-generic words ("study", "data", "analysis"). Max 5 items.

Do **not** repeat any name or organization already listed under CURRENT TERMS in
``suggested_dataset_names`` or ``suggested_flag_terms``. Those fields must contain only **new** strings (or empty arrays).

Consider:
1. Are dataset names specific enough for string search? (e.g., a bare acronym may be too ambiguous)
2. Are organizations/sponsors correctly represented for disambiguation?
3. Is there evidence of **wrong-domain** hits that justify exclude phrases?

Call the emit_terms_evaluation tool with your evaluation."""


def _drop_suggestions_already_in_query(
    suggested_names: list[str],
    suggested_flags: list[str],
    *,
    main_dataset_name: str,
    dataset_names: list[str],
    flag_terms: list[str],
) -> tuple[list[str], list[str]]:
    """
    Remove LLM suggestions that duplicate terms already used in the search (case-insensitive).

    ``main_dataset_name`` is treated as already in use for dataset-name suggestions.
    """
    def _key_set(items: list[str], extra: str | None) -> set[str]:
        keys = {str(x).strip().lower() for x in items if x and str(x).strip()}
        if extra and str(extra).strip():
            keys.add(str(extra).strip().lower())
        return keys

    name_used = _key_set(list(dataset_names or []), main_dataset_name)
    flag_used = _key_set(list(flag_terms or []), None)

    out_names: list[str] = []
    for x in suggested_names:
        if not isinstance(x, str):
            continue
        t = x.strip()
        if not t or t.lower() in name_used:
            continue
        out_names.append(t)

    out_flags: list[str] = []
    for x in suggested_flags:
        if not isinstance(x, str):
            continue
        t = x.strip()
        if not t or t.lower() in flag_used:
            continue
        out_flags.append(t)

    return out_names, out_flags


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
        raw = agent.get_structured(prompt, EMIT_TERMS_EVALUATION)
        result = (
            {
                "is_effective": bool(raw.get("is_effective", True)),
                "dataset_names_score": min(10, max(0, int(raw.get("dataset_names_score", 5)))),
                "flag_terms_score": min(10, max(0, int(raw.get("flag_terms_score", 5)))),
                "issues": list(raw.get("issues", []))[:5],
                "suggested_dataset_names": list(raw.get("suggested_dataset_names", []))[:5],
                "suggested_flag_terms": list(raw.get("suggested_flag_terms", []))[:5],
                "suggested_exclude_terms": list(raw.get("suggested_exclude_terms", []))[:5],
                "reasoning": str(raw.get("reasoning", ""))[:200],
            }
            if raw
            else {
                "is_effective": True,
                "dataset_names_score": 5,
                "flag_terms_score": 5,
                "issues": [],
                "suggested_dataset_names": [],
                "suggested_flag_terms": [],
                "suggested_exclude_terms": [],
                "reasoning": "No result from structured call",
            }
        )
        sn, sf = _drop_suggestions_already_in_query(
            result.get("suggested_dataset_names") or [],
            result.get("suggested_flag_terms") or [],
            main_dataset_name=dataset_name,
            dataset_names=dataset_names,
            flag_terms=flag_terms,
        )
        result["suggested_dataset_names"] = sn
        result["suggested_flag_terms"] = sf
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
            "suggested_exclude_terms": [],
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
    Split parts (`` - ``) are only added if length >= 5 to avoid injecting very short
    tokens (e.g. ORS) from ``Name - ORS`` that substring-match inside common words.
    Parenthetical acronyms ``(ABC)`` are added from length 3 onward (matched with
    word-boundary rules for short tokens in :func:`_find_first_substring_match`).
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
            if len(ac) >= 3:
                add(ac, "dataset")
        if " - " in term:
            for p in term.split(" - "):
                p = p.strip()
                if len(p) >= 5:
                    add(p, "dataset")

    for f in flag_terms or []:
        if isinstance(f, str):
            add(f, "flag")

    return ordered


def _find_first_substring_match(
    text: str,
    ordered_terms: list[tuple[str, str]],
    min_term_len: int = 3,
    word_boundary_max_len: int = 3,
) -> tuple[str, str, int, int] | None:
    """
    First hit in ordered_terms. Terms with len <= word_boundary_max_len use whole-token
    match (``\\b...\\b``, case-insensitive) to avoid hits inside unrelated words (e.g. ORS in authors).
    Longer terms use case-insensitive substring search via ``str.find`` on lowered text.
    Returns (matched_term, source, start, end_exclusive) in original ``text`` indices.
    """
    if not text or not ordered_terms:
        return None
    hay = text.lower()
    for term, src in ordered_terms:
        if len(term) < min_term_len:
            continue
        if len(term) <= word_boundary_max_len:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            m = pattern.search(text)
            if m:
                return (term, src, m.start(), m.end())
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


def _dimensions_publication_id(pub: object) -> str:
    """Dimensions publication id from a record dict (field ``id``)."""
    if isinstance(pub, dict):
        pid = pub.get("id")
        if pid is not None and str(pid).strip():
            return str(pid).strip()
    return ""


def _pub_index_by_publication_id(pubs: list[dict]) -> dict[str, int]:
    """First occurrence index per publication id (for stable ordering)."""
    out: dict[str, int] = {}
    for i, p in enumerate(pubs):
        pid = _dimensions_publication_id(p)
        if pid and pid not in out:
            out[pid] = i
    return out


def _pub_by_publication_id(pubs: list[dict], publication_id: str) -> dict | None:
    for p in pubs:
        if _dimensions_publication_id(p) == publication_id:
            return p
    return None


def _filter_publications_by_exclude_terms(
    publications: list[dict],
    exclude_terms: list[str] | None,
) -> list[dict]:
    """
    Drop publications whose title+abstract contain any exclude phrase (case-insensitive).

    Applied after the Dimensions fetch so the API query stays broad; avoids ``AND NOT``
    in the DSL which often over-shrinks recall.
    """
    if not publications or not exclude_terms:
        return publications
    needles = [t.strip().lower() for t in exclude_terms if t and len(t.strip()) > 2]
    if not needles:
        return publications
    out: list[dict] = []
    for pub in publications:
        title = (pub.get("title") or "").lower()
        abstract = (pub.get("abstract") or "").lower()
        blob = f"{title} {abstract}"
        if any(n in blob for n in needles):
            continue
        out.append(pub)
    return out


def _validate_publications_lexical(
    publications: list[dict],
    ordered_terms: list[tuple[str, str]],
    min_term_len: int = 3,
    word_boundary_max_len: int = 3,
    context_before: int = 500,
    context_after: int = 500,
) -> dict:
    """
    Prefilter: first term match in title+abstract (short tokens: word-boundary; long: substring).
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

    for pub in publications:
        title = pub.get("title", "") or ""
        abstract = pub.get("abstract", "") or ""
        text = f"{title} {abstract}"

        hit = _find_first_substring_match(
            text,
            ordered_terms,
            min_term_len=min_term_len,
            word_boundary_max_len=word_boundary_max_len,
        )
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
            "publication_id": _dimensions_publication_id(pub),
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


def _build_string_search_matched_publications(lexical_match_details: list[dict]) -> list[dict]:
    """
    Publications that matched the substring prefilter (each detail row must include
    ``publication_id`` and ``match_context``). Each output item: ``publication_id`` and
    ``match_snippet`` (same text as ``match_context``).
    """
    matched: list[dict] = []
    for d in lexical_match_details:
        if not d.get("valid"):
            continue
        snippet = str(d.get("match_context", "") or "")
        matched.append(
            {
                "publication_id": str(d.get("publication_id", "") or ""),
                "match_snippet": snippet,
            }
        )
    return matched


def build_dataset_validation_terms(record: DatasetRecord) -> list[str]:
    """Dataset name aliases plus acronyms extracted for lexical/LLM validation."""
    dataset_terms_raw = [record.main_dataset_name]
    if record.dataset_names:
        dataset_terms_raw.extend(record.dataset_names)
    dataset_terms_validation = list(dataset_terms_raw)
    for term in dataset_terms_raw:
        acronyms = re.findall(r"\(([A-Z]{2,})\)", term)
        dataset_terms_validation.extend(acronyms)
        if " - " in term:
            parts = term.split(" - ")
            dataset_terms_validation.extend(p.strip() for p in parts if len(p.strip()) > 1)
    return list({t for t in dataset_terms_validation if t and len(t) > 1})


def build_dimensions_literature_query(record: DatasetRecord, *, sample_size: int) -> str:
    """Build the Dimensions DSL string used by the literature gate and research preview."""
    dataset_terms_raw = [record.main_dataset_name]
    if record.dataset_names:
        dataset_terms_raw.extend(record.dataset_names)
    dataset_terms_query = [t.replace('"', '\\"') for t in dataset_terms_raw if t and len(t) > 2]
    flag_terms_raw = record.flag_terms if record.flag_terms else []
    flag_terms_query = [t.replace('"', '\\"') for t in flag_terms_raw if t and len(t) > 2]
    dataset_clause = " OR ".join(f'\\"{t}\\"' for t in dataset_terms_query)
    if flag_terms_query:
        flag_clause = " OR ".join(f'\\"{t}\\"' for t in flag_terms_query)
        search_inner = f"(({dataset_clause}) AND ({flag_clause}))"
    else:
        search_inner = f"({dataset_clause})"
    return (
        "search publications in full_data for "
        f'"{search_inner}" '
        "return publications[basics + abstract + concepts_scores + times_cited] "
        f"sort by times_cited limit {sample_size}"
    )


def execute_dimensions_literature_query(settings: Settings, q: str) -> tuple[object, int]:
    """Authenticate with Dimensions and run a literature DSL query."""
    from dataset_agent.adapters.dimensions_dsl import parse_total_count

    try:
        import dimcli  # type: ignore
    except ImportError as exc:
        raise RuntimeError("dimcli is not installed") from exc

    api_key = (settings.dimensions_api_key or "").strip()
    if not api_key:
        raise ValueError("Dimensions API key is not configured")
    dimcli.login(key=api_key)
    dsl = dimcli.Dsl()
    logger.info("Dimensions: running query: %s", q)
    result = dsl.query(q)
    return result, parse_total_count(result)


def fetch_dimensions_publication_metrics(
    record: DatasetRecord,
    *,
    sample_size: int,
    settings: Settings,
) -> tuple[str, int]:
    """Return (dsl_query, publications_total) without running validation."""
    q = build_dimensions_literature_query(record, sample_size=sample_size)
    _, total_count = execute_dimensions_literature_query(settings, q)
    return q, total_count


def promote_dimensions_metrics_from_detail(record: DatasetRecord, detail: dict) -> None:
    """Copy Dimensions DSL and publication count from gate detail onto the record."""
    if not detail:
        return
    if detail.get("query"):
        record.dsl_query = detail.get("query")
    if detail.get("publications_total") is not None:
        record.publications_total = detail.get("publications_total")


def _stem_text(text: str) -> str:
    """Light custom stem: lowercase, fold punctuation, collapse whitespace."""
    s = (text or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pub_text(pub: dict) -> str:
    """Concatenate title + abstract for stem matching."""
    title = publication_title(pub)
    abstract = ""
    basics = pub.get("basics")
    if isinstance(basics, dict):
        abstract = basics.get("abstract") or ""
    if not abstract:
        abstract = pub.get("abstract") or ""
    return f"{title} {abstract}"


def _find_matched_alias(pub: dict, aliases: list[str]) -> str:
    """Return the first alias whose stem appears in the publication text."""
    text = _stem_text(_pub_text(pub))
    for alias in aliases:
        if alias and _stem_text(alias) in text:
            return alias
    return ""


def _extract_snippet(text: str, alias: str, radius: int = 300) -> str:
    """Extract ~radius chars around the alias occurrence."""
    idx = text.lower().find(alias.lower())
    if idx == -1:
        return (text or "")[:600]
    start = max(0, idx - radius)
    end = min(len(text), idx + len(alias) + radius)
    return text[start:end]


def _build_for_clause_from_record(record: DatasetRecord) -> str:
    """Build a V4 for-clause from the record's terms."""
    from dataset_agent.adapters.query_builder import build_for_clause

    terms = [record.main_dataset_name] + list(record.dataset_names or [])
    terms = [t for t in terms if t and len(t) > 2]
    flags = [t for t in (record.flag_terms or []) if t and len(t) > 2]
    excludes = [t for t in (record.exclude_terms or []) if t and len(t) > 2]
    if not terms:
        return ""
    return build_for_clause(
        safe=terms,
        variant="V4",
        flag_terms=flags,
        exclusion_terms=excludes,
    )


def evaluate_for_clause_sync(
    for_clause: str,
    aliases: list[str],
    dataset_name: str,
    settings: Settings,
    agent: AgentPort | None = None,
) -> dict:
    """
    Sync candidate-DSL evaluator.

    Pulls up to *fp_sample_size* pubs, matches alias stems in title+abstract,
    conditionally resolves reference_ids, classifies matched pubs, and returns
    FP metrics.
    """
    try:
        import dimcli  # type: ignore[import-not-found]
    except ImportError:
        return {
            "fp_rate_pct": 0.0,
            "genuine_count": 0,
            "matched_count": 0,
            "total_pubs": 0,
            "error": "dimcli_not_installed",
        }

    api_key = (settings.dimensions_api_key or "").strip()
    if not api_key:
        return {
            "fp_rate_pct": 0.0,
            "genuine_count": 0,
            "matched_count": 0,
            "total_pubs": 0,
            "error": "no_api_key",
        }

    pull_size = min(int(settings.fp_sample_size), 1000)
    q = (
        f'search publications in full_data for "{for_clause}" '
        f"return publications[basics+title+abstract+reference_ids+times_cited] "
        f"sort by times_cited limit {pull_size}"
    )

    dimcli.login(key=api_key)
    dsl_client = dimcli.Dsl()
    result = dsl_client.query(q)
    total = parse_total_count(result)
    pubs = publications_from_result(result)

    if not pubs:
        return {
            "fp_rate_pct": 0.0,
            "genuine_count": 0,
            "matched_count": 0,
            "total_pubs": 0,
            "coverage_pct": 0.0,
            "for_clause": for_clause,
            "query": q,
            "publications_total": total,
        }

    alias_stems = [_stem_text(a) for a in aliases if a]

    matched: list[dict] = []
    nonmatched: list[dict] = []
    for pub in pubs:
        text = _stem_text(_pub_text(pub))
        if any(stem in text for stem in alias_stems):
            matched.append(pub)
        else:
            nonmatched.append(pub)

    coverage = 100.0 * len(matched) / len(pubs) if pubs else 0.0

    # Resolve references only when title+abstract coverage is below threshold
    if (
        coverage < settings.gate_ref_min_title_abstract_match_pct
        and settings.gate_ref_resolution_enabled
        and nonmatched
    ):
        ref_ids: list[str] = []
        for pub in nonmatched:
            refs = pub.get("reference_ids") or []
            if isinstance(refs, list):
                ref_ids.extend(str(r) for r in refs if r)
        # dedupe while preserving order
        seen: set[str] = set()
        unique_refs: list[str] = []
        for rid in ref_ids:
            if rid not in seen:
                seen.add(rid)
                unique_refs.append(rid)

        cap = settings.gate_ref_max_unique_ids
        batch = settings.gate_ref_batch_size
        ref_titles: dict[str, str] = {}
        from dataset_agent.adapters.dimensions_dsl import sanitize_alias

        for i in range(0, min(len(unique_refs), cap), batch):
            chunk = unique_refs[i : i + batch]
            id_list = ", ".join(f'"{sanitize_alias(rid)}"' for rid in chunk)
            ref_q = f"search publications where id in [{id_list}] return publications[id+title]"
            try:
                ref_result = dsl_client.query(ref_q)
                for rp in publications_from_result(ref_result):
                    pid = str(rp.get("id", "")).strip()
                    t = publication_title(rp)
                    if pid and t:
                        ref_titles[pid] = t
            except Exception as exc:
                logger.warning("Reference resolution batch failed: %s", exc)
            # tiny throttle between reference batches
            time.sleep(0.1)

        # Promote pubs whose references match an alias stem
        still_nonmatched: list[dict] = []
        for pub in nonmatched:
            refs = pub.get("reference_ids") or []
            promoted = False
            if isinstance(refs, list):
                for rid in refs:
                    title = ref_titles.get(str(rid))
                    if title and any(stem in _stem_text(title) for stem in alias_stems):
                        matched.append(pub)
                        promoted = True
                        break
            if not promoted:
                still_nonmatched.append(pub)
        nonmatched = still_nonmatched

    # Classify matched pubs (sampled/capped)
    fp_count = 0
    genuine_count = 0
    fp_hits: list[dict[str, str]] = []
    max_classify = max(1, settings.optimize_abstract_classify_max_calls or 20)

    for pub in matched[:max_classify]:
        matched_alias = _find_matched_alias(pub, aliases)
        if not matched_alias:
            genuine_count += 1
            continue

        snippet = _extract_snippet(_pub_text(pub), matched_alias)
        title = publication_title(pub)

        if agent is None:
            genuine_count += 1
            continue

        try:
            genuine = classify_alias_mention(
                agent,
                dataset_name=dataset_name,
                alias=matched_alias,
                snippet=snippet,
                title=title,
            )
        except Exception as exc:
            logger.warning("classify_alias_mention failed: %s", exc)
            genuine = True

        if genuine:
            genuine_count += 1
        else:
            fp_count += 1
            fp_hits.append(
                {
                    "publication_id": str(pub.get("id") or ""),
                    "matched_alias": matched_alias,
                    "snippet": snippet,
                }
            )

    # Assume unclassified matched pubs are genuine (conservative)
    unclassified = max(0, len(matched) - max_classify)
    genuine_count += unclassified

    matched_count = len(matched)
    fp_rate_pct = 100.0 * fp_count / matched_count if matched_count else 0.0

    return {
        "fp_rate_pct": fp_rate_pct,
        "genuine_count": genuine_count,
        "fp_count": fp_count,
        "matched_count": matched_count,
        "total_pubs": len(pubs),
        "coverage_pct": coverage,
        "for_clause": for_clause,
        "query": q,
        "publications_total": total,
        "fp_hits": fp_hits,
    }


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
        for_clause: str | None = None,
    ) -> LiteratureGateResult:
        _ = (sample_size, llm_batch_size)  # kept for signature compat
        logger.info(
            "Literature gate Dimensions: assessing dataset=%r",
            record.main_dataset_name,
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

        fc = for_clause or _build_for_clause_from_record(record)
        if not fc:
            logger.warning("Dimensions: empty for_clause; gate accepts")
            return LiteratureGateResult(True, 1.0, {"skip": "empty_for_clause"})

        aliases = [record.main_dataset_name] + list(record.dataset_names or [])
        try:
            ev = evaluate_for_clause_sync(
                for_clause=fc,
                aliases=aliases,
                dataset_name=record.main_dataset_name,
                settings=self._settings,
                agent=self._agent,
            )
        except Exception as exc:
            logger.exception("Dimensions gate evaluator error")
            return LiteratureGateResult(False, 0.0, {"error": str(exc)})

        if ev.get("error"):
            logger.warning("Dimensions evaluator error: %s", ev["error"])
            return LiteratureGateResult(
                False,
                0.0,
                {"error": ev["error"], "query": ev.get("query")},
            )

        fp_rate = float(ev.get("fp_rate_pct") or 0.0)
        indicator = max(0.0, 1.0 - fp_rate / 100.0)
        thr = self._settings.literature_threshold
        passed = indicator >= thr

        logger.info(
            "Dimensions gate result: fp_rate_pct=%.2f indicator=%.2f threshold=%.2f passed=%s",
            fp_rate,
            indicator,
            thr,
            passed,
        )

        detail = dict(ev)
        detail["passed"] = passed
        detail["indicator"] = indicator
        detail["threshold"] = thr
        # Backward-compatible keys for consumers that expect legacy detail shape
        detail.setdefault("validation_details", [])
        detail.setdefault("lexical_details", [])
        detail.setdefault("publications_analyzed", ev.get("matched_count") or 0)
        detail.setdefault("publications_valid", ev.get("genuine_count") or 0)
        return LiteratureGateResult(passed=passed, indicator=indicator, detail=detail)


def literature_gate_from_settings(
    settings: Settings, agent: AgentPort | None = None, dsl_port: Any = None
) -> LiteratureGatePort:
    _ = dsl_port  # reserved for future async evaluator wiring
    if settings.literature_gate == "dimensions":
        return DimensionsLiteratureGate(settings, agent=agent)
    return NoOpLiteratureGate()

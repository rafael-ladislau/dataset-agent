"""Refine dataset_names so they do not embed sponsor/org tokens already in flag_terms."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from dataset_agent.application import prompts
from dataset_agent.adapters.tools import EMIT_DATASET_NAMES, EMIT_FLAG_TERMS

if TYPE_CHECKING:
    from dataset_agent.domain.ports import AgentPort

logger = logging.getLogger(__name__)


def _dedupe_ci_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in items:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _remove_multiword_flags(text: str, flags: list[str]) -> str:
    s = text
    for f in sorted((x for x in flags if x and " " in x.strip()), key=len, reverse=True):
        s = re.sub(re.escape(f.strip()), "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" -–—,\t")


def _drop_tokens_matching_flags(text: str, flags: list[str]) -> str:
    """Remove whole tokens that match a single-token flag (case-insensitive)."""
    flag_lower = {
        f.strip().lower()
        for f in flags
        if f and len(f.strip()) > 0 and " " not in f.strip()
    }
    if not flag_lower:
        return text.strip()
    parts = re.split(r"(\s+|[-–—])", text)
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        if re.fullmatch(r"\s+", p) or p in "-–—":
            out.append(p)
            continue
        token = p.strip()
        if not token:
            continue
        tl = token.lower()
        if tl in flag_lower:
            continue
        out.append(p)
    merged = "".join(out)
    return re.sub(r"\s+", " ", merged).strip(" -–—,\t")


def heuristic_strip_flags_from_dataset_names(
    dataset_names: list[str],
    flag_terms: list[str],
) -> list[str]:
    """Fallback: strip org/sponsor tokens from each alias."""
    flags = [f for f in flag_terms if f and str(f).strip()]
    if not flags:
        return list(dataset_names)
    out: list[str] = []
    for name in dataset_names:
        if not name or not str(name).strip():
            continue
        s = _remove_multiword_flags(str(name).strip(), flags)
        s = _drop_tokens_matching_flags(s, flags)
        s = re.sub(r"\s+", " ", s).strip(" -–—,\t")
        if len(s) >= 2:
            out.append(s)
    return _dedupe_ci_preserve_order(out)


def refine_dataset_names_with_llm(
    agent: AgentPort,
    main_dataset_name: str,
    description: str,
    dataset_names: list[str],
    flag_terms: list[str],
) -> list[str]:
    """
    Ask the LLM to return dataset_names suitable for literature search: product titles
    and acronyms only, without repeating organization names already listed in flag_terms.

    Uses ``get_structured`` with the ``emit_dataset_names`` tool so the model is forced
    to return valid JSON matching the schema — no regex parsing required.
    On failure, falls back to :func:`heuristic_strip_flags_from_dataset_names`.
    """
    prompt = prompts.refine_dataset_aliases_prompt(
        main_dataset_name=main_dataset_name,
        description=description,
        dataset_names=dataset_names,
        flag_terms=flag_terms,
    )
    try:
        result = agent.get_structured(prompt, EMIT_DATASET_NAMES)
    except Exception as e:
        logger.warning("refine_dataset_names: LLM call failed: %s", e)
        return heuristic_strip_flags_from_dataset_names(dataset_names, flag_terms)

    names = result.get("dataset_names")
    if isinstance(names, list):
        parsed = [x.strip() for x in names if isinstance(x, str) and x.strip()][:25]
        if parsed:
            cleaned = _dedupe_ci_preserve_order(parsed)
            logger.info(
                "refine_dataset_names: LLM returned %s names (was %s)",
                len(cleaned),
                len(dataset_names),
            )
            return cleaned

    logger.warning("refine_dataset_names: empty/invalid result; using heuristic strip")
    return heuristic_strip_flags_from_dataset_names(dataset_names, flag_terms)


def refine_flag_terms_with_llm(
    agent: AgentPort,
    main_dataset_name: str,
    description: str,
    flag_terms: list[str],
) -> list[str]:
    """
    Prune flag_terms for literature search: drop redundant parent/sub-agency duplication.

    Uses ``get_structured`` with the ``emit_flag_terms`` tool so the model is forced
    to return valid JSON matching the schema — no regex parsing required.
    On failure, returns a case-insensitive dedupe of the input list.
    """
    prompt = prompts.refine_flag_terms_prompt(
        main_dataset_name=main_dataset_name,
        description=description,
        flag_terms=flag_terms,
    )
    try:
        result = agent.get_structured(prompt, EMIT_FLAG_TERMS)
    except Exception as e:
        logger.warning("refine_flag_terms: LLM call failed: %s", e)
        return _dedupe_ci_preserve_order(list(flag_terms))

    terms = result.get("flag_terms")
    if isinstance(terms, list):
        parsed = [x.strip() for x in terms if isinstance(x, str) and x.strip()][:15]
        if parsed:
            cleaned = _dedupe_ci_preserve_order(parsed)
            logger.info(
                "refine_flag_terms: LLM returned %s terms (was %s)",
                len(cleaned),
                len(flag_terms),
            )
            return cleaned

    logger.warning("refine_flag_terms: empty/invalid result; using deduped input")
    return _dedupe_ci_preserve_order(list(flag_terms))

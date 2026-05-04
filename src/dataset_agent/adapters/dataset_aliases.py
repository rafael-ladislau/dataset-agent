"""Refine dataset_names so they do not embed sponsor/org tokens already in flag_terms."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Literal

from dataset_agent.application import prompts
from dataset_agent.adapters.tools import (
    EMIT_DATASET_NAMES,
    EMIT_FLAG_TERMS,
    EMIT_SUBDATASET_ALIASES,
)

if TYPE_CHECKING:
    from dataset_agent.domain.ports import AgentPort, DimensionsDslPort

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


def validate_no_flag_alias_overlap(
    aliases: list[str],
    flag_terms: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Rule 1: drop aliases whose text equals a flag_term (case-insensitive).

    Returns ``(clean_aliases, clean_flag_terms, conflicts_removed)``.
    """
    flags_stripped = [str(f).strip() for f in flag_terms if f and str(f).strip()]
    flag_lower = {f.lower() for f in flags_stripped}
    clean_aliases: list[str] = []
    conflicts: list[str] = []
    for a in aliases:
        if not isinstance(a, str) or not a.strip():
            continue
        s = a.strip()
        if s.lower() in flag_lower:
            conflicts.append(s)
            continue
        clean_aliases.append(s)
    return clean_aliases, flags_stripped, conflicts


def detect_subdataset_aliases(
    agent: AgentPort,
    main_dataset_name: str,
    aliases: list[str],
) -> tuple[list[str], list[str]]:
    """Use structured LLM output to drop version/sub-product aliases (see spec §2.4)."""
    src = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
    if len(src) < 2:
        return list(src), []

    prompt = prompts.subdataset_aliases_prompt(main_dataset_name, src)
    try:
        result = agent.get_structured(prompt, EMIT_SUBDATASET_ALIASES)
    except Exception as e:
        logger.warning("detect_subdataset_aliases: LLM call failed: %s", e)
        return list(src), []

    keep_raw = result.get("keep_aliases")
    remove_raw = result.get("remove_aliases")
    if not isinstance(keep_raw, list):
        return list(src), []

    kept = [str(x).strip() for x in keep_raw if isinstance(x, str) and str(x).strip()][:40]
    removed = (
        [str(x).strip() for x in remove_raw if isinstance(x, str) and str(x).strip()][:40]
        if isinstance(remove_raw, list)
        else []
    )
    if not kept:
        return list(src), removed

    remove_l = {r.lower() for r in removed}
    merged = _dedupe_ci_preserve_order([k for k in kept if k.lower() not in remove_l])
    for a in src:
        if a.lower() in remove_l:
            continue
        if a.lower() not in {x.lower() for x in merged}:
            merged.append(a)

    main = main_dataset_name.strip()
    if main.lower() not in {x.lower() for x in merged}:
        merged.insert(0, main)

    return _dedupe_ci_preserve_order(merged), removed


async def classify_alias_risk(
    dsl_port: DimensionsDslPort,
    alias: str,
    full_name: str,
    *,
    search_in: str = "full_data",
    ratio_threshold: float = 10.0,
) -> Literal["safe", "risky"]:
    """10× Dimensions count rule: bare acronym risky if count(alias) > ratio * count(full_name)."""
    from dataset_agent.adapters.dimensions_dsl import run_alias_count

    a = (alias or "").strip()
    fn = (full_name or "").strip()
    if not a:
        return "safe"
    c_alias = await run_alias_count(dsl_port, a, search_in=search_in)
    if not fn:
        return "risky" if c_alias > 50_000 else "safe"
    c_full = await run_alias_count(dsl_port, fn, search_in=search_in)
    if c_full <= 0:
        return "risky" if c_alias > 0 else "safe"
    if c_alias > ratio_threshold * float(c_full):
        return "risky"
    return "safe"

"""Tests for dataset_names refinement vs flag_terms."""

from __future__ import annotations

import asyncio

from dataset_agent.adapters.dataset_aliases import (
    classify_alias_risk,
    detect_subdataset_aliases,
    heuristic_strip_flags_from_dataset_names,
    refine_dataset_names_with_llm,
    refine_flag_terms_with_llm,
    validate_no_flag_alias_overlap,
)


# ---------------------------------------------------------------------------
# Minimal stub that satisfies AgentPort for these unit tests.
# get_structured(prompt, result_tool) -> dict  (the new structured interface)
# ---------------------------------------------------------------------------

class _EmptyAgent:
    """Returns an empty dict — triggers fallback paths."""
    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        return {}


class _ErrorAgent:
    """Raises on every call — triggers exception-fallback paths."""
    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        raise RuntimeError("simulated failure")


class _OkNamesAgent:
    """Returns a valid dataset_names payload."""
    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        return {"dataset_names": ["O*NET", "ONET", "Occupational Information Network"]}


class _OkFlagAgent:
    """Returns a valid flag_terms payload."""
    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        return {
            "flag_terms": [
                "DOL",
                "Department of Labor",
                "Bureau of Labor Statistics",
                "BLS",
            ]
        }


# ---------------------------------------------------------------------------
# Heuristic (no LLM) tests — unchanged behaviour
# ---------------------------------------------------------------------------

def test_heuristic_strips_bls_dol_tokens_from_onet_style_aliases() -> None:
    flags = [
        "BLS",
        "Bureau of Labor Statistics",
        "DOL",
        "Department of Labor",
        "U.S. Bureau of Labor Statistics",
        "U.S. Department of Labor",
    ]
    names = [
        "Bls O*net",
        "Dol O*net",
        "Occupational Information Network Data - O*net",
    ]
    out = heuristic_strip_flags_from_dataset_names(names, flags)
    assert "Bls" not in " ".join(out)
    assert "Dol" not in " ".join(out)
    assert any("O*net" in x or "O*NET" in x for x in out)


# ---------------------------------------------------------------------------
# refine_dataset_names_with_llm
# ---------------------------------------------------------------------------

def test_refine_dataset_names_falls_back_when_agent_returns_empty() -> None:
    flags = ["BLS", "DOL"]
    names = ["Bls O*net"]
    out = refine_dataset_names_with_llm(
        _EmptyAgent(),  # type: ignore[arg-type]
        "Occupational Information Network - O*NET",
        "desc",
        names,
        flags,
    )
    assert len(out) >= 1
    assert all("bls" not in x.lower() for x in out if x)


def test_refine_dataset_names_falls_back_on_exception() -> None:
    flags = ["BLS", "DOL"]
    names = ["Bls O*net"]
    out = refine_dataset_names_with_llm(
        _ErrorAgent(),  # type: ignore[arg-type]
        "Occupational Information Network - O*NET",
        "desc",
        names,
        flags,
    )
    assert len(out) >= 1


def test_refine_dataset_names_uses_structured_result() -> None:
    out = refine_dataset_names_with_llm(
        _OkNamesAgent(),  # type: ignore[arg-type]
        "X",
        "d",
        ["Bls O*net"],
        ["BLS"],
    )
    assert "O*NET" in out
    assert "ONET" in out


# ---------------------------------------------------------------------------
# refine_flag_terms_with_llm
# ---------------------------------------------------------------------------

def test_refine_flag_terms_falls_back_to_deduped_input_when_empty() -> None:
    inp = ["DOL", "dol", "BLS"]
    out = refine_flag_terms_with_llm(
        _EmptyAgent(),  # type: ignore[arg-type]
        "O*NET",
        "desc",
        inp,
    )
    assert len(out) == 2


def test_refine_flag_terms_falls_back_on_exception() -> None:
    inp = ["DOL", "BLS"]
    out = refine_flag_terms_with_llm(
        _ErrorAgent(),  # type: ignore[arg-type]
        "O*NET",
        "desc",
        inp,
    )
    assert set(out) == {"DOL", "BLS"}


def test_validate_no_flag_alias_overlap_strips_exact_matches() -> None:
    aliases, flags, removed = validate_no_flag_alias_overlap(
        ["CPS", "Current Population Survey", "Census Bureau"],
        ["Census Bureau", "BLS"],
    )
    assert "Census Bureau" not in aliases
    assert "Census Bureau" in removed
    assert "CPS" in aliases
    assert flags == ["Census Bureau", "BLS"]


class _SubdatasetAgent:
    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        return {"keep_aliases": ["NLSY", "Other"], "remove_aliases": ["NLSY79", "NLSY97"]}


def test_detect_subdataset_aliases_merges_back_unremoved() -> None:
    out, removed = detect_subdataset_aliases(
        _SubdatasetAgent(),  # type: ignore[arg-type]
        "NLSY",
        ["NLSY", "NLSY79", "NLSY97", "Other"],
    )
    assert "NLSY79" not in out
    assert "NLSY97" not in out
    assert "Other" in out
    assert set(removed) >= {"NLSY79", "NLSY97"}


def test_classify_alias_risk_uses_counts() -> None:
    class _CountPort:
        def __init__(self) -> None:
            self.n = 0

        async def execute_dsl(self, dsl: str) -> object:
            self.n += 1
            from types import SimpleNamespace

            if self.n == 1:
                return SimpleNamespace(count_total=500_000)
            return SimpleNamespace(count_total=5_000)

    async def _run() -> None:
        port = _CountPort()
        r = await classify_alias_risk(port, "ACS", "American Community Survey")
        assert r == "risky"

    asyncio.run(_run())


def test_refine_flag_terms_uses_structured_result() -> None:
    noisy = [
        "DOL",
        "Department of Labor",
        "ETA",
        "Employment and Training Administration",
        "U.S. Department of Labor",
    ]
    out = refine_flag_terms_with_llm(
        _OkFlagAgent(),  # type: ignore[arg-type]
        "Occupational Information Network",
        "desc",
        noisy,
    )
    assert "ETA" not in out
    assert "DOL" in out

"""Tests for dataset_names refinement vs flag_terms."""

from __future__ import annotations

import json

from dataset_agent.adapters.dataset_aliases import (
    _parse_dataset_names_json,
    heuristic_strip_flags_from_dataset_names,
    refine_dataset_names_with_llm,
)


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


def test_parse_dataset_names_json_extracts_list() -> None:
    raw = """Here is the result:
{"dataset_names": ["O*NET", "ONET"]}
"""
    assert _parse_dataset_names_json(raw) == ["O*NET", "ONET"]


def test_parse_dataset_names_json_invalid_returns_none() -> None:
    assert _parse_dataset_names_json("not json") is None


def test_refine_dataset_names_falls_back_when_mock_agent_fails() -> None:
    class _BadAgent:
        def get_information(self, prompt: str) -> str:
            return "not valid json {{{{"

    flags = ["BLS", "DOL"]
    names = ["Bls O*net"]
    out = refine_dataset_names_with_llm(
        _BadAgent(),  # type: ignore[arg-type]
        "Occupational Information Network - O*NET",
        "desc",
        names,
        flags,
    )
    assert len(out) >= 1
    assert all("bls" not in x.lower() for x in out if x)


def test_refine_dataset_names_uses_json_when_valid() -> None:
    class _OkAgent:
        def get_information(self, prompt: str) -> str:
            return json.dumps(
                {"dataset_names": ["O*NET", "ONET", "Occupational Information Network"]}
            )

    out = refine_dataset_names_with_llm(
        _OkAgent(),  # type: ignore[arg-type]
        "X",
        "d",
        ["Bls O*net"],
        ["BLS"],
    )
    assert "O*NET" in out
    assert "ONET" in out

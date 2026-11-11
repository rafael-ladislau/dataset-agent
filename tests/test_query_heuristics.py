"""Tests for query optimization heuristics."""

from __future__ import annotations

import asyncio

from dataset_agent.adapters.query_heuristics import (
    _SUFFIX_CANDIDATES,
    apply_short_acronym_heuristic,
    check_suffix_necessity,
    detect_rhetorical_name,
    detect_sub_products,
    head_acronym,
    generate_multilingual_aliases,
    is_short_acronym,
    platform_exclude_suggestions,
    rhetorical_exclusions,
    strip_accents,
)


def test_is_short_acronym_five_char_not_short() -> None:
    assert is_short_acronym("CPS-X") is False
    assert is_short_acronym("CPS") is True


def test_apply_short_acronym_moves_to_risky() -> None:
    st, rk = apply_short_acronym_heuristic(
        ["Current Population Survey", "CPS"],
        [],
    )
    assert any(x.lower() == "cps" for x in rk)
    assert all(x.lower() != "cps" for x in st)


def test_detect_rhetorical_all_of_us() -> None:
    assert detect_rhetorical_name("All of Us") is True
    assert detect_rhetorical_name("American Community Survey") is False


def test_rhetorical_exclusions_contains_templates() -> None:
    ex = rhetorical_exclusions("All of Us")
    assert any("benefits" in x.lower() for x in ex)


def test_platform_exclude_linkedin() -> None:
    out = platform_exclude_suggestions("https://www.linkedin.com/economicgraph")
    assert any("social" in x.lower() for x in out)


def test_strip_accents() -> None:
    assert strip_accents("café").lower() == "cafe"


class _NamesAgent:
    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        if result_tool.get("name") == "emit_sub_product_names":
            return {"names": ["QWI", "LODES"]}
        if result_tool.get("name") == "emit_multilingual_aliases":
            return {"aliases": ["Encuesta", "Enquête"]}
        return {}


def test_detect_sub_products(monkeypatch) -> None:
    monkeypatch.setattr(
        "dataset_agent.adapters.query_heuristics.make_request",
        lambda url: "status_code=200 preview='QWI LODES LEHD'",
    )
    out = detect_sub_products("https://lehd.ces.census.gov/", _NamesAgent())  # type: ignore[arg-type]
    assert "QWI" in out


def test_generate_multilingual_aliases() -> None:
    out = generate_multilingual_aliases("Current Population Survey", _NamesAgent())  # type: ignore[arg-type]
    assert "Encuesta" in out


def test_check_suffix_necessity_returns_suffixed_when_ratio_high() -> None:
    class _P:
        def __init__(self) -> None:
            self.i = 0

        async def execute_dsl(self, dsl: str) -> object:
            from types import SimpleNamespace

            self.i += 1
            if "dataset" in dsl.lower():
                return SimpleNamespace(count_total=10)
            return SimpleNamespace(count_total=5000)

    async def _go() -> None:
        p = _P()
        rec = await check_suffix_necessity(p, "All of Us", ratio_threshold=50.0)
        assert rec is not None
        assert "dataset" in rec.lower()

    asyncio.run(_go())


def test_check_suffix_necessity_uses_bare_count_param() -> None:
    """When bare_count is supplied, skip the redundant bare-alias Dimensions call."""
    calls: list[str] = []

    class _P:
        async def execute_dsl(self, dsl: str) -> object:
            from types import SimpleNamespace

            calls.append(dsl)
            if "dataset" in dsl.lower():
                return SimpleNamespace(count_total=10)
            return SimpleNamespace(count_total=9999)

    async def _go() -> None:
        p = _P()
        rec = await check_suffix_necessity(
            p, "All of Us", bare_count=5000, ratio_threshold=50.0
        )
        assert rec is not None
        assert "dataset" in rec.lower()
        # bare_count skips one Dimensions round-trip; only suffix candidates are queried.
        assert len(calls) == len(_SUFFIX_CANDIDATES)

    asyncio.run(_go())


def test_is_acronym_suffix_form() -> None:
    from dataset_agent.adapters.query_heuristics import is_acronym_suffix_form

    assert all(is_acronym_suffix_form(a) for a in ["WR dataset", "Wr dataset", "QWR program", "ONET data", "Wrp dataset"])
    assert not any(is_acronym_suffix_form(a) for a in
                   ["Life data", "Census data", "Ui Wage Records survey", "WR", "NHANES dataset", "O*NET data"])


def test_head_acronym_rule() -> None:
    assert head_acronym("ADD Health Study") == "ADD"
    assert head_acronym("Wrp Panel") == "Wrp"  # no vowels
    assert head_acronym("A1 Panel") == "A1"
    assert head_acronym("Add Health") is None  # has vowels, not upper case
    assert head_acronym("NHIS") is None  # single word
    assert head_acronym("12 Panel") is None  # no letter
    assert head_acronym("ABCDEFG Panel") is None  # too long
    assert head_acronym("A Panel") is None  # too short
    assert head_acronym("AB-C Panel") is None  # not alphanumeric


def test_mixed_case_short_acronym_heads() -> None:
    from dataset_agent.adapters.query_heuristics import is_acronym_suffix_form

    assert head_acronym("AoU Research Program") == "AoU"
    assert head_acronym("eICU Collaborative") == "eICU"
    assert is_acronym_suffix_form("AoU dataset") and is_acronym_suffix_form("eICU data")
    for plain in ("All of Us", "Add Health", "The Study", "Ohio Panel"):
        assert head_acronym(plain) is None
    assert not is_acronym_suffix_form("Add dataset") and not is_acronym_suffix_form("All data")
    assert not is_acronym_suffix_form("Alongside dataset")  # longer than 4 chars

"""Tests for Dimensions query variant builders."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from dataset_agent.adapters.query_builder import (
    build_for_clause,
    build_readable_for_clause,
    default_variant_build_order,
    drop_contradicting_excludes,
    run_variant,
)


def test_v1_safe_only_or_group() -> None:
    fc = build_for_clause(
        safe=["ADP payroll data", "ADP payroll"],
        variant="V1",
    )
    assert " OR " in fc
    assert fc.startswith("(")
    assert "ADP payroll data" in fc.replace("\\", "")


def test_v3_matches_spec_shape() -> None:
    fc = build_for_clause(
        safe=["ADP payroll data", "ADP National Employment Report"],
        variant="V3",
        risky=["ADP"],
        flag_terms=["payroll", "employment report", "workforce"],
    )
    assert " AND " in fc
    assert " OR " in fc
    assert "NOT" not in fc


def test_v4_appends_not_exclusions() -> None:
    fc = build_for_clause(
        safe=["ADP payroll data"],
        variant="V4",
        risky=["ADP"],
        flag_terms=["payroll", "workforce"],
        exclusion_terms=["adenosine diphosphate", "kinase"],
    )
    assert " NOT " in fc
    assert "adenosine diphosphate" in fc.replace("\\", "")


def test_tier_hybrid_or_arm() -> None:
    fc = build_for_clause(
        safe=["All of Us Research Program"],
        variant="V3",
        risky=["All of Us"],
        flag_terms=["precision medicine", "biobank"],
        tier_hybrid_terms=["Controlled Tier", "Registered Tier"],
        tier_hybrid_qualifier="All of Us",
    )
    assert "Controlled Tier" in fc.replace("\\", "") or "Registered" in fc.replace("\\", "")


def test_readable_strips_escapes() -> None:
    r = build_readable_for_clause(safe=["O*NET data"], variant="V1")
    assert "\\" not in r
    assert "O*NET" in r


def test_default_variant_order_skips_v2_by_default() -> None:
    seq = default_variant_build_order(
        safe=["Dataset Alpha Long", "Dataset Beta Long"],
        risky=["XRisky"],
        flag_terms=["f1"],
        exclusion_terms=["noise"],
    )
    labels = [v for v, _ in seq]
    assert labels == ["V1", "V3", "V4"]


def test_default_variant_includes_v2_when_asked() -> None:
    seq = default_variant_build_order(
        safe=["Dataset Longname"],
        risky=["XRisky"],
        flag_terms=["f"],
        exclusion_terms=[],
        include_v2_diagnostic=True,
    )
    assert "V2" in [v for v, _ in seq]


def test_run_variant_uses_port() -> None:
    class FakePort:
        def __init__(self) -> None:
            self.dsl: str | None = None

        async def execute_dsl(self, dsl: str) -> object:
            self.dsl = dsl
            return SimpleNamespace(
                count_total=42,
                publications=[
                    {"basics": {"title": "Paper One"}},
                    {"title": "Paper Two"},
                ],
            )

    async def _go() -> None:
        port = FakePort()
        out = await run_variant(port, "V1", '(\\"CPS\\")', limit=5)
        assert out["label"] == "V1"
        assert out["expected_count"] == 42
        assert out["top_titles"][0] == "Paper One"
        assert "limit 5" in (port.dsl or "")

    asyncio.run(_go())


def test_top_titles_from_result_integration_shape() -> None:
    from dataset_agent.adapters.dimensions_dsl import top_titles_from_result

    res = SimpleNamespace(
        publications=[
            {"basics": {"title": "  T1  "}},
            {"basics": {}},
        ]
    )
    assert top_titles_from_result(res, max_titles=5) == ["T1"]


_AOU_FLAGS = ["All of Us Research Program", "National Institutes of Health", "NIH"]


def test_v4_drops_flag_equal_to_risky_alias_aou_shape() -> None:
    kw = dict(safe=["AoU survey"], variant="V4", risky=["AoU", "All of Us Researcher Workbench"])
    clean = build_for_clause(flag_terms=_AOU_FLAGS, **kw)
    dup = build_for_clause(flag_terms=[*_AOU_FLAGS, "aou"], **kw)
    assert dup == clean
    assert clean.count("AoU") == 2  # "AoU survey" and the risky "AoU"


def test_v4_keeps_flags_when_every_flag_collides() -> None:
    fc = build_for_clause(safe=["X survey"], variant="V4", risky=["AoU", "X Workbench"], flag_terms=["AoU"])
    assert " AND " in fc  # never the ungated risky group


def test_flag_equal_to_safe_alias_is_kept() -> None:
    fc = build_for_clause(safe=["NIH"], variant="V4", risky=["AoU"], flag_terms=["NIH", "CDC"])
    assert fc.count("NIH") == 2


def test_exclude_guard_atus_shaped_terms() -> None:
    kept, dropped = drop_contradicting_excludes(
        ["American Time Use Survey", "BLS", "time-use", "Census Bureau", "ATUS-X", "IPUMS",
         "-atus", "species", "leisure", "genus"],
        names=["American Time Use Survey", "ATUS", "ATUS data"],
        flag_terms=["BLS", "Bureau of Labor Statistics", "Census Bureau", "IPUMS"],
    )
    assert kept == ["species", "leisure", "genus"]
    assert [d for d, _ in dropped] == [
        "American Time Use Survey", "BLS", "time-use", "Census Bureau", "ATUS-X", "IPUMS",
        "-atus",
    ]
    assert dict(dropped)["time-use"] == "American Time Use Survey"


def test_exclude_guard_acs_shaped_terms() -> None:
    kept, dropped = drop_contradicting_excludes(
        ["census", "pubs.acs.org", "chemistry"],
        names=["American Community Survey", "ACS"],
        flag_terms=["Census Bureau"],
    )
    assert dropped == [("census", "Census Bureau")]
    # A dot between two letters stays inside the token: "pubs.acs.org" does not contain "ACS".
    assert kept == ["pubs.acs.org", "chemistry"]


def test_exclude_guard_folds_case_and_plural_but_not_substrings() -> None:
    kept, dropped = drop_contradicting_excludes(
        ["STUDENT SURVEYS", "stent", "Student"],
        names=[], flag_terms=["Student Survey"],
    )
    assert [d for d, _ in dropped] == ["STUDENT SURVEYS", "Student"]
    assert kept == ["stent"]


def test_exclude_guard_empty_and_order() -> None:
    assert drop_contradicting_excludes(None, names=None, flag_terms=None) == ([], [])
    assert drop_contradicting_excludes([], names=["A"], flag_terms=["B"]) == ([], [])
    kept, _ = drop_contradicting_excludes(
        ["zeta", "alpha", "mid"], names=["Alpha Study"], flag_terms=[]
    )
    assert kept == ["zeta", "mid"]


def test_build_for_clause_still_emits_an_alias_in_exclusion_terms() -> None:
    fc = build_for_clause(
        safe=["Long Dataset Name"], variant="V4", risky=["RISK"], flag_terms=["NIH Study"],
        exclusion_terms=["RISK"],
    )
    assert "NOT" in fc and fc.endswith('(\\"RISK\\")')

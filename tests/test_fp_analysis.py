"""Tests for false-positive analysis helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from dataset_agent.adapters.dimensions_dsl import build_for_clause_count_dsl
from dataset_agent.adapters.fp_analysis import (
    classify_alias_mention,
    classify_title,
    derive_exclude_terms_from_fps,
    generate_domain_positive_keywords,
    generate_flag_terms_for_alias,
    identify_fp_domains,
    run_scope_comparison,
    scan_abstracts_for_aliases,
    score_title_signal3,
    web_search_alias_meanings,
)


def test_classify_title_matches_keywords() -> None:
    fp = {"bio": ["adenosine", "kinase"], "hr": ["hazard ratio"]}
    m = classify_title("ADP kinase activity and phosphorylation in mitochondria", fp)
    assert "bio" in m
    assert "kinase" in m["bio"]


def test_scan_abstracts_for_aliases_finds_substring() -> None:
    pubs = [
        {
            "id": "pub1",
            "abstract": "We use the Current Population Survey (CPS) for wages.",
        }
    ]
    hits = scan_abstracts_for_aliases(pubs, ["CPS", "Current Population Survey"])
    assert any(h["matched_alias"] == "CPS" for h in hits)


def test_build_for_clause_count_dsl_uses_for_clause() -> None:
    q = build_for_clause_count_dsl('(\\"ACS\\")', search_in="title_abstract_only", limit=1)
    assert "title_abstract_only" in q
    assert "ACS" in q or "\\" in q


def test_run_scope_comparison_ratio() -> None:
    class P:
        def __init__(self) -> None:
            self.n = 0

        async def execute_dsl(self, dsl: str) -> object:
            self.n += 1
            if "full_data" in dsl:
                return SimpleNamespace(count_total=100)
            return SimpleNamespace(count_total=10)

    async def _go() -> None:
        port = P()
        r = await run_scope_comparison(port, '(\\"X\\")')
        assert abs(r - 10.0) < 0.01

    asyncio.run(_go())


class _StructAgent:
    def get_structured(self, prompt: str, result_tool: dict) -> dict:
        name = result_tool.get("name", "")
        if name == "emit_fp_analysis":
            return {
                "domains": [
                    {"domain_name": "biochemistry", "indicator_terms": ["ATP", "kinase"]},
                ]
            }
        if name == "emit_flag_terms_for_alias":
            return {"flag_terms": ["payroll", "workforce"]}
        if name == "emit_domain_positive_keywords":
            return {"keywords": ["Census Bureau", "household survey"]}
        if name == "emit_alias_mention_classification":
            return {"is_genuine_dataset_reference": False, "confidence": 8, "reason": "off domain"}
        if name == "emit_exclude_terms":
            return {"exclude_terms": ["cardiology"], "reasoning": "medical FP"}
        if name == "emit_title_relevance":
            return {
                "mention_score": 6,
                "context_score": 5,
                "mentioned_term": "CPS",
                "reason": "borderline",
            }
        if name == "emit_web_alias_meanings":
            return {
                "domains": [
                    {
                        "domain_name": "medicine",
                        "candidate_exclude_terms": ["patient", "clinical trial"],
                    }
                ]
            }
        return {}


def test_identify_fp_domains_parses_domains() -> None:
    out = identify_fp_domains(_StructAgent(), "ADP Payroll", "ADP")  # type: ignore[arg-type]
    assert out.get("biochemistry") == ["ATP", "kinase"]


def test_generate_flag_terms_for_alias() -> None:
    out = generate_flag_terms_for_alias(_StructAgent(), "ADP Payroll", "ADP")  # type: ignore[arg-type]
    assert "payroll" in out


def test_generate_domain_positive_keywords() -> None:
    out = generate_domain_positive_keywords(_StructAgent(), "ACS")  # type: ignore[arg-type]
    assert "Census Bureau" in out


def test_classify_alias_mention_false() -> None:
    assert (
        classify_alias_mention(
            _StructAgent(),  # type: ignore[arg-type]
            dataset_name="X",
            alias="Y",
            snippet="z",
        )
        is False
    )


def test_derive_exclude_terms_from_fps() -> None:
    hits = [{"matched_alias": "a", "snippet": "noise about cardiology"}]
    out = derive_exclude_terms_from_fps(_StructAgent(), dataset_name="DS", fp_hits=hits)  # type: ignore[arg-type]
    assert "cardiology" in out


def test_score_title_signal3() -> None:
    out = score_title_signal3(
        _StructAgent(),  # type: ignore[arg-type]
        dataset_name="CPS",
        dataset_description="Household survey",
        title="Labor supply in CPS",
    )
    assert out["mention_score"] == 6


def test_web_search_alias_meanings(monkeypatch) -> None:
    monkeypatch.setattr(
        "dataset_agent.adapters.fp_analysis.web_search",
        lambda q: "- hit1\n  http://x\n  body",
    )
    out = web_search_alias_meanings(_StructAgent(), "NLX")  # type: ignore[arg-type]
    assert "medicine" in out
    assert "patient" in out["medicine"]

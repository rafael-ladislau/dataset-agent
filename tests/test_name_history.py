"""Former and successor names (renames, merges) added in optimize phase 1."""

from __future__ import annotations

from dataset_agent.adapters.name_history import HISTORY_TOOL
from tests.test_name_grounding import TITLE, _record, _run

NOTA_WEB = (
    "- Network of the Americas | EarthScope\n  The Network of the Americas (NOTA), formerly "
    "the Plate Boundary Observatory, spans 20 countries. UNAVCO merged with IRIS to form "
    "EarthScope Consortium in 2023."
)


def _history_calls(agent) -> list:
    return [c for c in agent.get_structured.call_args_list if c.args[1] is HISTORY_TOOL]


def _llm(history: dict, names: list[str] | None = None):
    def reply(prompt, tool, *a, **k):
        return history if tool is HISTORY_TOOL else {"names": names or []}
    return reply


def test_llm_history_adds_names_and_renamed_organizations(tmp_path, monkeypatch) -> None:
    rec = _record([TITLE, "Network of the Americas"], [])
    rec.flag_terms = ["UNAVCO"]
    history = {
        "dataset_names": [
            {"name": "Plate Boundary Observatory", "relation": "former", "year": 2018},
            {"name": "PBO", "relation": "former"},
            {"name": "Tiny Old Network", "relation": "former"},
        ],
        "organizations": [
            {"name": "EarthScope Consortium", "of": "UNAVCO", "relation": "successor"},
            {"name": "National Science Foundation", "of": "UNAVCO", "relation": "former"},
            {"name": "IRIS Consortium", "of": "Seismology Group", "relation": "merged_into"},
            {"name": "Incorporated Research Institutions for Seismology", "of": "UNAVCO",
             "relation": "merged_from"},
        ],
    }
    counts = {"Network of the Americas": 1011, "Plate Boundary Observatory": 2115,
              "Tiny Old Network": 2}
    out, agent = _run(
        tmp_path, monkeypatch, counts, aliases=[], suggested=[], llm_names=[], record=rec,
        web="no phrases here", llm=_llm(history),
    )
    pool = out.aliases.safe + out.aliases.risky
    assert "Plate Boundary Observatory" in pool
    assert "PBO" not in pool and "Tiny Old Network" not in pool
    assert "'Plate Boundary Observatory' (former, 2018; 2115 hits)" in out.notes
    assert out.flag_terms == ['UNAVCO'] and 'added organization name' not in out.notes


def test_name_history_can_be_disabled(tmp_path, monkeypatch) -> None:
    rec = _record([TITLE, "Network of the Americas"], [])
    _out, agent = _run(
        tmp_path, monkeypatch, {"Network of the Americas": 1011}, aliases=[], suggested=[],
        llm_names=[], record=rec, web=NOTA_WEB, optimize_name_history_enabled=False,
    )
    assert _history_calls(agent) == []


def test_name_history_does_not_run_when_the_title_has_hits(tmp_path, monkeypatch) -> None:
    # The title has hits, so name history is not asked at all; "All of Us" stays a name.
    import asyncio
    from unittest.mock import MagicMock

    from dataset_agent.application import query_optimization as qopt_mod
    from dataset_agent.application.query_optimization import QueryOptimizationUseCase
    from dataset_agent.domain.models import OptimizeRequest
    from dataset_agent.settings import Settings
    from tests.test_name_grounding import _KeyedDSL

    title = "All of Us NIH"
    rec = _record([title, "All of Us"], [])
    rec.main_dataset_name = title
    rec.flag_terms = ["All of Us Research Program"]
    history = {"dataset_names": [{"name": "Precision Medicine Initiative Cohort Program",
                                  "relation": "former", "year": 2016}], "organizations": []}
    counts = {"Precision Medicine Initiative Cohort Program": 300, "All of Us Research Program": 9000,
              "All of Us NIH": 86, "All of Us": 5000}
    monkeypatch.setattr(qopt_mod, "web_search", lambda q: "no phrases here")
    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync",
        lambda *a, **k: {"fp_rate_pct": 0.0, "fp_hits": [], "publications_total": 10,
                         "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                         "total_pubs": 10, "coverage_pct": 100.0, "query": "",
                         "for_clause": k.get("for_clause", "")},
    )
    settings = Settings(
        tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out", dimensions_api_key="k",
        optimize_max_dimensions_calls=60, optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0, optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=0, optimize_web_disambig_max_aliases=0,
    )
    research = MagicMock()
    research.execute.return_value = (rec, tmp_path / "r.json")
    agent = MagicMock()
    agent.get_structured.side_effect = _llm(history)
    uc = QueryOptimizationUseCase(dsl_port=_KeyedDSL(counts), research=research, agent=agent, settings=settings)
    out = asyncio.run(uc.execute(OptimizeRequest(dataset_name=title)))
    pool = out.aliases.safe + out.aliases.risky
    assert _history_calls(agent) == []
    assert "Precision Medicine Initiative Cohort Program" not in pool
    assert "All of Us" in pool
    assert "dropped organization alias" not in out.notes

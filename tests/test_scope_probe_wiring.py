"""Probe wiring: with a resource map, every probe's evidence goes through the scope check."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import OptimizeRequest, ResourceMap
from tests.test_fulltext_loop import CountDSL, FakeClient, make_settings


def _pub(pid, *, used=False, text="The Example Program aims to enrol people."):
    return {
        "publication_id": pid, "doi": None, "title": f"T {pid}", "citation_count": 1,
        "fulltext": {"status": "delivered", "source": "pmc", "reason": None}, "analyzed": True,
        "validated_datasets": [{"name": "X", "is_used": used, "is_mentioned": True, "text_quote": "q"}],
        "term_contexts": [{"term": "Example Program", "role": "alias", "part": "body", "text": text}],
    }


class Judge:
    def __init__(self):
        self.prompts = []

    def get_structured(self, prompt, _tool):
        self.prompts.append(prompt)
        return {"scope": "program_only"}


MAP = ResourceMap(program_names=["Example Program"], data_resource_names=["Example Workbench"])


def _uc(tmp_path, judge, resource_map):
    client = FakeClient([{"publications": [_pub("a"), _pub("u", used=True)]}])
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=make_settings(tmp_path), extraction_client=client,
                                  referent_agent=judge)
    uc._resource_map = resource_map
    uc._dataset_description = "A cohort."
    return uc


def _probe(uc):
    cand = QueryCandidate(main_dataset_name="Example Program", safe=["Example Program"])
    return asyncio.run(uc._probe_candidate(cand, OptimizeRequest(dataset_name="Example Program")))


def test_with_a_map_the_probe_runs_the_scope_check(tmp_path):
    judge = Judge()
    uc = _uc(tmp_path, judge, MAP)
    ev = _probe(uc)
    assert len(judge.prompts) == 1 and "Example Workbench" in judge.prompts[0]
    assert ev.n_program_only == 1 and ev.corrected_precision == 0.5
    assert uc._scope_calls == 1


def test_without_a_map_there_are_no_scope_calls(tmp_path):
    judge = Judge()
    uc = _uc(tmp_path, judge, None)
    ev = _probe(uc)
    assert judge.prompts == [] and ev.n_program_only is None and uc._scope_calls == 0


def test_generated_phrases_cover_each_program_name():
    from dataset_agent.adapters.scope_check import generated_data_phrases

    assert generated_data_phrases(["A B"]) == ["A B data", "A B dataset", "data from the A B"]


def test_window_starts_at_a_confirmed_first_release_year(tmp_path):
    uc = _uc(tmp_path, Judge(), MAP.model_copy(update={"first_release_year": 2021}))
    assert uc._dataset_years() == (2021, uc._settings.default_years_end)
    assert "year in [2021:" in uc._build_where_dsl('\\"x\\"')
    _probe(uc)
    assert uc._get_extraction().seeds[0]["years_range"]["start_year"] == 2021
    uc._resource_map = MAP  # no confirmed year: the default window
    assert uc._dataset_years()[0] == uc._settings.default_years_start
    uc._resource_map = MAP.model_copy(update={"first_release_year": 1990})
    assert uc._dataset_years()[0] == uc._settings.default_years_start  # never earlier


def test_with_a_map_program_names_are_gated_by_the_data_names(tmp_path, monkeypatch):
    from dataset_agent.application import query_optimization as qopt_mod
    from dataset_agent.domain.models import DatasetRecord, Group, YearsRange

    clauses: list[str] = []

    def fake_eval(*a, **k):
        clauses.append(k.get("for_clause", ""))
        return {"fp_rate_pct": 5.0, "genuine_count": 10, "fp_count": 0, "matched_count": 10,
                "total_pubs": 100, "coverage_pct": 100.0, "for_clause": k.get("for_clause", ""),
                "publications_total": 100, "fp_hits": []}

    monkeypatch.setattr(qopt_mod, "evaluate_for_clause_sync", fake_eval)
    rec = DatasetRecord(
        engine="dimensions", group=Group(name="default"), main_dataset_name="Example Program NIH",
        description="A cohort.", dataset_names=["Example Program NIH", "Example Cohort Study"],
        flag_terms=["Example Program", "National Agency"],
        years_range=YearsRange(start_year=2015, end_year=2026),
        resource_map=ResourceMap(program_names=["Example Program NIH", "Example Program"],
                                 data_resource_names=["Example Workbench", "Controlled Tier"]),
    )
    dsl = CountDSL()
    uc = QueryOptimizationUseCase(dsl_port=dsl, research=MagicMock(), agent=MagicMock(),
                                  settings=make_settings(tmp_path), extraction_client=FakeClient([{}]))

    async def stored(_req):
        return rec.model_copy(deep=True)

    uc._run_research = stored
    out = asyncio.run(uc.execute(OptimizeRequest(dataset_name="Example Program NIH")))
    assert out.success
    assert out.flag_terms == ["Example Workbench", "Controlled Tier"]
    assert "Example Program NIH" in out.aliases.risky and "Example Program" in out.aliases.risky
    assert "Example Program NIH" not in out.aliases.safe
    assert "Example Cohort Study" in out.aliases.safe
    counted = " ".join(dsl.queries)
    assert "Example Program data" in counted and "data from the Example Program" in counted
    assert "National Agency" not in out.for_clause
    assert out.resource_map is not None and out.scope_check_calls == 0
    assert "resource map: program" in out.notes
    assert out.program_datasets is not None
    assert {d.name for d in out.program_datasets.datasets} >= {"Example Workbench", "Controlled Tier"}
    assert "Datasets of the program" in out.notes


def test_seed_description_names_the_program_and_data_resources():
    from dataset_agent.application.query_optimization import resource_map_description

    text = resource_map_description("A cohort.", MAP)
    assert text == ("A cohort. Program names: Example Program. "
                    "Data resource names: Example Workbench.")
    assert resource_map_description("A cohort.", None) == "A cohort."

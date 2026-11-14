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

    assert generated_data_phrases(["A B"]) == ["A B data", "A B dataset", "A B participants",
                                              "data from the A B"]

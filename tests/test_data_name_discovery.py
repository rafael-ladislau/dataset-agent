"""Round 2, item 14: data-name discovery from documentation pages and used papers."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from dataset_agent.adapters.fulltext_evidence import build_evidence
from dataset_agent.adapters.resource_map import (
    discover_from_web,
    discovery_pages,
    literature_candidates,
    map_for_record,
)
from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import DatasetRecord, Group, ResourceMap, YearsRange
from tests.test_fulltext_loop import CountDSL, FakeClient, make_settings

PROGRAMS = ["Example Program"]


def _map():
    return ResourceMap(program_names=list(PROGRAMS), data_resource_names=["Example Workbench"],
                       cooccurrence={"Example Workbench": 40}, name_sources={"Example Workbench": "pages"})


def _search(q):
    return f"- result for {q}\n  https://docs.example.org/{abs(hash(q)) % 1000}\n  snippet"


def test_discovery_fetches_the_top_result_pages():
    fetched = []
    text, sources = discovery_pages("Example Program", search=_search,
                                    fetch=lambda u: fetched.append(u) or f"page {u}")
    assert len(fetched) == 5 and all(u in sources for u in fetched)
    assert sum(s.startswith("search: ") for s in sources) == 5 and "page https://" in text


def test_web_discovery_keeps_only_literal_names_with_cooccurrence():
    class Agent:
        def get_structured(self, _p, _t):
            return {"names": [{"name": "Controlled Tier", "kind": "data_resource"}],
                    "data_resource_names_in_pages": ["Registered Tier", "Invented Portal",
                                                     "Example Program"]}

    def count(clause):
        return 0 if "Registered Tier" in clause and "AND" in clause else 30

    rmap = _map()
    added = discover_from_web(
        Agent(), rmap, dataset_name="Example Program", description="", count=count,
        search=_search, fetch=lambda u: "Use the Controlled Tier or the Registered Tier.")
    assert added == ["Controlled Tier"]
    assert rmap.name_sources["Controlled Tier"] == "discovery"
    assert "Controlled Tier" not in rmap.specific_data_names  # generic words: gate only


def test_unsplit_datasets_see_no_discovery():
    searched = []
    rec = DatasetRecord(engine="dimensions", group=Group(name="default"),
                        main_dataset_name="Example Data Commons", description="",
                        dataset_names=[], flag_terms=[],
                        years_range=YearsRange(start_year=2015, end_year=2026))

    class Agent:
        def get_structured(self, _p, _t):
            return {"names": [{"name": "Example Data Commons", "kind": "data_resource"}]}

    m = map_for_record(Agent(), rec, count=lambda c: 5, pages="x", discover=True,
                       search=lambda q: searched.append(q) or "", fetch=lambda u: "")
    assert m is None and searched == []


def test_literature_candidates_need_two_papers_and_the_program_nearby():
    passages = {
        "a": ["Data were accessed through the Example Program Controlled Tier Dataset v7."],
        "b": ["We used the Controlled Tier Dataset of the Example Program."],
        "c": ["The United States census.", "Example Program and the United States."],
    }
    assert literature_candidates(passages, PROGRAMS, []) == ["Controlled Tier Dataset"]


def _pub(pid, text):
    return {"publication_id": pid, "doi": None, "title": "t", "citation_count": 1,
            "fulltext": {"status": "delivered", "source": "pmc", "reason": None}, "analyzed": True,
            "validated_datasets": [{"is_mentioned": True, "is_used": True, "text_quote": "q"}],
            "term_contexts": [{"term": "Example Program", "role": "alias", "part": "body",
                               "text": text}]}


def test_literature_discovery_adds_checked_names_to_the_query(tmp_path):
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=make_settings(tmp_path), extraction_client=FakeClient([{}]))
    uc._resource_map = _map()
    ev = build_evidence({"task_id": "T", "publications": [
        _pub("a", "Example Program Curated Data Repository files were used."),
        _pub("b", "the Curated Data Repository of the Example Program")]})
    cand = QueryCandidate(main_dataset_name="Example Program", safe=["Example Program data"],
                          risky=["Example Program"], flag_terms=["Example Workbench"])
    notes: list[str] = []

    async def go():
        return await uc._discover_from_literature(ev, cand, notes, 0)

    out = asyncio.run(go())
    assert out is not None and "Curated Data Repository" not in out.flag_terms
    assert uc._resource_map.name_sources["Curated Data Repository"] == "literature"
    assert any("literature discovery" in n for n in notes)

"""S4: full text decides the release of an AI-model seed (fake extraction and judge)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dataset_agent.adapters.ai_model_research import HubEvidence
from dataset_agent.adapters.extraction_client import TopupResult
from dataset_agent.application import query_optimization as qopt_mod
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import OptimizeRequest
from dataset_agent.settings import Settings
from tests.test_ai_model_queries import (
    PHASENET_LLM,
    PHASENET_TITLE,
    CorpusDSL,
    _fake_eval,
    _phasenet_regate_docs,
)


def _pub(i, genuine):
    row = {"name": "PhaseNet", "is_mentioned": genuine, "is_used": genuine, "text_quote": "q",
           "match_type": "llm_validation", "usage_type": "inference" if genuine else None}
    return {
        "publication_id": f"pub.{i}", "doi": None, "title": f"T{i}", "citation_count": 100 - i,
        "fulltext": {"status": "delivered", "source": "pmc", "reason": None},
        "analyzed": True, "validation_failed": False, "validated_ai_models": [row],
        "category_for": [], "term_contexts": [
            {"term": "PhaseNet", "role": "alias", "part": "body", "text": f"PhaseNet in T{i}"}],
    }


def _body(n, genuine):
    return {"task_id": "T", "probe": True, "seed_type": "ai_model",
            "publications": [_pub(i, i < genuine) for i in range(n)]}


class ModelClient:
    """Fake extraction: one llm-results body per probe, in order."""

    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.seeds: list[dict] = []
        self.kwargs: list[dict] = []

    def run_probe_topup(self, seed, *, target, batch_size, max_batches, pull_cap, **extra):
        self.seeds.append(seed)
        self.kwargs.append(dict(target=target, entity=extra.get("entity")))
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        n = len(body["publications"])
        return TopupResult(results=body, batches=1, attempted=n, delivered=n, target=target,
                           stop_reason="pool_exhausted", task_id=f"P{len(self.seeds)}")


@pytest.fixture
def patched(monkeypatch):
    state = SimpleNamespace(eval=_fake_eval())
    monkeypatch.setattr(qopt_mod, "fetch_hub_evidence", lambda _id: HubEvidence(
        hugging_face_id="x/phasenet", found=True, paper_titles=[PHASENET_TITLE]))
    monkeypatch.setattr(qopt_mod, "web_search", lambda _q: "")
    monkeypatch.setattr(qopt_mod, "evaluate_for_clause_sync", lambda *a, **k: state.eval(*a, **k))
    return state


def _run(tmp_path, client, *, referent_answer=True, **over):
    s = Settings(tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out",
                 dimensions_api_key="k", dimensions_rate_limit_seconds=0.0,
                 ai_model_plain_word_max_safe_hits=20_000, ai_model_max_mined_context_terms=0,
                 optimize_max_iterations=1, referent_votes=1, **over)
    agent = MagicMock()
    llm = dict(PHASENET_LLM, paper_titles=[], context_terms=["STEAD"])
    agent.get_structured.side_effect = lambda prompt, _tool: (
        {"is_genuine_dataset_reference": referent_answer(prompt) if callable(referent_answer)
         else referent_answer}
        if prompt.startswith("AI model:") else llm
    )
    uc = QueryOptimizationUseCase(dsl_port=CorpusDSL(_phasenet_regate_docs()), research=MagicMock(),
                                  agent=agent, settings=s, extraction_client=client)
    req = OptimizeRequest(dataset_name="PhaseNet", entity_type="ai_model", hugging_face_id="x/phasenet")
    return asyncio.run(uc.execute(req))


def test_full_text_off_never_probes(tmp_path, patched):
    client = ModelClient(_body(20, 20))
    out = _run(tmp_path, client, ai_model_fulltext_enabled=False)
    assert client.seeds == [] and out.ai_model_seed is not None


def test_full_text_at_the_bar_releases_the_shipped_seed(tmp_path, patched):
    client = ModelClient(_body(20, 17))
    out = _run(tmp_path, client, ai_model_fulltext_enabled=True)
    assert out.ai_model_seed is not None
    assert client.seeds == [out.ai_model_seed]  # the shipped seed, exactly
    assert client.kwargs[0]["entity"] == "ai_model"
    it = out.fulltext_iterations[-1]
    assert it.n_evaluated == 20 and it.fulltext_precision == 0.85
    assert "full text released the seed: 17 of 20 genuine (0.85" in out.notes
    assert out.confidence.value == "medium"


def test_full_text_below_the_bar_regates_once_then_withholds(tmp_path, patched):
    client = ModelClient(_body(20, 14), _body(20, 14))
    out = _run(tmp_path, client, ai_model_fulltext_enabled=True)
    assert len(client.seeds) == 2
    assert client.seeds[1]["flag_terms"] and "STEAD" in client.seeds[1]["flag_terms"]
    assert out.ai_model_seed is None and out.confidence.value == "low"
    assert "re-gated after full text" in out.notes
    assert "seed withheld: full text measured 14 of 20 genuine (0.70" in out.notes
    assert len(out.fulltext_iterations) == 2


def test_full_text_below_the_bar_regated_query_can_pass(tmp_path, patched):
    client = ModelClient(_body(20, 14), _body(20, 19))
    out = _run(tmp_path, client, ai_model_fulltext_enabled=True)
    assert out.ai_model_seed is not None and out.ai_model_seed == client.seeds[1]
    assert out.aliases.risky == ["PhaseNet"]
    assert out.ai_model_seed_count == out.ai_model_query_count


def test_too_few_papers_leave_the_abstract_decision(tmp_path, patched):
    client = ModelClient(_body(8, 2))
    out = _run(tmp_path, client, ai_model_fulltext_enabled=True)
    assert len(client.seeds) == 1 and out.ai_model_seed is not None
    assert "full text not decisive: 8 papers judged (fewer than 10)" in out.notes


def test_full_text_releases_a_seed_the_abstract_check_withheld(tmp_path, patched):
    patched.eval = _fake_eval([{"fp_rate_pct": 50.0, "fp_count": 6, "genuine_count": 6,
                                "judged_count": 12}])
    client = ModelClient(_body(20, 19))
    out = _run(tmp_path, client, ai_model_fulltext_enabled=True, ai_model_title_flag_enabled=False)
    assert out.ai_model_seed is not None
    assert "full text released the seed" in out.notes


def test_referent_rejections_count_against_the_release(tmp_path, patched):
    # The judge accepts 18 of 20; the referent check turns down 5 of them -> 13/20.
    client = ModelClient(_body(20, 18), _body(20, 18))
    out = _run(tmp_path, client, ai_model_fulltext_enabled=True,
               referent_answer=lambda p: not any(f"T{i}'" in p for i in range(5)))
    first = out.fulltext_iterations[0]
    assert first.raw_fulltext_precision == 0.9 and first.fulltext_precision == 0.65
    assert first.referent_rejected_count == 5

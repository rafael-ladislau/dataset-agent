"""Owner decisions 2026-11-14: the resource map is on by default."""

from dataset_agent.settings import Settings


def test_resource_map_is_on_by_default(monkeypatch):
    monkeypatch.delenv("DATASET_AGENT_RESOURCE_MAP_ENABLED", raising=False)
    assert Settings(_env_file=None).resource_map_enabled is True


def test_the_ai_model_path_never_builds_a_map(tmp_path, monkeypatch):
    from dataset_agent.application.query_optimization import QueryOptimizationUseCase
    from tests.test_ai_model_fulltext_release import ModelClient, _body, _run

    async def boom(*_a, **_k):
        raise AssertionError("resource map built on the AI-model path")

    monkeypatch.setattr(QueryOptimizationUseCase, "_research_resource_map", boom)
    from tests.test_ai_model_fulltext_release import _fake_eval, qopt_mod
    from dataset_agent.adapters.ai_model_research import HubEvidence
    from tests.test_ai_model_queries import PHASENET_TITLE

    state = _fake_eval()
    monkeypatch.setattr(qopt_mod, "fetch_hub_evidence", lambda _id: HubEvidence(
        hugging_face_id="x/phasenet", found=True, paper_titles=[PHASENET_TITLE]))
    monkeypatch.setattr(qopt_mod, "web_search", lambda _q: "")
    monkeypatch.setattr(qopt_mod, "evaluate_for_clause_sync", lambda *a, **k: state(*a, **k))
    out = _run(tmp_path, ModelClient(_body(20, 20)), resource_map_enabled=True)
    assert out.resource_map is None and out.program_datasets is None
    assert out.scope_check_calls in (None, 0)


def test_ai_model_full_text_and_referent_thinking_are_on_by_default(monkeypatch):
    for k in ("AI_MODEL_FULLTEXT_ENABLED", "REFERENT_THINKING_BUDGET", "REFERENT_TOP_P",
              "REFERENT_TOP_K", "REFERENT_TEMPERATURE"):
        monkeypatch.delenv(f"DATASET_AGENT_{k}", raising=False)
    s = Settings(_env_file=None)
    assert s.ai_model_fulltext_enabled is True
    # The values the reported AI-model figures were measured with (Qwen thinking recipe).
    assert s.referent_thinking_budget == 4096
    assert (s.referent_temperature, s.referent_top_p, s.referent_top_k) == (0.6, 0.95, 20)


def test_referent_thinking_falls_back_without_the_anthropic_style(tmp_path, caplog):
    from dataset_agent.bootstrap import build_referent_agent

    s = Settings(_env_file=None, llm_provider="openai_compatible", llm_api_style="openai",
                 openai_compatible_base_url="http://127.0.0.1:1", openai_compatible_model="m",
                 referent_thinking_budget=4096)
    agent = build_referent_agent(s)  # no error: the main agent, with a warning
    assert agent is not None
    assert any("thinking" in r.message for r in caplog.records)

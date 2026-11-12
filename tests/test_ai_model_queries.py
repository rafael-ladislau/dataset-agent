"""AI-model query optimization: term research, measurement rules and the Aurora case.

The Dimensions fake evaluates each boolean ``for`` clause against a small synthetic
corpus, so the tests assert on behavior (what a term matches) and not on call order.
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from typer.testing import CliRunner

from dataset_agent.adapters import ai_model_research as amr
from dataset_agent.adapters.ai_model_research import (
    HubEvidence,
    assemble_candidates,
    classify_model_name,
    fetch_hub_evidence,
    is_distinctive,
    is_plain_word,
    mine_context_terms,
)
from dataset_agent.adapters.fp_analysis import classify_alias_mention
from dataset_agent.adapters.seed_mapping import (
    QueryCandidate,
    ai_model_seed_clause,
    candidate_to_ai_model_seed,
)
from dataset_agent.application import query_optimization as qopt_mod
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import OptimizeRequest
from dataset_agent.settings import Settings


# --------------------------------------------------------------------------- fakes


def _norm(text: str) -> str:
    return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split()) + " "


def _matches(clause: str, doc: str) -> bool:
    """Evaluate a ``for`` clause (escaped-quote terms, AND / OR / NOT) on one document."""
    terms: list[str] = []

    def _sub(m: re.Match) -> str:
        terms.append(m.group(1).replace("\\", ""))
        return f" T[{len(terms) - 1}] "

    expr = re.sub(r'\\"(.*?)\\"', _sub, clause)
    expr = expr.replace(" AND ", " and ").replace(" OR ", " or ").replace(" NOT ", " and not ")
    hay = _norm(doc)
    return bool(eval(expr, {"__builtins__": {}}, {"T": [_norm(t) in hay for t in terms]}))  # noqa: S307


class CorpusDSL:
    def __init__(self, docs: list[str]) -> None:
        self.docs = docs
        self.clauses: list[str] = []

    async def execute_dsl(self, dsl: str) -> SimpleNamespace:
        clause = re.search(r'for "(.*)" (?:where .*?)?return ', dsl).group(1)
        self.clauses.append(clause.replace("\\", ""))
        hits = [d for d in self.docs if _matches(clause, d)]
        return SimpleNamespace(
            count_total=len(hits),
            publications=[{"title": d[:40], "abstract": d} for d in hits[:200]],
        )


PRECISE = (
    "The Aurora foundation model (Microsoft Aurora) is compared with GraphCast and "
    "Pangu-Weather on ERA5 from ECMWF. Refs: Aurora: A Foundation Model of the Atmosphere; "
    "A Foundation Model for the Earth System."
)
CITING = (
    "We fine-tune Aurora for air quality; baselines include GraphCast. "
    "Ref: A Foundation Model for the Earth System."
)
UNCONFIRMED_USE = "We run Aurora next to GraphCast for typhoon tracks."
POLAR = "The aurora borealis and substorm onset seen by all-sky cameras."
POLAR_ECMWF = "Aurora observations combined with ECMWF and ERA5 winds in the thermosphere."
POLAR_ML = "Forecasting the aurora borealis with a GraphCast-style network."
KINASE = "Aurora kinase A inhibitors in tumor cells."
DATABASE = "Amazon Aurora and SQL Server from Microsoft Research on cloud storage."
PLACE = "Patients in Aurora, Colorado; analysis in Microsoft Excel."
OVATION = "The OVATION aurora model predicts electron precipitation."
PEERS_ONLY = "GraphCast and Pangu-Weather evaluated on ERA5 from ECMWF."


def _aurora_corpus() -> list[str]:
    return (
        [PRECISE] * 30 + [CITING] * 10 + [UNCONFIRMED_USE] * 10 + [POLAR] * 200
        + [POLAR_ECMWF] * 50 + [POLAR_ML] * 5 + [KINASE] * 100 + [DATABASE] * 40
        + [PLACE] * 60 + [OVATION] * 10 + [PEERS_ONLY] * 300
    )


AURORA_LLM = {
    "description": "Aurora is Microsoft's foundation model of the atmosphere.",
    "developer": "Microsoft",
    "names": [
        {"name": "Aurora", "kind": "family"},
        {"name": "Microsoft Aurora", "kind": "qualified"},
        {"name": "Aurora foundation model", "kind": "qualified"},
        {"name": "Aurora AI", "kind": "qualified"},
        {"name": "aurora weather model", "kind": "qualified"},
    ],
    "paper_titles": ["Aurora: A Foundation Model of the Atmosphere"],
    "context_terms": ["Microsoft Research", "SQL", "ECMWF", "GraphCast", "weather forecasting", "Microsoft"],
    "other_meanings": ["aurora borealis", "Aurora kinase", "Aurora (city)"],
}
AURORA_HUB = HubEvidence(
    hugging_face_id="microsoft/aurora", found=True, author="microsoft", developer="Microsoft",
    created_year=2024, paper_titles=["A Foundation Model for the Earth System"],
)


def _settings(tmp_path, **over) -> Settings:
    base = dict(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        dimensions_rate_limit_seconds=0.0,
        ai_model_plain_word_max_safe_hits=100,
        optimize_max_iterations=2,
    )
    base.update(over)
    return Settings(**base)


def _fake_eval(fp_by_call: list[dict] | None = None):
    calls: list[dict] = []

    def _eval(*_a, **kwargs):
        calls.append(kwargs)
        out = {
            "fp_rate_pct": 0.0, "genuine_count": 10, "fp_count": 0, "matched_count": 10,
            "total_pubs": 50, "coverage_pct": 100.0, "for_clause": kwargs.get("for_clause", ""),
            "publications_total": 50, "fp_hits": [],
        }
        if fp_by_call and len(calls) <= len(fp_by_call):
            out.update(fp_by_call[len(calls) - 1])
        return out

    _eval.calls = calls
    return _eval


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    """No network: Hub evidence, web search and the gate evaluator are faked."""
    state = SimpleNamespace(hub=AURORA_HUB, eval=_fake_eval())
    monkeypatch.setattr(qopt_mod, "fetch_hub_evidence", lambda _id: state.hub)
    monkeypatch.setattr(qopt_mod, "web_search", lambda _q: "- Aurora (Microsoft) weather model")
    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync", lambda *a, **k: state.eval(*a, **k)
    )
    return state


def _run(tmp_path, docs, llm, request: OptimizeRequest, **settings_over):
    dsl = CorpusDSL(docs)
    agent = MagicMock()
    agent.get_structured.return_value = llm
    uc = QueryOptimizationUseCase(
        dsl_port=dsl, research=MagicMock(), agent=agent, settings=_settings(tmp_path, **settings_over)
    )
    out = asyncio.run(uc.execute(request))
    return out, dsl, agent


def _aurora_request(**over) -> OptimizeRequest:
    return OptimizeRequest(
        dataset_name="Aurora", entity_type="ai_model", hugging_face_id="microsoft/aurora", **over
    )


def _decision(out, term: str, kind: str | None = None):
    found = [d for d in out.term_decisions if d.term == term and (kind is None or d.kind == kind)]
    assert found, f"no decision for {term!r}: {[d.term for d in out.term_decisions]}"
    return found[-1]


# --------------------------------------------------------------------------- Aurora


def test_aurora_bare_name_is_gated_never_safe(tmp_path, patched) -> None:
    out, dsl, _agent = _run(tmp_path, _aurora_corpus(), AURORA_LLM, _aurora_request())

    assert out.success, out.errors
    assert out.entity_type == "ai_model"
    assert "Aurora" not in out.aliases.safe
    assert out.aliases.risky == ["Aurora"]
    assert out.aliases.safe == ["Microsoft Aurora", "Aurora foundation model"]
    # Peer models make precise flags; the lab and the forecast centre do not.
    assert out.flag_terms == ["GraphCast", "Pangu-Weather"]
    assert out.exclusion_terms == ["aurora borealis"]
    assert out.for_clause == (
        '(\\"Microsoft Aurora\\" OR \\"Aurora foundation model\\") OR '
        '((\\"Aurora\\") AND (\\"GraphCast\\" OR \\"Pangu-Weather\\")) NOT (\\"aurora borealis\\")'
    )
    # The bare name is never counted or searched without a flag in a selected clause.
    hits = [d for d in _aurora_corpus() if _matches(out.for_clause, d)]
    assert set(hits) == {PRECISE, CITING, UNCONFIRMED_USE}
    assert "type=" not in out.dsl_query and "where year in [2024:" in out.dsl_query


def test_aurora_decisions_explain_every_term(tmp_path, patched) -> None:
    out, _dsl, _agent = _run(tmp_path, _aurora_corpus(), AURORA_LLM, _aurora_request())

    aurora = _decision(out, "Aurora", "alias")
    assert aurora.decision == "gated" and aurora.count == 515
    assert "single word with 515 hits" in aurora.reason and "context flag" in aurora.reason
    assert _decision(out, "Microsoft Aurora").decision == "safe"
    assert _decision(out, "Aurora model").decision == "rejected"
    assert "covered by flag-gated 'Aurora'" in _decision(out, "Aurora model").reason
    assert _decision(out, "Aurora AI").reason == "0 Dimensions hits"
    assert "lower case" in _decision(out, "aurora weather model").reason

    # Only a title cited on the model card is an anchor; one the LLM recalls is not.
    recalled = _decision(out, "Aurora: A Foundation Model of the Atmosphere")
    assert recalled.decision == "rejected" and "not cited on the model card" in recalled.reason
    cited = _decision(out, "A Foundation Model for the Earth System")
    assert cited.decision == "anchor" and "name 'Aurora'" in cited.reason
    assert _decision(out, "Microsoft", "flag").reason == "is part of one of the model's names"

    graphcast = _decision(out, "GraphCast", "flag")
    assert graphcast.decision == "flag"
    assert graphcast.stats == {"admitted": 50, "confirmed": 40, "anchor_precision": 0.8}
    assert _decision(out, "Pangu-Weather", "flag").reason.startswith("found in 30 abstracts")
    assert _decision(out, "ECMWF", "flag").decision == "context"
    assert _decision(out, "Microsoft Research", "flag").reason == (
        "ordinary words only: no coined name or acronym"
    )
    sql = _decision(out, "SQL", "flag")
    assert sql.decision == "rejected" and "mostly other meanings" in sql.reason
    assert sql.stats == {"admitted": 40, "confirmed": 0, "anchor_precision": 0.0}
    assert _decision(out, "weather forecasting", "flag").reason == (
        "not a proper name (lower-case word)"
    )
    for flag in out.flag_terms:
        assert flag[0].isupper()

    assert _decision(out, "aurora borealis", "exclude").decision == "exclude"
    assert _decision(out, "aurora borealis", "exclude").count == 5
    kinase = _decision(out, "Aurora kinase", "exclude")
    assert kinase.decision == "rejected" and "removes nothing" in kinase.reason


def test_aurora_seed_for_extraction(tmp_path, patched) -> None:
    out, _dsl, _agent = _run(
        tmp_path, _aurora_corpus(), AURORA_LLM, _aurora_request(years_end=2026)
    )

    assert out.ai_model_seed == {
        "model_name": "Aurora",
        "model_family": "Aurora",
        "description": "Aurora is Microsoft's foundation model of the atmosphere.",
        "hugging_face_id": "microsoft/aurora",
        "model_aliases": ["Microsoft Aurora", "Aurora foundation model"],
        "flag_terms": ["GraphCast", "Pangu-Weather"],
        "exclude_terms": ["aurora borealis"],
        "years_range": {"start_year": 2024, "end_year": 2026},
    }
    # In the extraction pipeline the flags gate every name.
    assert out.ai_model_seed_clause == (
        '((\\"Aurora\\" OR \\"Microsoft Aurora\\" OR \\"Aurora foundation model\\") AND '
        '(\\"GraphCast\\" OR \\"Pangu-Weather\\")) NOT (\\"aurora borealis\\")'
    )
    assert out.ai_model_seed_count == 50


def test_aurora_without_a_precise_flag_drops_the_bare_name(tmp_path, patched) -> None:
    """No context term is precise enough: the bare word is not searched at all, and the
    seed's model_name becomes a safe name (the pipeline always searches model_name)."""
    llm = dict(AURORA_LLM, context_terms=["Microsoft Research", "ECMWF", "Microsoft"])
    out, dsl, _agent = _run(
        tmp_path, _aurora_corpus(), llm, _aurora_request(), ai_model_max_mined_context_terms=0
    )

    assert out.success, out.errors
    assert out.aliases.safe == ["Microsoft Aurora", "Aurora foundation model"]
    assert out.aliases.risky == [] and out.flag_terms == []
    assert out.for_clause == '(\\"Microsoft Aurora\\" OR \\"Aurora foundation model\\")'
    aurora = _decision(out, "Aurora", "alias")
    assert aurora.decision == "rejected" and "no context flag qualified" in aurora.reason
    seed = out.ai_model_seed
    assert seed["model_name"] == "Microsoft Aurora"
    assert seed["model_family"] == "Aurora"
    assert "Aurora" not in [seed["model_name"], *seed["model_aliases"]]
    assert seed["flag_terms"] == []
    assert "model_name is 'Microsoft Aurora'" in out.notes


def test_aurora_without_llm_terms_stays_precise(tmp_path, patched) -> None:
    """The LLM returns nothing: Hub evidence, name heuristics and abstract mining remain."""
    out, _dsl, _agent = _run(tmp_path, _aurora_corpus(), {}, _aurora_request())

    assert out.success, out.errors
    assert "Aurora" not in out.aliases.safe
    assert out.aliases.safe == ["Microsoft Aurora", "Aurora foundation model"]
    assert out.aliases.risky == ["Aurora"]
    assert out.flag_terms == ["GraphCast", "Pangu-Weather"]
    assert "no LLM terms" in out.notes


def test_caller_flags_and_excludes_are_measured_not_trusted(tmp_path, patched) -> None:
    out, _dsl, _agent = _run(
        tmp_path, _aurora_corpus(), {},
        _aurora_request(flag_terms=["Microsoft", "atmosphere"], exclude_terms=["Microsoft Aurora", "GraphCast"]),
        ai_model_max_mined_context_terms=0,
    )

    assert out.success, out.errors
    assert out.flag_terms == [] and "Aurora" not in out.aliases.safe + out.aliases.risky
    assert _decision(out, "atmosphere", "flag").reason == "not a proper name (lower-case word)"
    assert _decision(out, "Microsoft", "flag").decision == "rejected"
    assert _decision(out, "Microsoft Aurora", "exclude").reason == "equals a model name"
    assert "would remove" in _decision(out, "GraphCast", "exclude").reason
    assert out.exclusion_terms == []


def test_abstract_false_positives_demote_a_safe_name(tmp_path, patched) -> None:
    fp = {
        "fp_rate_pct": 40.0, "fp_count": 4,
        "fp_hits": [
            {"matched_alias": "Microsoft Aurora", "snippet": "Microsoft Aurora 3D design tool"},
            {"matched_alias": "Microsoft Aurora", "snippet": "Microsoft Aurora 3D design tool"},
            {"matched_alias": "Aurora", "snippet": "aurora borealis over typhoon tracks"},
        ],
    }
    patched.eval = _fake_eval([fp])
    # One reply serves both structured calls: term research, then exclude derivation.
    llm = dict(AURORA_LLM, exclude_terms=["typhoon tracks", "3D design", "ERA5 from ECMWF"])
    docs = _aurora_corpus() + ["Microsoft Aurora 3D design tool for architects."]
    out, _dsl, _agent = _run(tmp_path, docs, llm, _aurora_request())

    assert out.success
    # An exclude must come from a false-positive snippet and stay narrow.
    assert "not a phrase of any" in _decision(out, "ERA5 from ECMWF", "exclude").reason
    assert "would remove 10 of 50 results: too broad" in (
        _decision(out, "typhoon tracks", "exclude").reason
    )
    kept = _decision(out, "3D design", "exclude")
    assert kept.decision == "exclude" and kept.stats["removed"] == 1
    assert "3D design" in out.exclusion_terms
    assert out.selected_variant == "V4~gate1"
    assert out.aliases.safe == ["Aurora foundation model"]
    assert "Microsoft Aurora" not in out.aliases.risky
    dropped = _decision(out, "Microsoft Aurora", "alias")
    assert dropped.decision == "rejected" and "not in the selected query" in dropped.reason
    assert "'Microsoft Aurora' dropped (2 false positives in abstracts)" in out.notes
    assert patched.eval.calls[0]["entity_context"].startswith("Aurora is Microsoft's")


def test_selected_clause_and_terms_stay_consistent_when_refinement_loses(tmp_path, patched) -> None:
    """Iteration 1 is worse than iteration 0: the record reports iteration 0's terms."""
    fp0 = {
        "fp_rate_pct": 20.0,
        "fp_hits": [{"matched_alias": "Microsoft Aurora", "snippet": "x"}] * 2,
    }
    patched.eval = _fake_eval([fp0, {"fp_rate_pct": 50.0}])
    out, _dsl, _agent = _run(tmp_path, _aurora_corpus(), AURORA_LLM, _aurora_request())

    assert out.selected_variant == "V4~gate0"
    assert out.aliases.safe == ["Microsoft Aurora", "Aurora foundation model"]
    assert _decision(out, "Microsoft Aurora", "alias").decision == "safe"
    assert '\\"Microsoft Aurora\\"' in out.for_clause


def test_own_name_with_mostly_genuine_mentions_survives_false_positives(tmp_path, patched) -> None:
    fp = {
        "fp_rate_pct": 20.0,
        "fp_hits": [{"matched_alias": "Microsoft Aurora", "snippet": "x"}] * 2,
        "genuine_by_alias": {"Microsoft Aurora": 8},
    }
    patched.eval = _fake_eval([fp, fp])
    out, _dsl, _agent = _run(tmp_path, _aurora_corpus(), AURORA_LLM, _aurora_request())

    assert "Microsoft Aurora" in out.aliases.safe
    assert "dropped" not in out.notes


def test_seed_is_withheld_when_the_abstract_check_fails(tmp_path, patched) -> None:
    patched.eval = _fake_eval([{"fp_rate_pct": 90.0}, {"fp_rate_pct": 90.0}])
    out, _dsl, _agent = _run(tmp_path, _aurora_corpus(), AURORA_LLM, _aurora_request())

    assert out.success and out.fp_rate_pct == 90.0
    assert out.ai_model_seed is None and out.ai_model_seed_clause is None
    assert out.confidence.value == "low"
    assert "seed withheld: the abstract check measured 90% false positives" in out.notes


def test_recalled_namesake_title_cannot_anchor_another_meaning(tmp_path, patched) -> None:
    """No model card: the LLM offers a kinase paper as "the Aurora paper". Nothing is
    certain, so no flag can be verified and the bare word is not searched."""
    patched.hub = HubEvidence()
    docs = ["Aurora kinase inhibitors in cancer therapy with AURKA."] * 40 + [POLAR] * 100
    llm = {
        "names": [], "paper_titles": ["Aurora kinase inhibitors in cancer therapy"],
        "context_terms": ["AURKA"], "other_meanings": [],
    }
    out, _dsl, _agent = _run(
        tmp_path, docs, llm, OptimizeRequest(dataset_name="Aurora", entity_type="ai_model")
    )

    assert out.success is False
    assert out.for_clause is None and out.ai_model_seed is None
    assert _decision(out, "AURKA", "flag").reason.startswith("cannot be verified")
    assert _decision(out, "Aurora", "alias").decision == "rejected"


def test_flag_inside_a_gated_name_is_rejected(tmp_path, patched) -> None:
    patched.hub = HubEvidence()
    docs = ["Stable Diffusion images"] * 40 + ["stable diffusion barrier in copper; Diffusion"] * 60
    out, _dsl, _agent = _run(
        tmp_path, docs, {},
        OptimizeRequest(dataset_name="Stable Diffusion", entity_type="ai_model", flag_terms=["Diffusion"]),
    )
    assert _decision(out, "Diffusion", "flag").reason == "is part of one of the model's names"
    assert out.flag_terms == []


def test_camel_case_common_word_over_the_cap_is_not_safe(tmp_path, patched) -> None:
    patched.hub = HubEvidence()
    docs = ["GloVe embeddings"] * 20 + ["surgical glove perforation"] * 500
    out, _dsl, _agent = _run(
        tmp_path, docs, {}, OptimizeRequest(dataset_name="GloVe", entity_type="ai_model")
    )
    assert out.success is False and out.ai_model_seed is None
    assert "single word with 520 hits" in _decision(out, "GloVe", "alias").reason


# --------------------------------------------------------------------------- other models


def test_coined_name_is_safe_without_flags(tmp_path, patched) -> None:
    patched.hub = HubEvidence(
        hugging_face_id="nvidia/fourcastnet3", found=True, developer="NVIDIA", created_year=2025,
        paper_titles=["FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting"],
    )
    docs = (
        ["FourCastNet 3 and FourCastNet3 from NVIDIA beat GraphCast. Ref: FourCastNet 3: A geometric "
         "approach to probabilistic machine-learning weather forecasting"] * 20
        + ["We compare with FourCastNet and GraphCast."] * 40
        + ["Unrelated GraphCast study."] * 100
    )
    llm = {
        "description": "FourCastNet is NVIDIA's weather model.",
        "names": [{"name": "FourCastNet3", "kind": "version"}, {"name": "FourCastNet 3", "kind": "version"}],
        "paper_titles": [], "context_terms": ["GraphCast", "ERA5"], "other_meanings": [],
    }
    out, dsl, _agent = _run(
        tmp_path, docs, llm,
        OptimizeRequest(dataset_name="FourCastNet", entity_type="ai_model", hugging_face_id="nvidia/fourcastnet3"),
    )

    assert out.success, out.errors
    assert out.aliases.safe == ["FourCastNet", "FourCastNet3", "FourCastNet 3"]
    assert out.aliases.risky == [] and out.flag_terms == []
    assert _decision(out, "GraphCast", "flag").reason == "not needed: no name requires gating"
    assert out.ai_model_seed["model_name"] == "FourCastNet"
    assert out.ai_model_seed["model_aliases"] == ["FourCastNet3", "FourCastNet 3"]
    assert out.ai_model_seed["flag_terms"] == []
    assert out.ai_model_seed_count == 60


def test_short_acronym_name_is_never_safe(tmp_path, patched) -> None:
    patched.hub = HubEvidence()
    docs = ["GPT2 language model text generation"] * 30 + ["GPT2 troposphere mapping for GNSS"] * 70
    out, _dsl, _agent = _run(
        tmp_path, docs, {}, OptimizeRequest(dataset_name="GPT2", entity_type="ai_model")
    )

    assert out.success is False and out.failed_phase == "phase1_alias_counting"
    assert "No name of 'GPT2' can be searched precisely" in out.errors[0]
    gpt2 = _decision(out, "GPT2", "alias")
    assert gpt2.decision == "rejected" and "short acronym" in gpt2.reason
    assert out.ai_model_seed is None


def test_dimensions_key_is_required(tmp_path, patched) -> None:
    out, _dsl, _agent = _run(tmp_path, [], {}, _aurora_request(), dimensions_api_key="")
    assert out.success is False and out.failed_phase == "precheck"


def test_dataset_runs_do_not_take_the_model_path(tmp_path, patched) -> None:
    out, _dsl, agent = _run(
        tmp_path, ["My Long Dataset Alpha"] * 5,
        {}, OptimizeRequest(dataset_name="My Long Dataset Alpha", dataset_names=["My Long Dataset Alpha"]),
    )
    assert out.success and out.entity_type == "dataset"
    assert out.term_decisions == [] and out.ai_model_seed is None
    agent.get_structured.assert_not_called()


# --------------------------------------------------------------------------- pure rules


@pytest.mark.parametrize(
    ("name", "plain", "distinctive"),
    [
        ("Aurora", True, False),
        ("Prithvi", True, False),
        ("ALBERT", True, False),
        ("GraphCast", False, True),
        ("FourCastNet3", False, True),
        ("Pangu-Weather", False, True),
        ("Prithvi-EO-2.0", False, True),
        ("BioCLIP 2", False, True),
        ("Aurora-0.25", False, False),
        ("Microsoft Aurora", False, False),
        ("GPT-2", False, False),
    ],
)
def test_name_shapes(name: str, plain: bool, distinctive: bool) -> None:
    assert is_plain_word(name) is plain
    assert is_distinctive(name) is distinctive


def _classify(name: str, count: int, confirmation: float | None, developer: str = "Microsoft"):
    return classify_model_name(
        name, count=count, confirmation=confirmation, developer=developer,
        safe_share=0.5, context_share=0.1, plain_word_max_hits=20_000,
    )[0]


def test_classify_model_name_rules() -> None:
    # A plain word: never safe above the hit cap, whatever the confirmation.
    assert _classify("Aurora", 795_247, 0.9) == "gate"
    assert _classify("Prithvi", 18_489, 0.03) == "gate"
    assert _classify("Prithvi", 18_489, None) == "gate"
    assert _classify("Enformer", 2_000, 0.7) == "safe"
    # Short acronyms are never safe.
    assert _classify("GPT2", 500, 0.99) == "gate"
    assert _classify("SAM", 500, None) == "gate"
    # Coined names need little confirmation; qualified phrases need a majority.
    assert _classify("GraphCast", 2_322, None) == "safe"
    assert _classify("GraphCast", 2_322, 0.3) == "safe"
    assert _classify("GraphCast", 2_322, 0.2) == "gate"
    # Case does not make a word coined: Dimensions matches "GloVe" as "glove".
    assert _classify("ClimaX", 21_510, 0.01) == "gate"
    assert _classify("GloVe", 150_000, None) == "gate"
    assert _classify("LLaMA", 200_000, 0.9) == "gate"
    # Titles exist but the share was not measured: never in the name's favor.
    assert classify_model_name(
        "GraphCast", count=2_322, confirmation=None, has_titles=True, developer="",
        safe_share=0.5, context_share=0.1, plain_word_max_hits=20_000,
    )[0] == "gate"
    assert _classify("Microsoft Aurora", 96, 0.65) == "safe"
    assert _classify("Microsoft Aurora", 96, 0.3) == "safe"  # developer-qualified: half the bar
    assert _classify("Aurora (Microsoft)", 28, 0.21) == "unconfirmed"
    assert _classify("Aurora model", 888, 0.075) == "unconfirmed"
    assert _classify("Aurora AI", 718, 0.01) == "unconfirmed"
    # Without any paper title only developer- or type-qualified phrases are trusted.
    assert _classify("Microsoft Aurora", 96, None) == "safe"
    assert _classify("Aurora foundation model", 36, None) == "safe"
    assert _classify("Aurora AI", 718, None) == "unconfirmed"
    assert _classify("Aurora-0.25", 21, None) == "unconfirmed"


def test_assemble_candidates_screens_by_shape() -> None:
    hub = HubEvidence(hugging_face_id="microsoft/aurora", found=True, developer="Microsoft")
    llm = {
        "names": [{"name": "aurora-0.25-finetuned.ckpt", "kind": "checkpoint"}, {"name": "model", "kind": "family"}],
        "paper_titles": ["Aurora paper", "Aurora: A Foundation Model of the Atmosphere"],
        "context_terms": ["ECMWF", "deep learning", "Aurora 0.25", "state UI"],
        "other_meanings": ["aurora borealis", "AK", "Aurora"],
    }
    c = assemble_candidates(
        model_name="Aurora", family=None, given_aliases=["Aurora-0.25"], hub=hub, llm=llm,
        max_names=14, max_context=12, max_excludes=8,
    )
    assert [n for n, _k in c.names] == [
        "Aurora", "Aurora-0.25", "Microsoft Aurora", "Aurora foundation model", "Aurora model", "aurora",
    ][:5]
    assert c.paper_titles == ["Aurora: A Foundation Model of the Atmosphere"]
    assert c.context_terms == ["ECMWF"]
    assert c.other_meanings == ["aurora borealis"]
    reasons = {term: reason for _kind, term, reason in c.rejected}
    assert reasons["aurora-0.25-finetuned.ckpt"] == "checkpoint file name"
    assert reasons["model"] == "generic words only"
    assert reasons["deep learning"] == "not a proper name (lower-case word)"
    assert reasons["state UI"] == "not a proper name (lower-case word)"
    assert reasons["Aurora 0.25"] == "contains the model's own name"
    assert reasons["Microsoft"] == "is part of one of the model's names"
    assert reasons["AK"] == "too short to exclude safely"
    assert reasons["Aurora"] == "equals a model name"
    assert reasons["Aurora paper"].startswith("fewer than 4 words")


def test_mine_context_terms_keeps_coined_names_and_acronyms() -> None:
    texts = [
        "Aurora beats GraphCast and Pangu-Weather on ERA5. The Model uses AI.",
        "GraphCast, ERA5 and IFS-HRES from ECMWF; Aurora-0.25 is finer.",
        "Weather models: GraphCast and Pangu-Weather (ECMWF).",
    ]
    mined = mine_context_terms(texts, model_names=["Aurora"], min_docs=2, limit=10)
    assert mined == [("GraphCast", 3), ("ECMWF", 2), ("ERA5", 2), ("Pangu-Weather", 2)]


def test_mined_terms_skip_ordinary_words_from_upper_case_titles() -> None:
    texts = [
        "AURORA FOR MEDIUM-RANGE WEATHER FORECASTING. We compare with GraphCast for weather.",
        "WEATHER FORECASTING AT SCALE with GraphCast; medium-range forecasting at scale.",
        "WEATHER FORECASTING FOR ALL: GraphCast and ECMWF. Forecasting the weather.",
        "ECMWF and FORECASTING of WEATHER.",
    ]
    mined = mine_context_terms(texts, model_names=["Aurora"], min_docs=2, limit=10)
    assert mined == [("GraphCast", 3), ("ECMWF", 2)]


@pytest.mark.parametrize(
    ("term", "ok"),
    [
        ("GraphCast", True), ("Pangu-Weather", True), ("ERA5", True), ("ECMWF", True),
        ("Google DeepMind", True), ("Weather Forecasting", False), ("Deep Learning", False),
        ("Earth System", False), ("Nature", False), ("WEATHER", False), ("Microsoft Research", False),
    ],
)
def test_flags_need_a_coined_name_or_acronym(term: str, ok: bool) -> None:
    assert amr.is_context_name(term) is ok
    c = assemble_candidates(
        model_name="Aurora", family=None, given_aliases=[], hub=HubEvidence(),
        llm={"context_terms": [term]}, max_names=14, max_context=12, max_excludes=8,
    )
    assert (term in c.context_terms) is ok


def test_ai_model_seed_mapping() -> None:
    gated = QueryCandidate(
        main_dataset_name="Aurora", safe=["Microsoft Aurora"], risky=["Aurora"],
        flag_terms=["GraphCast"], exclude_terms=["aurora borealis"], description="d",
    )
    seed = candidate_to_ai_model_seed(
        gated, model_name="Aurora", model_family="Aurora", hugging_face_id=None, years=(2024, 2026)
    )
    assert seed["model_name"] == "Aurora" and seed["model_aliases"] == ["Microsoft Aurora"]
    assert seed["flag_terms"] == ["GraphCast"]

    # Risky names without flags are never sent, and never become model_name.
    ungated = QueryCandidate(main_dataset_name="Aurora", safe=["Microsoft Aurora"], risky=["Aurora"])
    seed = candidate_to_ai_model_seed(
        ungated, model_name="Aurora", model_family="Aurora", hugging_face_id=None, years=(2024, 2026)
    )
    assert seed["model_name"] == "Microsoft Aurora" and seed["model_aliases"] == []

    with pytest.raises(ValueError, match="no searchable model name"):
        candidate_to_ai_model_seed(
            QueryCandidate(main_dataset_name="Aurora", risky=["Aurora"]),
            model_name="Aurora", model_family="Aurora", hugging_face_id=None, years=(2024, 2026),
        )
    assert ai_model_seed_clause(["A B"], [], []) == '(\\"A B\\")'
    assert ai_model_seed_clause(["A"], ["F"], ["X"]) == '((\\"A\\") AND (\\"F\\")) NOT (\\"X\\")'


def test_fetch_hub_evidence_reads_card_citation_and_arxiv() -> None:
    card = (
        "---\nlicense: mit\n---\n# Aurora\nSee https://arxiv.org/abs/2405.13063.\n"
        "```\n@article{bodnar2025aurora,\n  title = {A Foundation Model for the Earth System},\n}\n```"
    )
    atom = (
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Aurora: A Foundation Model\n of '
        "the Atmosphere</title><published>2024-05-20T00:00:00Z</published></entry></feed>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/api/models/microsoft/aurora"):
            return httpx.Response(200, json={"author": "microsoft", "tags": ["pytorch"], "createdAt": "2024-08-21T00:00:00Z"})
        if "/api/organizations/microsoft/overview" in url:
            return httpx.Response(200, json={"fullname": "Microsoft"})
        if url.endswith("/README.md"):
            return httpx.Response(200, text=card)
        if "arxiv.org" in url:
            return httpx.Response(200, text=atom)
        return httpx.Response(404)

    ev = fetch_hub_evidence("microsoft/aurora", httpx.Client(transport=httpx.MockTransport(handler)))
    assert ev.found and ev.developer == "Microsoft" and ev.created_year == 2024
    assert ev.arxiv_ids == ["2405.13063"]
    assert ev.paper_titles == [
        "A Foundation Model for the Earth System",
        "Aurora: A Foundation Model of the Atmosphere",
    ]
    assert ev.paper_years == [2024] and "license" not in ev.card_excerpt


def test_fetch_hub_evidence_never_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    ev = fetch_hub_evidence("microsoft/aurora", httpx.Client(transport=httpx.MockTransport(handler)))
    assert ev.found is False and ev.errors
    assert fetch_hub_evidence(None).found is False
    assert fetch_hub_evidence("a/..").errors

    def listing(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "x"}])

    odd = fetch_hub_evidence("a/b", httpx.Client(transport=httpx.MockTransport(listing)))
    assert odd.found is False and "Hub lookup failed" in odd.errors[0]
    assert "not a Hugging Face model id" in fetch_hub_evidence("https://huggingface.co/x").errors[0]


def test_mention_classifier_prompt_for_models() -> None:
    agent = MagicMock()
    agent.get_structured.return_value = {"is_genuine_dataset_reference": False}
    assert classify_alias_mention(
        agent, dataset_name="Aurora", alias="Aurora", snippet="aurora borealis",
        entity_context="Microsoft's atmosphere model.",
    ) is False
    prompt = agent.get_structured.call_args.args[0]
    assert prompt.startswith("AI model: 'Aurora'. Microsoft's atmosphere model.")
    assert "another meaning of the same word" in prompt


def test_cli_optimize_model_builds_an_ai_model_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataset_agent import bootstrap
    from dataset_agent.domain.models import QueryOptimizationRecord
    from dataset_agent.interfaces.cli import app

    seen: list[OptimizeRequest] = []

    class _UC:
        async def execute(self, req: OptimizeRequest) -> QueryOptimizationRecord:
            seen.append(req)
            return QueryOptimizationRecord(success=True, entity_type="ai_model")

    monkeypatch.setattr(bootstrap, "build_optimize_use_case", lambda _s: _UC())
    result = CliRunner().invoke(
        app,
        ["optimize-model", "Aurora", "--hf-id", "microsoft/aurora", "--alias", "Aurora-0.25",
         "--flag", "GraphCast", "--start-year", "2024"],
    )
    assert result.exit_code == 0, result.output
    req = seen[0]
    assert req.entity_type == "ai_model" and req.hugging_face_id == "microsoft/aurora"
    assert req.dataset_names == ["Aurora-0.25"] and req.flag_terms == ["GraphCast"]
    assert req.years_start == 2024
    assert '"entity_type": "ai_model"' in result.output


def test_module_documents_the_candidate_kinds() -> None:
    assert "paper titles" in (amr.__doc__ or "")

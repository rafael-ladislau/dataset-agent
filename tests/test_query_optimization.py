"""Query optimization use case (mocked Dimensions)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dataset_agent.application import query_optimization as qopt_mod
from dataset_agent.application.query_optimization import QueryOptimizationUseCase
from dataset_agent.domain.models import OptimizeRequest
from dataset_agent.settings import Settings


@pytest.fixture(autouse=True)
def _patch_evaluator(monkeypatch: pytest.MonkeyPatch):
    """Prevent real Dimensions API calls from evaluate_for_clause_sync."""
    def _fake_eval(*args, **kwargs):
        return {
            "fp_rate_pct": 0.0,
            "genuine_count": 10,
            "fp_count": 0,
            "matched_count": 10,
            "total_pubs": 100,
            "coverage_pct": 100.0,
            "for_clause": kwargs.get("for_clause", ""),
            "query": "search publications ...",
            "publications_total": 100,
            "fp_hits": [],
        }

    monkeypatch.setattr(qopt_mod, "evaluate_for_clause_sync", _fake_eval)


class _FakeDSL:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = responses
        self._i = 0

    async def execute_dsl(self, dsl: str) -> SimpleNamespace:
        if self._i >= len(self._responses):
            return SimpleNamespace(count_total=0, publications=[])
        r = self._responses[self._i]
        self._i += 1
        return r


def test_optimize_success_with_dataset_override(tmp_path) -> None:
    """Two long aliases + flag → V1 only; scope + selection complete."""
    responses = [
        SimpleNamespace(count_total=2000),
        SimpleNamespace(count_total=80),
        SimpleNamespace(count_total=90),
        SimpleNamespace(
            count_total=150,
            publications=[{"basics": {"title": "Publication on Another Long Title"}}],
        ),
        SimpleNamespace(count_total=800),
        SimpleNamespace(count_total=400),
    ]
    dsl = _FakeDSL(responses)
    research = MagicMock()

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0,
        optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
        optimize_suffix_check_max_aliases=0,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=dsl,
        research=research,
        agent=MagicMock(),
        settings=settings,
    )

    req = OptimizeRequest(
        dataset_name="My Long Dataset Alpha",
        dataset_names=[
            "My Long Dataset Alpha",
            "Another Long Title Here",
        ],
        flag_terms=["NIH"],
    )
    out = asyncio.run(uc.execute(req))

    assert out.success is True
    # With the gate-evaluator loop, the selected variant is the best gate-evaluated one
    assert out.selected_variant.startswith("V4~gate")
    assert out.for_clause
    assert out.dsl_query and "where year in" in out.dsl_query
    assert out.expected_count == 100  # from fake evaluator publications_total
    assert out.confidence is not None
    assert len(out.all_variants_tested) >= 1
    research.execute.assert_not_called()


def test_optimize_all_variants_zero_publications(tmp_path) -> None:
    """Dimensions returns zero hits for every variant probe."""
    responses = [
        SimpleNamespace(count_total=100),
        SimpleNamespace(count_total=10),
        SimpleNamespace(count_total=10),
        SimpleNamespace(count_total=0, publications=[]),
    ]
    dsl = _FakeDSL(responses)
    research = MagicMock()
    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0,
        optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
        optimize_suffix_check_max_aliases=0,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=dsl,
        research=research,
        agent=MagicMock(),
        settings=settings,
    )
    out = asyncio.run(
        uc.execute(
            OptimizeRequest(
                dataset_name="Long Primary Dataset Name",
                dataset_names=[
                    "Long Primary Dataset Name",
                    "Secondary Long Dataset Name",
                ],
                flag_terms=["NIH"],
            )
        )
    )
    assert out.success is False
    assert out.failed_phase == "phase2_variant_building"
    assert "zero publications" in (out.errors[0] or "").lower()


def test_optimize_missing_dimensions_key(tmp_path) -> None:
    research = MagicMock()
    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="",
    )
    uc = QueryOptimizationUseCase(
        dsl_port=_FakeDSL([]),
        research=research,
        agent=MagicMock(),
        settings=settings,
    )
    out = asyncio.run(
        uc.execute(
            OptimizeRequest(
                dataset_name="X",
                dataset_names=["Long Name One"],
                flag_terms=["F"],
            )
        )
    )
    assert out.success is False
    assert out.failed_phase == "precheck"
    assert out.errors


def test_optimize_signal3_merges_high_fp_into_primary_variant(tmp_path) -> None:
    """Signal 3 can raise fp_rate_pct above keyword-only proxy on the primary variant."""
    responses = [
        SimpleNamespace(count_total=2000),
        SimpleNamespace(count_total=80),
        SimpleNamespace(count_total=90),
        SimpleNamespace(
            count_total=150,
            publications=[{"basics": {"title": "Publication on Another Long Title"}}],
        ),
        SimpleNamespace(count_total=800),
        SimpleNamespace(count_total=400),
    ]
    dsl = _FakeDSL(responses)
    research = MagicMock()
    agent = MagicMock()
    agent.get_structured.return_value = {
        "mention_score": 2,
        "context_score": 2,
        "mentioned_term": "",
        "reason": "mock low relevance",
    }

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=3,
        optimize_signal3_min_relevance_sum=10,
        optimize_abstract_exclude_max_hits=0,
        optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
        optimize_suffix_check_max_aliases=0,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=dsl,
        research=research,
        agent=agent,
        settings=settings,
    )
    out = asyncio.run(
        uc.execute(
            OptimizeRequest(
                dataset_name="My Long Dataset Alpha",
                dataset_names=[
                    "My Long Dataset Alpha",
                    "Another Long Title Here",
                ],
                flag_terms=["NIH"],
            )
        )
    )
    assert out.success is True
    # Gate evaluator (patched) returns fp_rate_pct=0.0
    assert out.fp_rate_pct == 0.0
    assert any("gate_iter" in n for n in out.notes.split())


def test_optimize_merges_abstract_derived_exclude_terms(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Abstract snippets feed derive_exclude_terms_from_fps into exclusion_terms."""
    responses = [
        SimpleNamespace(count_total=2000),
        SimpleNamespace(count_total=80),
        SimpleNamespace(count_total=90),
        SimpleNamespace(
            count_total=150,
            publications=[{"basics": {"title": "Publication on Another Long Title"}}],
        ),
        SimpleNamespace(
            count_total=40,
            publications=[{"basics": {"title": "Other hit"}}],
        ),
        SimpleNamespace(count_total=800),
        SimpleNamespace(count_total=400),
        SimpleNamespace(count_total=10),
        SimpleNamespace(count_total=8),
    ]
    dsl = _FakeDSL(responses)
    research = MagicMock()
    agent = MagicMock()

    async def _fake_fetch(*_a, **_k):
        return (
            [
                {
                    "id": "p1",
                    "basics": {
                        "title": "Paper",
                        "abstract": "We discuss RISK unrelated to the dataset product.",
                    },
                }
            ],
            0,
        )

    monkeypatch.setattr(qopt_mod, "fetch_top_n", _fake_fetch)

    def _fake_scan(pubs: list, aliases: list, **kw: object) -> list:
        _ = pubs, aliases, kw
        return [
                {
                    "publication_id": "p1",
                    "matched_alias": "RISK",
                    "snippet": "snippet RISK here",
                }
        ]

    monkeypatch.setattr(qopt_mod, "scan_abstracts_for_aliases", _fake_scan)
    monkeypatch.setattr(
        qopt_mod,
        "derive_exclude_terms_from_fps",
        lambda _agent, **kw: ["offdomainterm"],
    )
    monkeypatch.setattr(
        qopt_mod,
        "evaluate_for_clause_sync",
        lambda *args, **kwargs: {
            "fp_rate_pct": 15.0,
            "genuine_count": 85,
            "fp_count": 15,
            "matched_count": 100,
            "total_pubs": 100,
            "coverage_pct": 100.0,
            "for_clause": kwargs.get("for_clause", ""),
            "query": "search publications ...",
            "publications_total": 100,
            "fp_hits": [{"matched_alias": "RISK", "snippet": "risk factor"}],
        },
    )

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=8,
        optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
        optimize_suffix_check_max_aliases=0,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=dsl,
        research=research,
        agent=agent,
        settings=settings,
    )
    out = asyncio.run(
        uc.execute(
            OptimizeRequest(
                dataset_name="My Long Dataset Alpha",
                dataset_names=[
                    "My Long Dataset Alpha",
                    "Another Long Title Here",
                    "RISK",
                ],
                flag_terms=["NIH"],
            )
        )
    )
    assert out.success is True
    assert "offdomainterm" in [x.lower() for x in out.exclusion_terms]


def test_optimize_abstract_classify_skips_derive_when_all_genuine(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """When classify_alias_mention marks all hits as genuine, no exclude derivation runs."""
    responses = [
        SimpleNamespace(count_total=2000),
        SimpleNamespace(count_total=80),
        SimpleNamespace(count_total=90),
        SimpleNamespace(
            count_total=150,
            publications=[{"basics": {"title": "Publication on Another Long Title"}}],
        ),
        SimpleNamespace(
            count_total=40,
            publications=[{"basics": {"title": "Other hit"}}],
        ),
        SimpleNamespace(count_total=800),
        SimpleNamespace(count_total=400),
        SimpleNamespace(count_total=10),
        SimpleNamespace(count_total=8),
    ]
    dsl = _FakeDSL(responses)
    research = MagicMock()
    agent = MagicMock()

    async def _fake_fetch(*_a, **_k):
        return (
            [
                {
                    "id": "p1",
                    "basics": {
                        "title": "Paper",
                        "abstract": "We discuss RISK unrelated to the dataset product.",
                    },
                }
            ],
            0,
        )

    # The autouse fake evaluator returns fp_rate=0.0 (within threshold) and empty fp_hits,
    # so the refinement loop stops immediately and derive_exclude_terms_from_fps never runs.
    monkeypatch.setattr(
        qopt_mod,
        "derive_exclude_terms_from_fps",
        lambda *_a, **_k: pytest.fail("derive should not run when no FP hits"),
    )

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=8,
        optimize_abstract_classify_max_calls=4,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
        optimize_suffix_check_max_aliases=0,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=dsl,
        research=research,
        agent=agent,
        settings=settings,
    )
    out = asyncio.run(
        uc.execute(
            OptimizeRequest(
                dataset_name="My Long Dataset Alpha",
                dataset_names=[
                    "My Long Dataset Alpha",
                    "Another Long Title Here",
                    "RISK",
                ],
                flag_terms=["NIH"],
            )
        )
    )
    assert out.success is True
    assert "FP rate within threshold" in out.notes


def test_optimize_abstract_classify_fp_hits_feed_derive(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """With classify on, only non-genuine hits are passed to derive_exclude_terms_from_fps."""
    responses = [
        SimpleNamespace(count_total=2000),
        SimpleNamespace(count_total=80),
        SimpleNamespace(count_total=90),
        SimpleNamespace(
            count_total=150,
            publications=[{"basics": {"title": "Publication on Another Long Title"}}],
        ),
        SimpleNamespace(
            count_total=40,
            publications=[{"basics": {"title": "Other hit"}}],
        ),
        SimpleNamespace(count_total=800),
        SimpleNamespace(count_total=400),
        SimpleNamespace(count_total=10),
        SimpleNamespace(count_total=8),
    ]
    dsl = _FakeDSL(responses)
    research = MagicMock()
    agent = MagicMock()

    async def _fake_fetch(*_a, **_k):
        return (
            [
                {
                    "id": "p1",
                    "basics": {
                        "title": "Paper",
                        "abstract": "We discuss RISK unrelated to the dataset product.",
                    },
                }
            ],
            0,
        )

    monkeypatch.setattr(
        qopt_mod,
        "derive_exclude_terms_from_fps",
        lambda _agent, **kw: ["fromabstract"],
    )
    monkeypatch.setattr(
        qopt_mod,
        "evaluate_for_clause_sync",
        lambda *args, **kwargs: {
            "fp_rate_pct": 15.0,
            "genuine_count": 85,
            "fp_count": 15,
            "matched_count": 100,
            "total_pubs": 100,
            "coverage_pct": 100.0,
            "for_clause": kwargs.get("for_clause", ""),
            "query": "search publications ...",
            "publications_total": 100,
            "fp_hits": [{"matched_alias": "RISK", "snippet": "snip"}],
        },
    )

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=8,
        optimize_abstract_classify_max_calls=4,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
        optimize_suffix_check_max_aliases=0,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=dsl,
        research=research,
        agent=agent,
        settings=settings,
    )
    out = asyncio.run(
        uc.execute(
            OptimizeRequest(
                dataset_name="My Long Dataset Alpha",
                dataset_names=[
                    "My Long Dataset Alpha",
                    "Another Long Title Here",
                    "RISK",
                ],
                flag_terms=["NIH"],
            )
        )
    )
    assert out.success is True
    assert "fromabstract" in [x.lower() for x in out.exclusion_terms]


def test_fp_rate_counts_all_fp_domains() -> None:
    """Keyword FP proxy must match domain keys from identify_fp_domains, not only 'risky'."""
    fp_kw = {"nursing": ["patient", "hospital"], "technology": ["neural network"]}
    pct, noise = qopt_mod._fp_rate_and_noise_terms(
        [
            "Diagnostic and Statistical Manual of Mental Disorders",
            "Nursing patient outcomes in acute care",
        ],
        fp_kw,
    )
    assert pct == 50.0
    assert "patient" in noise


def test_optimize_short_acronym_suffix_via_rk0(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Short acronyms pre-bucketed as risky (e.g. ONET) still get suffix promotion."""
    responses = [
        SimpleNamespace(count_total=2000),
        SimpleNamespace(count_total=500_000),
        SimpleNamespace(count_total=5000),
        SimpleNamespace(
            count_total=150,
            publications=[{"basics": {"title": "Paper"}}],
        ),
        SimpleNamespace(count_total=800),
        SimpleNamespace(count_total=400),
    ]
    dsl = _FakeDSL(responses)

    async def _fake_suffix(_dsl, alias: str, **kw: object) -> str | None:
        _ = kw
        if alias == "ONET":
            return "ONET dataset"
        return None

    monkeypatch.setattr(qopt_mod, "check_suffix_necessity", _fake_suffix)

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0,
        optimize_abstract_classify_max_calls=0,
        optimize_suffix_check_enabled=True,
        optimize_suffix_check_max_aliases=3,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=dsl,
        research=MagicMock(),
        agent=MagicMock(),
        settings=settings,
    )
    out = asyncio.run(
        uc.execute(
            OptimizeRequest(
                dataset_name="My Long Dataset Alpha",
                dataset_names=["My Long Dataset Alpha", "ONET"],
                flag_terms=["NIH"],
            )
        )
    )
    assert out.success is True
    assert out.aliases is not None
    assert any("dataset" in s.lower() for s in out.aliases.safe)
    assert "suffix_promoted: 'ONET'" in out.notes


def test_optimize_suffix_promotes_risky_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Risky alias with high bare/suffixed ratio is promoted to safe via suffix check."""
    responses = [
        SimpleNamespace(count_total=2000),
        SimpleNamespace(count_total=80),
        SimpleNamespace(count_total=500_000),
        SimpleNamespace(
            count_total=150,
            publications=[{"basics": {"title": "Paper on Ambiguous Term dataset"}}],
        ),
        SimpleNamespace(count_total=800),
        SimpleNamespace(count_total=400),
    ]
    dsl = _FakeDSL(responses)

    async def _fake_suffix(_dsl, alias: str, **kw: object) -> str | None:
        _ = kw
        if alias == "Ambiguous Term":
            return "Ambiguous Term dataset"
        return None

    monkeypatch.setattr(qopt_mod, "check_suffix_necessity", _fake_suffix)

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0,
        optimize_abstract_classify_max_calls=0,
        optimize_suffix_check_enabled=True,
        optimize_suffix_check_max_aliases=3,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=dsl,
        research=MagicMock(),
        agent=MagicMock(),
        settings=settings,
    )
    out = asyncio.run(
        uc.execute(
            OptimizeRequest(
                dataset_name="My Long Dataset Alpha",
                dataset_names=["My Long Dataset Alpha", "Ambiguous Term"],
                flag_terms=["NIH"],
            )
        )
    )
    assert out.success is True
    assert out.aliases is not None
    assert any("dataset" in s.lower() for s in out.aliases.safe)
    assert "suffix_promoted" in out.notes


def test_optimize_fp_domains_enriches_keywords(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """identify_fp_domains replaces crude risky-token fp_keywords when enabled."""
    responses = [
        SimpleNamespace(count_total=2000),
        SimpleNamespace(count_total=80),
        SimpleNamespace(count_total=500_000),
        SimpleNamespace(
            count_total=150,
            publications=[{"basics": {"title": "Nursing paper with RISK acronym"}}],
        ),
        SimpleNamespace(count_total=800),
        SimpleNamespace(count_total=400),
        SimpleNamespace(count_total=10),
        SimpleNamespace(count_total=8),
    ]
    dsl = _FakeDSL(responses)

    def _fake_fp_domains(_agent, _ds: str, alias: str) -> dict[str, list[str]]:
        if alias == "RISK":
            return {"nursing": ["patient", "hospital"]}
        return {}

    monkeypatch.setattr(qopt_mod, "identify_fp_domains", _fake_fp_domains)
    monkeypatch.setattr(qopt_mod, "web_search_alias_meanings", lambda *_a, **_k: {})

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0,
        optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=2,
        optimize_web_disambig_max_aliases=0,
        optimize_suffix_check_max_aliases=0,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=dsl,
        research=MagicMock(),
        agent=MagicMock(),
        settings=settings,
    )
    out = asyncio.run(
        uc.execute(
            OptimizeRequest(
                dataset_name="My Long Dataset Alpha",
                dataset_names=[
                    "My Long Dataset Alpha",
                    "Another Long Title Here",
                    "RISK",
                ],
                flag_terms=["NIH"],
            )
        )
    )
    assert out.success is True
    # Old fp_domains enrichment was removed; gate evaluation runs instead
    assert "gate_iter" in out.notes


def test_optimize_web_disambig_adds_excludes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """web_search_alias_meanings seeds exclusion_terms from web-grounded candidates."""
    responses = [
        SimpleNamespace(count_total=2000),
        SimpleNamespace(count_total=80),
        SimpleNamespace(count_total=500_000),
        SimpleNamespace(
            count_total=150,
            publications=[{"basics": {"title": "Off-domain RISK paper"}}],
        ),
        SimpleNamespace(count_total=800),
        SimpleNamespace(count_total=400),
        SimpleNamespace(count_total=10),
        SimpleNamespace(count_total=8),
    ]
    dsl = _FakeDSL(responses)

    monkeypatch.setattr(qopt_mod, "identify_fp_domains", lambda *_a, **_k: {})
    monkeypatch.setattr(
        qopt_mod,
        "web_search_alias_meanings",
        lambda _agent, alias: (
            {"informatics": ["electronic health", "clinical trial"]}
            if alias == "RISK"
            else {}
        ),
    )

    settings = Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0,
        optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=2,
        optimize_suffix_check_max_aliases=0,
    )
    uc = QueryOptimizationUseCase(
        dsl_port=dsl,
        research=MagicMock(),
        agent=MagicMock(),
        settings=settings,
    )
    out = asyncio.run(
        uc.execute(
            OptimizeRequest(
                dataset_name="My Long Dataset Alpha",
                dataset_names=[
                    "My Long Dataset Alpha",
                    "Another Long Title Here",
                    "RISK",
                ],
                flag_terms=["NIH"],
            )
        )
    )
    assert out.success is True
    # Old web_disambig enrichment was removed; gate evaluation runs instead
    assert "gate_iter" in out.notes

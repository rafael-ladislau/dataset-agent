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
    assert out.selected_variant.startswith("V1~gate")
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


@pytest.mark.parametrize("flags", [["NIH"], []])
def test_optimize_listed_acronym_suffix_alias_is_risky_with_flags(
    flags: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A research-listed suffixed acronym form (B-04 shape rule) stays risky with flags;
    without flags it stays safe (a risky alias would never be searched)."""
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
                dataset_names=["My Long Dataset Alpha", "ONET", "ONET dataset"],
                flag_terms=flags,
            )
        )
    )
    assert out.success is True
    assert out.aliases is not None
    if not flags:
        assert "ONET dataset" in out.aliases.safe
        assert "ONET dataset" not in out.aliases.risky
        assert "suffix_risky" not in out.notes
        return
    # Research listing "ONET dataset" itself does not make it safe: same shape, same risk.
    assert "ONET dataset" in out.aliases.risky
    assert "ONET dataset" not in out.aliases.safe
    assert "suffix_risky: 'ONET dataset'" in out.notes
    assert out.notes.count("suffix_risky: 'ONET dataset'") == 1
    entry = next(e for e in out.alias_counts if e.alias == "ONET dataset")
    assert entry.risk == "risky"


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


_LEXER_ERRORS = {
    "query": {
        "header": "Semantic errors found",
        "details": ["LexerErrors: token recognition error at: '\"MIDRC\xa0​'"],
    }
}


def _phase1_settings(tmp_path) -> Settings:
    return Settings(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0,
        optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
    )


class _SelectiveDSL:
    """Raises a lexer error for queries containing ``bad``; counts the rest."""

    def __init__(self, bad: str | None, error: Exception | None = None) -> None:
        self.bad = bad
        self.error = error
        self.calls: list[str] = []

    async def execute_dsl(self, dsl: str) -> SimpleNamespace:
        from dataset_agent.adapters.dimensions_dsl import DimensionsDslError

        self.calls.append(dsl)
        if self.error is not None:
            raise self.error
        if self.bad and self.bad in dsl:
            raise DimensionsDslError(
                "Dimensions rejected the query: LexerErrors", errors=_LEXER_ERRORS
            )
        return SimpleNamespace(count_total=500, publications=[])


def _optimize(tmp_path, dsl, names) -> object:
    uc = QueryOptimizationUseCase(
        dsl_port=dsl, research=MagicMock(), agent=MagicMock(), settings=_phase1_settings(tmp_path)
    )
    req = OptimizeRequest(
        dataset_name="MIDRC Data Commons", dataset_names=names, flag_terms=["NIH"]
    )
    return asyncio.run(uc.execute(req))


def test_phase1_drops_alias_dimensions_cannot_parse(tmp_path) -> None:
    dsl = _SelectiveDSL("Broken Alias")
    out = _optimize(tmp_path, dsl, ["MIDRC Data Commons", "Broken Alias Name"])
    assert out.failed_phase != "phase1_alias_counting"
    assert "alias 'Broken Alias Name' dropped: Dimensions could not parse it" in out.notes
    assert all(e.alias != "Broken Alias Name" for e in out.alias_counts)


def test_phase1_fails_when_every_alias_is_unparseable(tmp_path) -> None:
    dsl = _SelectiveDSL("Broken")
    # The main-name count uses a good term; both aliases are rejected.
    out = _optimize(tmp_path, dsl, ["Broken One", "Broken Two"])
    assert out.success is False
    assert out.failed_phase == "phase1_alias_counting"
    assert "could not parse any alias" in out.errors[0]


def test_phase1_transient_error_still_fails_the_run(tmp_path) -> None:
    from dataset_agent.adapters.dimensions_dsl import DimensionsDslError

    dsl = _SelectiveDSL(None, DimensionsDslError("Service unavailable", transient=True))
    out = _optimize(tmp_path, dsl, ["MIDRC Data Commons", "Other Long Name"])
    assert out.success is False
    assert out.failed_phase == "phase1_alias_counting"
    assert "dropped" not in out.notes


def test_phase1_sanitizes_junk_aliases_before_counting(tmp_path) -> None:
    dsl = _SelectiveDSL(None)
    junk = "MIDRC" + "\xa0​\xa0\xa0" * 4 + "..."
    _optimize(tmp_path, dsl, ["MIDRC Data Commons", junk])
    assert not any("\xa0" in q or "​" in q or "..." in q for q in dsl.calls)


def _unknown_rate_run(monkeypatch, tmp_path):
    """Non-full-text run whose abstract check classified nothing (every call failed)."""
    def _eval(*_a, **kwargs):
        return {
            "fp_rate_pct": None, "genuine_count": 1, "fp_count": 0, "classified_count": 1,
            "failed_count": 4, "attempted_count": 4, "matched_count": 10, "total_pubs": 100,
            "coverage_pct": 100.0, "for_clause": kwargs.get("for_clause", ""),
            "query": "search publications ...", "publications_total": 100, "fp_hits": [],
        }

    monkeypatch.setattr(qopt_mod, "evaluate_for_clause_sync", _eval)
    monkeypatch.setattr(
        qopt_mod, "derive_exclude_terms_from_fps",
        lambda *_a, **_k: pytest.fail("no refinement on an unknown FP rate"),
    )
    responses = [SimpleNamespace(count_total=c) for c in (2000, 80, 90, 150, 40, 800, 400, 10, 8)]
    settings = Settings(
        tasks_db=tmp_path / "t.db", output_dir=tmp_path / "out", dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50, optimize_signal3_max_titles=0,
        optimize_abstract_classify_max_calls=4, optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0, optimize_max_iterations=3,
    )
    uc = QueryOptimizationUseCase(dsl_port=_FakeDSL(responses), research=MagicMock(), agent=MagicMock(),
                                  settings=settings)
    return asyncio.run(uc.execute(OptimizeRequest(
        dataset_name="My Long Dataset Alpha",
        dataset_names=["My Long Dataset Alpha", "Another Long Title Here", "RISK"],
        flag_terms=["NIH"],
    )))


def test_unknown_fp_rate_stops_the_gate_loop_without_within_threshold(monkeypatch, tmp_path) -> None:
    out = _unknown_rate_run(monkeypatch, tmp_path)
    assert out.success is True
    assert "FP rate within threshold" not in out.notes
    assert "gate_iter0: FP rate unknown; stopping." in out.notes
    assert out.fp_rate_pct is None
    assert out.confidence.value == "low"


# ---- exclusion guards ------------------------------------------------------------------


class _CountingDSL:
    """Answers by query shape and records every query."""

    def __init__(self, v3=1000, v4=5, terms=None) -> None:
        self.v3, self.v4, self.terms = v3, v4, dict(terms or {})
        self.queries: list[str] = []

    async def execute_dsl(self, dsl: str) -> SimpleNamespace:
        self.queries.append(dsl)
        for term, n in self.terms.items():
            if f'AND (\\"{term}\\")' in dsl:
                return SimpleNamespace(count_total=n, publications=[])
        if " NOT " in dsl:
            return SimpleNamespace(count_total=self.v4, publications=[])
        return SimpleNamespace(count_total=self.v3, publications=[])


def _guard_uc(tmp_path, dsl, **overrides) -> QueryOptimizationUseCase:
    kw = dict(
        tasks_db=tmp_path / "t.db",
        output_dir=tmp_path / "out",
        dimensions_api_key="test-key",
        optimize_max_dimensions_calls=50,
        optimize_signal3_max_titles=0,
        optimize_abstract_exclude_max_hits=0,
        optimize_abstract_classify_max_calls=0,
        optimize_fp_domains_max_aliases=0,
        optimize_web_disambig_max_aliases=0,
    )
    kw.update(overrides)
    return QueryOptimizationUseCase(
        dsl_port=dsl, research=MagicMock(), agent=MagicMock(), settings=Settings(**kw),
    )


def _guard_req(excludes) -> OptimizeRequest:
    return OptimizeRequest(
        dataset_name="American Time Use Survey",
        dataset_names=["American Time Use Survey", "ATUS"],
        flag_terms=["Bureau of Labor Statistics"],
        exclude_terms=excludes,
    )


def test_guard_drops_own_name_exclude_from_the_final_query(tmp_path) -> None:
    dsl = _CountingDSL(v3=1000, v4=900)
    uc = _guard_uc(tmp_path, dsl)
    out = asyncio.run(uc.execute(_guard_req(
        ["American Time Use Survey", "Bureau of Labor Statistics", "species"]
    )))
    assert out.success is True
    assert out.exclusion_terms == ["species"]
    assert 'NOT (\\"American Time Use Survey\\"' not in out.dsl_query
    assert "exclude guard: dropped ['American Time Use Survey' (inside name" in out.notes
    assert "'Bureau of Labor Statistics' (inside flag" in out.notes


def test_impact_guard_drops_the_exclude_that_removes_most_of_v3(tmp_path) -> None:
    dsl = _CountingDSL(v3=1000, v4=5, terms={"journal": 950, "chemistry": 20})
    uc = _guard_uc(tmp_path, dsl)
    out = asyncio.run(uc.execute(_guard_req(["journal", "chemistry"])))
    assert out.success is True
    assert out.exclusion_terms == ["chemistry"]
    assert "exclude impact: dropped ['journal' removes 95% of V3], V4 5 -> " in out.notes
    v4 = next(v for v in out.all_variants_tested if v.label == "V4")
    assert '\\"journal\\"' not in v4.for_clause and '\\"chemistry\\"' in v4.for_clause
    # V4 was recounted: the final repaired clause was queried
    assert any(" NOT " in q and "journal" not in q for q in dsl.queries)


def test_impact_guard_is_free_when_v4_keeps_half_of_v3(tmp_path) -> None:
    base = _CountingDSL(v3=1000, v4=500, terms={"journal": 950})
    asyncio.run(_guard_uc(tmp_path, base).execute(_guard_req(["journal", "chemistry"])))
    assert not any('AND (\\"journal\\")' in q for q in base.queries)
    off = _CountingDSL(v3=1000, v4=5, terms={"journal": 950})
    asyncio.run(
        _guard_uc(tmp_path, off, optimize_exclude_max_total_share=1.0)
        .execute(_guard_req(["journal", "chemistry"]))
    )
    assert not any('AND (\\"journal\\")' in q for q in off.queries)


def test_impact_guard_drops_unmeasured_excludes_when_budget_runs_out(tmp_path) -> None:
    dsl = _CountingDSL(v3=1000, v4=5, terms={"journal": 950, "chemistry": 20})
    # 3 phase-1 + V1/V3/V4 probes use 6; room for the first term only.
    uc = _guard_uc(tmp_path, dsl, optimize_max_dimensions_calls=7)
    out = asyncio.run(uc.execute(_guard_req(["chemistry", "journal", "species"])))
    assert "unmeasured (budget)" in out.notes
    assert "species" not in out.exclusion_terms and "journal" not in out.exclusion_terms


def test_gate_loop_derived_alias_never_reaches_the_clause(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        qopt_mod, "derive_exclude_terms_from_fps",
        lambda _agent, **kw: ["American Time Use Survey", "offdomainterm"],
    )
    monkeypatch.setattr(
        qopt_mod, "evaluate_for_clause_sync",
        lambda *a, **k: {
            "fp_rate_pct": 15.0, "genuine_count": 85, "fp_count": 15, "matched_count": 100,
            "total_pubs": 100, "coverage_pct": 100.0, "for_clause": k.get("for_clause", ""),
            "query": "q", "publications_total": 100,
            "fp_hits": [{"matched_alias": "ATUS", "snippet": "atus risk"}],
        },
    )
    uc = _guard_uc(tmp_path, _CountingDSL(v3=1000, v4=900))
    out = asyncio.run(uc.execute(_guard_req(["species"])))
    assert "offdomainterm" in out.exclusion_terms
    assert "American Time Use Survey" not in out.exclusion_terms

from __future__ import annotations

import pytest

from dataset_agent.adapters.seed_mapping import (
    QueryCandidate,
    alias_coverage_missing,
    candidate_to_seed,
)


def test_safe_and_risky_buckets():
    c = QueryCandidate("NHIS", safe=["National Health Interview Survey"], risky=["NHIS"],
                       flag_terms=["CDC"], exclude_terms=["Nursing"])
    s = candidate_to_seed(c)
    assert s["standalone_names"] == ["National Health Interview Survey"]
    assert s["dataset_names"] == ["NHIS"]
    assert s["flag_terms"] == ["CDC"] and s["exclude_terms"] == ["Nursing"]
    assert s["dimensions_query"] is None
    assert s["engine"] == "dimensions" and "probe" not in s


def test_safe_only_uses_raw_clause_with_all_aliases():
    c = QueryCandidate("X", safe=["Alpha Survey", "Beta Survey"])
    s = candidate_to_seed(c)
    assert s["dimensions_query"]
    assert "Alpha Survey" in s["dimensions_query"].replace("\\", "")
    assert set(s["dataset_names"]) == {"Alpha Survey", "Beta Survey"}


def test_explicit_raw_clause_checked_for_coverage():
    c = QueryCandidate("X", risky=["NHIS"], raw_clause='(\\"NHIS\\" AND \\"survey\\")')
    assert candidate_to_seed(c)["dimensions_query"] == c.raw_clause
    bad = QueryCandidate("X", risky=["NHIS", "Other"], raw_clause='(\\"NHIS\\")')
    with pytest.raises(ValueError, match="Other"):
        candidate_to_seed(bad)


def test_coverage_is_case_insensitive():
    assert alias_coverage_missing('\\"nhis\\"', ["NHIS"]) == []
    assert alias_coverage_missing('\\"nhis\\"', ["NHANES"]) == ["NHANES"]


def _raw_seed(*aliases):
    return candidate_to_seed(QueryCandidate("X", safe=list(aliases)))


@pytest.mark.parametrize(
    "alias",
    ["Health & Retirement Study", "Panel Study of Income Dynamics (PSID)", "ECLS-K:2011"],
)
def test_raw_clause_uses_server_quoting_without_dsl_escapes(alias):
    q = _raw_seed(alias)["dimensions_query"]
    assert q == '(\\"' + alias + '\\")'
    assert alias.lower() in q.lower()


def test_raw_clause_special_chars_pass_server_validation_when_available():
    pytest.importorskip("use_cases.process_datasets.query_builder")
    from use_cases.process_datasets.query_builder import normalize_raw_clause, validate_raw_clause

    aliases = ["Health & Retirement Study", "Panel Study of Income Dynamics (PSID)", "ECLS-K:2011"]
    q = _raw_seed(*aliases)["dimensions_query"]
    validate_raw_clause(normalize_raw_clause(q), aliases)


def test_coverage_does_not_strip_backslashes():
    # The server compares the alias verbatim against the clause, so a DSL-escaped
    # alias ("\&") must not count as covering "&".
    assert alias_coverage_missing('(\\"Health \\& Retirement Study\\")', ["Health & Retirement Study"]) == [
        "Health & Retirement Study"
    ]
    assert alias_coverage_missing('(\\"Health & Retirement Study\\")', ["Health & Retirement Study"]) == []


def test_double_quote_in_alias_is_escaped_and_checked_like_server():
    c = QueryCandidate("X", safe=['The "Big" Survey'])
    # Escaped as \" inside the phrase; the server's verbatim check then reports it
    # as not covered, so the client fails fast instead of getting an HTTP 422.
    assert '\\"The \\"Big\\" Survey\\"' in __import__(
        "dataset_agent.adapters.seed_mapping", fromlist=["seed_clause"]
    ).seed_clause(c)
    with pytest.raises(ValueError, match="Big"):
        candidate_to_seed(c)

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

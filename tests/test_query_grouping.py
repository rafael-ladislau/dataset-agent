"""Every emitted Dimensions ``for`` clause is fully parenthesised.

Measured on Dimensions full_data (2026-11-15): an unparenthesised ``A OR B AND C``
returns exactly ``B AND C`` (``A`` is dropped) and ``NOT`` binds loosest. A clause
must therefore never mix two different operators at one parenthesis level.
"""

from __future__ import annotations

import itertools

import pytest

from dataset_agent.adapters.query_builder import build_for_clause
from dataset_agent.adapters.resource_map import data_wording_clause
from dataset_agent.adapters.seed_mapping import (
    QueryCandidate,
    ai_model_seed_clause,
    candidate_clause,
    complement_clause,
    seed_clause,
)


def operators_by_level(clause: str) -> list[set[str]]:
    """Operators found at each parenthesis group (outside quoted phrases)."""
    stack: list[set[str]] = [set()]
    groups: list[set[str]] = []
    in_quote = False
    i = 0
    word = ""

    def flush() -> None:
        nonlocal word
        if word in ("OR", "AND", "NOT"):
            stack[-1].add(word)
        word = ""

    while i < len(clause):
        ch = clause[i]
        if ch == "\\" and i + 1 < len(clause):
            if clause[i + 1] == '"':
                flush()
                in_quote = not in_quote
            i += 2
            continue
        if in_quote:
            i += 1
            continue
        if ch == "(":
            flush()
            stack.append(set())
        elif ch == ")":
            flush()
            groups.append(stack.pop())
        elif ch.isalpha():
            word += ch
        else:
            flush()
        i += 1
    flush()
    assert not in_quote and len(stack) == 1, clause
    groups.append(stack.pop())
    return groups


def assert_grouped(clause: str) -> None:
    for ops in operators_by_level(clause):
        assert len(ops) <= 1, f"operators {sorted(ops)} share one level in {clause!r}"


def test_helper_flags_the_measured_drop_shape() -> None:
    with pytest.raises(AssertionError):
        assert_grouped('(\\"A\\") OR ((\\"B\\") AND (\\"C\\")) NOT (\\"D\\")')
    with pytest.raises(AssertionError):
        assert_grouped('\\"A\\" OR \\"B\\" AND \\"C\\"')
    assert_grouped('((\\"A\\") OR ((\\"B\\") AND (\\"C\\"))) NOT (\\"D\\")')


GROUPS = {
    "safe": ["All of Us Research Program", "Precision Medicine Initiative Cohort"],
    "risky": ["AOU", "All of Us"],
    "flag_terms": ["NIH", "Researcher Workbench"],
    "exclusion_terms": ["aurora borealis", "randomized clinical trials"],
}


@pytest.mark.parametrize("variant", ["V1", "V2", "V3", "V4", "V5"])
@pytest.mark.parametrize("server_quoting", [False, True])
def test_every_combination_is_grouped(variant: str, server_quoting: bool) -> None:
    names = list(GROUPS)
    for mask in itertools.product([False, True], repeat=len(names) + 1):
        kw = {n: (GROUPS[n] if on else []) for n, on in zip(names, mask[:-1])}
        if mask[-1]:
            kw["tier_hybrid_terms"] = ["Controlled Tier", "Registered Tier"]
            kw["tier_hybrid_qualifier"] = "All of Us"
        clause = build_for_clause(variant=variant, server_quoting=server_quoting, **kw)
        assert_grouped(clause)
        assert not clause.startswith("NOT"), (kw, clause)


def test_v4_wraps_the_positive_body_before_not() -> None:
    clause = build_for_clause(
        safe=["Microsoft Aurora"], variant="V4", risky=["Aurora"],
        flag_terms=["GraphCast"], exclusion_terms=["aurora borealis"],
    )
    assert clause == (
        '((\\"Microsoft Aurora\\") OR ((\\"Aurora\\") AND (\\"GraphCast\\")))'
        ' NOT (\\"aurora borealis\\")'
    )


def test_v4_single_group_body_is_not_double_wrapped() -> None:
    clause = build_for_clause(
        safe=["PhaseNet"], variant="V4", exclusion_terms=["Phase Unwrapping"],
    )
    assert clause == '(\\"PhaseNet\\") NOT (\\"Phase Unwrapping\\")'


def test_v4_without_a_positive_part_emits_nothing() -> None:
    """A bare ``NOT (E)`` would search everything outside E."""
    assert build_for_clause(safe=[], variant="V4", exclusion_terms=["x"]) == ""


def test_seed_shapes_are_grouped() -> None:
    c = QueryCandidate(
        main_dataset_name="All of Us", safe=GROUPS["safe"], risky=GROUPS["risky"],
        flag_terms=GROUPS["flag_terms"], exclude_terms=GROUPS["exclusion_terms"],
    )
    assert_grouped(candidate_clause(c))
    assert_grouped(seed_clause(c))
    assert_grouped(complement_clause("All of Us Research Hub", c))
    assert_grouped(complement_clause("All of Us Research Hub", c, server_quoting=True))
    for names, flags, ex in itertools.product(
        [["Aurora", "Microsoft Aurora"]], [[], ["GraphCast"]], [[], ["aurora borealis"]]
    ):
        assert_grouped(ai_model_seed_clause(names, flags, ex))


def test_data_wording_clause_is_grouped() -> None:
    from types import SimpleNamespace

    rmap = SimpleNamespace(
        program_names=["All of Us Research Program", "AOU"],
        specific_data_names=["All of Us Registered Tier"],
        data_resource_names=["Researcher Workbench"],
        data_name_hits={},
    )
    clause = data_wording_clause(rmap)
    assert clause and " AND " in clause
    assert_grouped(clause)

"""Resource map: deterministic split rule over an LLM proposal (no network)."""

from __future__ import annotations

import pytest

from dataset_agent.adapters.resource_map import (
    build_map,
    candidates,
    cooccurrence_clause,
    propose,
    release_year,
)

PAGES = ("The Example Program, formerly the Parent Cohort Program, enrolls people. "
         "Approved researchers use the Example Workbench and the Controlled Tier. "
         "The first data release to researchers was in May 2020. Funded by the Agency.")


def _raw(**over):
    raw = {
        "names": [{"name": "Example Program", "kind": "program"},
                  {"name": "Example Program Cohort", "kind": "program"},
                  {"name": "Agency", "kind": "organization"},
                  {"name": "Example Workbench", "kind": "data_resource"}],
        "data_resource_names_in_pages": ["Controlled Tier", "Hidden Portal"],
        "former_or_parent_program_names": ["Parent Cohort Program", "Invented Name"],
        "first_data_release_year": 2020,
        "first_data_release_quote": "The first data release to researchers was in May 2020.",
    }
    raw.update(over)
    return raw


def _count(hits):
    calls = []

    def count(clause):
        calls.append(clause)
        return next((n for k, n in hits.items() if k in clause), 0)

    count.calls = calls
    return count


def test_split_needs_a_program_main_name_and_literal_data_names():
    progs, data = candidates(_raw(), dataset_name="Example Program", pages=PAGES)
    assert progs == ["Example Program", "Example Program Cohort", "Parent Cohort Program"]
    assert data == ["Example Workbench", "Controlled Tier"]  # Hidden Portal is not in the pages


def test_main_name_that_is_data_gives_no_map():
    raw = _raw(names=[{"name": "Example Program", "kind": "data_resource"},
                      {"name": "Example Workbench", "kind": "data_resource"}])
    assert candidates(raw, dataset_name="Example Program", pages=PAGES) is None


def test_data_name_inside_a_program_name_does_not_count():
    raw = _raw(names=[{"name": "Example Program", "kind": "program"},
                      {"name": "Example", "kind": "data_resource"}],
               data_resource_names_in_pages=[])
    assert candidates(raw, dataset_name="Example Program", pages=PAGES) is None


def test_map_needs_dimensions_hits_with_the_program():
    count = _count({"Example Workbench": 40})
    m = build_map(_raw(), dataset_name="Example Program", pages=PAGES, count=count)
    assert m.data_resource_names == ["Example Workbench"] and m.cooccurrence == {"Example Workbench": 40}
    assert m.former_or_parent_names == ["Parent Cohort Program"] and m.first_release_year == 2020
    assert len(count.calls) == 3  # two co-occurrence counts, one count of the kept name alone
    assert build_map(_raw(), dataset_name="Example Program", pages=PAGES, count=_count({})) is None


def test_clause_ors_the_program_names():
    assert cooccurrence_clause(["A B", "C"], "D E") == '(\\"A B\\" OR \\"C\\") AND (\\"D E\\")'


def test_release_year_needs_a_quote_from_the_pages():
    assert release_year(_raw(), PAGES) == 2020
    assert release_year(_raw(first_data_release_quote="Data came out in 2020."), PAGES) is None
    assert release_year(_raw(first_data_release_year=None), PAGES) is None


def test_failed_call_is_an_empty_proposal():
    class Boom:
        def get_structured(self, *_a):
            raise RuntimeError("down")

    assert propose(Boom(), dataset_name="E", description="", names=[], pages="") == {}


def test_map_for_record_asks_with_the_research_names_and_pages():
    from dataset_agent.adapters.resource_map import gather_pages, map_for_record
    from dataset_agent.domain.models import DatasetRecord, Group, YearsRange

    rec = DatasetRecord(engine="dimensions", group=Group(name="default"),
                        main_dataset_name="Example Program", description="A cohort.",
                        dataset_names=["Example Program Cohort"], flag_terms=["Agency"],
                        years_range=YearsRange(start_year=2015, end_year=2026))

    class Agent:
        prompts = []

        def get_structured(self, prompt, _tool):
            self.prompts.append(prompt)
            return _raw()

    agent = Agent()
    m = map_for_record(agent, rec, count=_count({"Workbench": 5}), pages=PAGES)
    assert m is not None and m.data_resource_names == ["Example Workbench"]
    assert "- Example Program Cohort" in agent.prompts[0] and "- Agency" in agent.prompts[0]
    pages = gather_pages("X", "http://h", search=lambda q: f"[{q}]", fetch=lambda u: f"<{u}>")
    assert pages == ("<http://h>\n[X data access researchers]\n"
                     "[X history formerly known as launched by initiative]\n"
                     "[X first data release to researchers year]")


def test_release_year_is_confirmed_by_counts():
    from dataset_agent.adapters.resource_map import confirm_release_year

    def where(before):
        return lambda clause, w: before if w else 100

    assert confirm_release_year(2020, ["P"], ["D"], count=where(3), max_share_before=0.05) == (2020, 0.03)
    assert confirm_release_year(2022, ["P"], ["D"], count=where(20), max_share_before=0.05) == (None, 0.2)
    m = build_map(_raw(), dataset_name="Example Program", pages=PAGES,
                  count=_count({"Example Workbench": 40}), count_where=where(30))
    assert m.proposed_release_year == 2020 and m.first_release_year is None
    assert m.share_before_release == 0.3


def test_main_name_that_is_a_data_noun_is_never_split():
    for name in ("Example Data Commons", "Example Survey", "Example Cohort", "Example Database"):
        raw = _raw(names=[{"name": name, "kind": "program"},
                          {"name": "Example Workbench", "kind": "data_resource"}])
        assert candidates(raw, dataset_name=name, pages=PAGES + " " + name) is None


def test_cooccurrence_leaves_out_program_names_inside_the_data_name():
    raw = _raw(names=[{"name": "Example Program", "kind": "program"},
                      {"name": "Example Program Cohort", "kind": "program"},
                      {"name": "Example Program Workbench", "kind": "data_resource"}],
               data_resource_names_in_pages=[], former_or_parent_program_names=[])
    pages = PAGES + " Use the Example Program Workbench."
    count = _count({"Example Program Workbench": 9})
    m = build_map(raw, dataset_name="Example Program", pages=pages, count=count)
    assert m.data_resource_names == ["Example Program Workbench"]
    assert all('\\"Example Program\\" OR' not in c for c in count.calls)
    assert '(\\"Example Program Cohort\\") AND (\\"Example Program Workbench\\")' in count.calls[0]
    only = _raw(names=[{"name": "Example Program", "kind": "program"},
                       {"name": "Example Program Workbench", "kind": "data_resource"}],
                data_resource_names_in_pages=[], former_or_parent_program_names=[])
    assert build_map(only, dataset_name="Example Program", pages=pages,
                     count=_count({"Workbench": 9})) is None


def test_remainder_of_a_program_prefixed_data_name_is_kept_too():
    from dataset_agent.adapters.resource_map import remainders

    assert remainders(["Example Program Researcher Workbench", "Data Browser"],
                      ["Example", "Example Program"]) == ["Researcher Workbench"]
    raw = _raw(names=[{"name": "Example Program", "kind": "program"},
                      {"name": "Example Program Researcher Workbench", "kind": "data_resource"}],
               data_resource_names_in_pages=[], former_or_parent_program_names=[])
    pages = PAGES + " Approved users log in to the Example Program Researcher Workbench."
    m = build_map(raw, dataset_name="Example Program", pages=pages,
                  count=_count({"Researcher Workbench": 30}))
    # the full name contains every program name (not measurable); its remainder is kept
    assert m.data_resource_names == ["Researcher Workbench"]
    raw2 = _raw(names=[{"name": "Example Program", "kind": "program"},
                       {"name": "Example Program Cohort", "kind": "program"},
                       {"name": "Example Program Researcher Workbench", "kind": "data_resource"}],
                data_resource_names_in_pages=[], former_or_parent_program_names=[])
    m = build_map(raw2, dataset_name="Example Program", pages=pages,
                  count=_count({"Researcher Workbench": 30}))
    assert m.data_resource_names == ["Example Program Researcher Workbench", "Researcher Workbench"]


def test_specific_names_need_a_high_program_share_and_a_proper_name():
    from dataset_agent.adapters.resource_map import is_generic_data_name, specific_names

    assert is_generic_data_name("Data Browser") and is_generic_data_name("Public Tier dataset")
    assert not is_generic_data_name("Researcher Workbench")
    hits = {"Researcher Workbench": 900, "Data Browser": 300, "Example Hub": 10}
    alone = {"Researcher Workbench": 1200, "Data Browser": 40_000, "Example Hub": 100}
    assert specific_names(hits, alone, min_share=0.5, max_hits=500_000) == ["Researcher Workbench"]
    assert specific_names(hits, alone, min_share=0.5, max_hits=1000) == []


def test_failed_home_page_falls_back_to_more_searches():
    from dataset_agent.adapters.resource_map import gather_pages_with_sources

    text, sources = gather_pages_with_sources("X", "http://h", search=lambda q: f"[{q}]",
                                              fetch=lambda u: "")
    assert sources[0] == "failed: http://h" and len(sources) == 6
    assert "[X data portal researcher platform]" in text
    text, sources = gather_pages_with_sources("X", "http://h", search=lambda q: f"[{q}]",
                                              fetch=lambda u: "home")
    assert sources[0] == "http://h" and len(sources) == 4


def test_a_remainder_needs_half_of_its_hits_with_the_program():
    from dataset_agent.adapters.resource_map import check_data_names

    def count(clause):
        if "AND" in clause:
            return 100
        return 2000 if "Research Hub" in clause else 120

    found: dict = {}
    hits, _ = check_data_names(["Example Program Research Hub", "Example Workbench"],
                               ["Example Program", "Example Cohort"], count=count, found=found)
    assert "Research Hub" not in hits and "Example Workbench" in hits
    assert found["research hub"]["reason"] == "only 100 of 2000 hits carry a program name"


def test_map_is_undecided_when_every_count_failed():
    # Dimensions outage, throttling or an exhausted budget: no count is not "no split".
    from dataset_agent.adapters.resource_map import ResourceMapUndecided

    calls = []

    def failed(clause):
        calls.append(clause)
        return None

    with pytest.raises(ResourceMapUndecided, match="every count failed"):
        build_map(_raw(), dataset_name="Example Program", pages=PAGES, count=failed)
    assert calls  # candidates existed and were counted


def test_low_counts_or_one_failed_count_stay_a_firm_answer():
    assert build_map(_raw(), dataset_name="Example Program", pages=PAGES, count=_count({})) is None

    def one_failed(clause):
        return None if "Example Workbench" in clause else 0

    assert build_map(_raw(), dataset_name="Example Program", pages=PAGES, count=one_failed) is None

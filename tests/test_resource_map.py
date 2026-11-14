"""Resource map: deterministic split rule over an LLM proposal (no network)."""

from __future__ import annotations

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
    assert len(count.calls) == 2
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

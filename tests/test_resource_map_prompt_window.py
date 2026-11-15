"""Round 4, item 4: every search's snippets reach the resource-map LLM, whatever pages
were fetched, and failed fetches do not use up the discovery fetch cap (fake search and
fetch; no network)."""

from __future__ import annotations

from dataset_agent.adapters.resource_map import (
    discover_from_web,
    discovery_pages,
    map_for_record,
)
from dataset_agent.domain.models import DatasetRecord, Group, ResourceMap, YearsRange

LONG_PAGE = "Example Program news and stories about participants. " * 1200  # ~64k chars


class Capture:
    def __init__(self):
        self.prompts: list[str] = []

    def get_structured(self, prompt, _tool):
        self.prompts.append(prompt)
        return {}


def _search(q):
    slug = abs(hash(q)) % 10_000
    body = ("The Registered Tier and Controlled Tier hold the data."
            if "tiers" in q else "General information about the program.")
    return (f"- result one for {q}\n  https://site.example.org/{slug}/a\n  {body}\n"
            f"- result two for {q}\n  https://site.example.org/{slug}/b\n  more text")


def test_tiers_snippets_reach_the_llm_when_long_pages_were_fetched():
    agent = Capture()
    rmap = ResourceMap(program_names=["Example Program"], data_resource_names=["Example Workbench"])
    discover_from_web(agent, rmap, dataset_name="Example Program", description="",
                      count=lambda c: 10, search=_search, fetch=lambda u: LONG_PAGE)
    assert agent.prompts, "propose was not called"
    assert "Registered Tier and Controlled Tier" in agent.prompts[0]
    # Every search's snippets are in the window.
    assert agent.prompts[0].count("General information about the program.") == 4


def test_map_searches_reach_the_llm_when_the_home_page_is_long():
    agent = Capture()

    def search(q):
        return f"- r\n  https://x.example.org/1\n  snippet for {q}"

    rec = DatasetRecord(engine="dimensions", group=Group(name="default"),
                        main_dataset_name="Example Program", description="d",
                        dataset_names=[], flag_terms=[], home_url="https://home.example.org",
                        years_range=YearsRange(start_year=2015, end_year=2026))
    map_for_record(agent, rec, count=lambda c: 10, search=search, fetch=lambda u: LONG_PAGE)
    p = agent.prompts[0]
    for q in ("data access researchers", "history formerly known", "first data release"):
        assert f"snippet for Example Program {q}" in p


def test_failed_fetches_do_not_count_toward_the_page_cap():
    calls: list[str] = []

    def fetch(u):
        calls.append(u)
        return "" if len(calls) <= 6 else f"page {u}"  # the first six fetches fail

    text, sources = discovery_pages("Example Program", search=_search, fetch=fetch,
                                    per_query=2, max_pages=6)
    ok = [s for s in sources if s.startswith("https://")]
    assert len(calls) > 6  # queries 4 and 5 still get their pages fetched
    assert len(ok) == len(calls) - 6 and ok

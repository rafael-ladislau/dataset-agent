"""Round 3, item 2: for a split dataset the release bar counts program-only papers as
genuine; the scope check labels them, and the data, program-only and used shares are
reported next to the precision, each with a Wilson interval."""

from __future__ import annotations

from dataset_agent.adapters.fulltext_evidence import Evidence, PubVerdict
from dataset_agent.adapters.scope_check import check_scope


def _v(pid, verdict, text="The Example Program aims to enrol one million people."):
    return PubVerdict(publication_id=pid, doi=None, title=f"T {pid}", citation_count=1,
                      verdict=verdict, quote="q",
                      contexts=[{"term": "Example Program", "role": "alias", "part": "body",
                                 "text": text}])


class Judge:
    def get_structured(self, prompt, _tool):
        return {"scope": "program_only"}


def _ev():
    ev = Evidence(verdicts=[_v("u", "used"), _v("p", "mentioned"), _v("n", "not_mentioned"),
                            _v("r", "mentioned")])
    ev.verdicts[3].contexts[0]["part"] = "references"
    ev.verdicts[3].quote = ""  # no text quote: the references-only rule applies
    check_scope(ev, agent=Judge(), dataset_name="Example Program", description="",
                program_names=["Example Program"], data_names=["Example Workbench"])
    return ev


def test_program_only_keeps_its_verdict_and_carries_a_label():
    ev = _ev()
    p = ev.verdicts[1]
    assert p.scope == "program_only" and p.verdict == "mentioned" and p.genuine
    assert p.overridden_from is None
    assert ev.n_program_only == 1
    # References-only citations are still not genuine.
    assert not ev.verdicts[3].genuine and ev.verdicts[3].scope_references_only


def test_precision_counts_program_only_and_shares_are_reported():
    ev = _ev()
    assert ev.corrected_precision == 0.5  # used + program-only of 4
    assert ev.raw_precision == 0.75  # adds back the references-only drop only
    m = ev.metrics(0, "clause", None)
    assert m.fulltext_precision == 0.5
    assert m.data_share == 0.25 and m.program_only_rate == 0.25 and m.used_rate == 0.25
    for name in ("fulltext_precision_wilson95", "data_share_wilson95",
                 "program_only_wilson95", "used_share_wilson95"):
        lo, hi = getattr(m, name)
        assert 0.0 <= lo <= hi <= 1.0


def test_without_a_scope_check_no_shares():
    ev = Evidence(verdicts=[_v("u", "used"), _v("p", "mentioned")])
    m = ev.metrics(0, "clause", None)
    assert m.data_share is None and m.data_share_wilson95 is None
    assert m.program_only_wilson95 is None


def test_a_probe_keeps_program_only_papers_genuine(tmp_path):
    import asyncio
    from unittest.mock import MagicMock

    from dataset_agent.application.query_optimization import QueryOptimizationUseCase
    from dataset_agent.domain.models import OptimizeRequest, ResourceMap
    from tests.test_fulltext_loop import CountDSL, FakeClient, make_settings

    pub = {"publication_id": "a", "doi": None, "title": "T", "citation_count": 1,
           "fulltext": {"status": "delivered", "source": "pmc", "reason": None},
           "analyzed": True,
           "validated_datasets": [{"name": "X", "is_used": False, "is_mentioned": True,
                                   "text_quote": "q"}],
           "term_contexts": [{"term": "Example Program", "role": "alias", "part": "body",
                              "text": "The Example Program aims to enrol people."}]}
    uc = QueryOptimizationUseCase(dsl_port=CountDSL(), research=MagicMock(), agent=MagicMock(),
                                  settings=make_settings(tmp_path),
                                  extraction_client=FakeClient([{"publications": [pub]}]),
                                  referent_agent=Judge())
    uc._resource_map = ResourceMap(program_names=["Example Program"],
                                   data_resource_names=["Example Workbench"])
    from dataset_agent.adapters.seed_mapping import QueryCandidate

    cand = QueryCandidate(main_dataset_name="Example Program", safe=["Example Program"])
    ev = asyncio.run(uc._probe_candidate(cand, OptimizeRequest(dataset_name="Example Program")))
    m = ev.metrics(0, "c", None)
    assert m.fulltext_precision == 1.0 and m.program_only_rate == 1.0 and m.data_share == 0.0

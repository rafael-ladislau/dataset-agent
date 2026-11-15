"""One references-only rule on both sides: a paper the scope check turns to
not_mentioned as references-only is references-only in the four-way split,
one rule shared with fulltext_evidence._references_only."""

from __future__ import annotations

import pytest

from dataset_agent.adapters.fulltext_evidence import (
    Evidence,
    PubVerdict,
)
from dataset_agent.adapters.scope_check import check_scope

PROGRAM = ["Example Program"]
REF = {"term": "Example Program", "role": "alias", "part": "references",
       "text": "The Example Program cohort profile. 2019."}


class _Judge:
    def get_structured(self, _prompt, _tool):
        return {"scope": "data"}


def _run(contexts, quote="The Example Program cohort profile"):
    ev = Evidence(entity="dataset")
    ev.verdicts.append(PubVerdict(publication_id="p", doi=None, title="", citation_count=0,
                                  verdict="mentioned", quote=quote, contexts=contexts))
    check_scope(ev, agent=_Judge(), dataset_name="Example", description="A cohort.",
                program_names=PROGRAM, data_names=["Example Program Workbench"])
    return ev.verdicts[0]


@pytest.mark.parametrize("contexts", [
    [REF],
    # A body context whose text was not extracted.
    [REF, {"term": "Example Program", "role": "alias", "part": "body", "text": ""}],
    # Only a flag snippet, in the references.
    [dict(REF, role="flag")],
])
def test_a_paper_the_scope_check_turns_is_references_only(contexts):
    v = _run(contexts)
    if v.scope_references_only:
        assert v.verdict == "not_mentioned" and v.four_way == "references_only"
    else:
        assert v.verdict == "mentioned"


def test_a_snippet_only_in_the_references_is_turned():
    assert _run([REF]).scope_references_only

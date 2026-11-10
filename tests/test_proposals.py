from __future__ import annotations


import pytest
from fastapi.testclient import TestClient

from dataset_agent.adapters.proposals_json import JsonProposalRepository
from dataset_agent.adapters.seed_mapping import QueryCandidate
from dataset_agent.application.proposal_service import (
    ProposalNotFound,
    ProposalService,
    ProposalStateError,
    build_proposal,
    merge_diffs,
)
from dataset_agent.domain.proposals import TermDiff
from dataset_agent.interfaces import api
from dataset_agent.settings import Settings


class FakeClient:
    def __init__(self):
        self.payloads = []

    def submit(self, payload):
        self.payloads.append(payload)
        return {"task_id": "T9"}


def make(tmp_path, **kw):
    s = Settings(output_dir=tmp_path / "out", tasks_db=tmp_path / "t.db", **kw)
    repo = JsonProposalRepository(s.output_dir)
    init = QueryCandidate("X", safe=["Alpha Long", "Beta Long"], risky=["AB"], flag_terms=["health"])
    prop = QueryCandidate("X", safe=["Alpha Long"], risky=["AB"], flag_terms=["health"], exclude_terms=["nursing"])
    p = build_proposal(dataset_name="X", initial=init, proposed=prop,
                       diffs=[TermDiff(kind="alias", term="Beta Long", action="drop", status="verified")],
                       iterations=[], settings=s)
    repo.save(p)
    return s, repo, p


def test_proposal_contents(tmp_path):
    _, _, p = make(tmp_path)
    assert p.status == "pending"
    assert p.start_payload["probe"] is False
    assert p.start_payload["exclude_terms"] == ["nursing"]
    assert p.initial.seed["standalone_names"] == ["Alpha Long", "Beta Long"]
    assert p.proposed.for_clause


def test_merge_diffs_change_beats_keep():
    keep = TermDiff(kind="alias", term="A", action="keep", status="unverified")
    drop = TermDiff(kind="alias", term="a", action="drop", status="verified")
    assert [d.action for d in merge_diffs([[keep], [drop]])] == ["drop"]
    assert [d.action for d in merge_diffs([[drop], [keep]])] == ["drop"]


def test_submit_requires_approval_and_defaults_to_dev(tmp_path):
    s, repo, p = make(tmp_path)
    fake, targets = FakeClient(), []
    svc = ProposalService(repo, s, client_factory=lambda t: (targets.append(t), fake)[1])
    with pytest.raises(ProposalStateError):
        svc.submit(p.id)
    assert fake.payloads == []
    svc.approve(p.id, "rafael")
    out = svc.submit(p.id)
    assert targets == ["dev"] and out.status == "submitted" and out.submit_result == {"task_id": "T9"}
    assert fake.payloads[0]["probe"] is False
    with pytest.raises(ProposalStateError):   # no double submit
        svc.submit(p.id)
    with pytest.raises(ProposalNotFound):
        svc.get("doesnotexist1")


def test_prod_needs_explicit_setting_and_approval(tmp_path):
    s, repo, p = make(tmp_path, submit_target="prod")
    fake, targets = FakeClient(), []
    svc = ProposalService(repo, s, client_factory=lambda t: (targets.append(t), fake)[1])
    with pytest.raises(ProposalStateError):
        svc.submit(p.id)
    assert targets == []
    svc.approve(p.id)
    svc.submit(p.id)
    assert targets == ["prod"]


def test_api_flow(tmp_path, monkeypatch):
    s, repo, p = make(tmp_path)
    fake = FakeClient()
    from dataset_agent.application import proposal_service as ps

    monkeypatch.setattr(ps.ExtractionClient, "for_target", classmethod(lambda cls, st, t: fake))
    api.app.dependency_overrides[api.get_settings] = lambda: s
    try:
        c = TestClient(api.app)
        assert c.get(f"/proposals/{p.id}").json()["status"] == "pending"
        assert c.post(f"/proposals/{p.id}/submit").status_code == 409
        assert c.post(f"/proposals/{p.id}/approve", json={"approved_by": "rafael"}).json()["status"] == "approved"
        r = c.post(f"/proposals/{p.id}/submit")
        assert r.status_code == 200 and r.json()["submit_target"] == "dev"
        assert c.get("/proposals/nope").status_code == 404
    finally:
        api.app.dependency_overrides.clear()

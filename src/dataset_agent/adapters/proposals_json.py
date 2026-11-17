"""JSON-file persistence for query proposals (next to ``optimize-*.json`` results)."""

from __future__ import annotations

import re
from pathlib import Path

from dataset_agent.domain.proposals import QueryProposal

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class JsonProposalRepository:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path(self, proposal_id: str) -> Path:
        if not _ID_RE.match(proposal_id or ""):
            raise KeyError(proposal_id)
        return self._root / f"proposal-{proposal_id}.json"

    def save(self, proposal: QueryProposal) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._path(proposal.id).write_text(proposal.model_dump_json(indent=2), encoding="utf-8")

    def get(self, proposal_id: str) -> QueryProposal | None:
        try:
            p = self._path(proposal_id)
        except KeyError:
            return None
        if not p.is_file():
            return None
        return QueryProposal.model_validate_json(p.read_text(encoding="utf-8"))

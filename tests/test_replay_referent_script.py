"""Pure parts of scripts/replay_referent_check.py (fake judge, no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "replay_referent_check", ROOT / "scripts" / "replay_referent_check.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

ANCHOR = "PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method"
ROWS = [
    {"publication_id": "p0", "title": "PhaseNet paper", "verdict": "used",
     "text_quote": f"Zhu and Beroza. {ANCHOR}. GJI."},
    {"publication_id": "p1", "title": "Optical metrology", "verdict": "mentioned",
     "text_quote": "PhaseNet for phase unwrapping."},
    {"publication_id": "p2", "title": "Video", "verdict": "mentioned",
     "text_quote": "PhaseNet interpolates frames."},
    {"publication_id": "p3", "title": "Quakes", "verdict": "mentioned",
     "text_quote": "We pick with PhaseNet."},
    {"publication_id": "p4", "title": "Other", "verdict": "not_mentioned", "text_quote": ""},
]
LABELS = {"other": [1, 2], "criteria": {"other_kept_max": 0, "same_lost_max": 0,
                                         "corrected_precision_max": 0.6}}


class Judge:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def get_structured(self, prompt, _tool):
        self.prompts.append(prompt)
        return {"is_genuine_dataset_reference": self.answer(prompt)}


def _seismic(prompt):
    return "unwrapping" not in prompt and "interpolates" not in prompt


def test_rows_become_model_evidence_with_the_quote_as_passage():
    ev = replay.evidence_from_rows(ROWS, ANCHOR and [ANCHOR], model_name="PhaseNet")
    assert ev.entity == "ai_model" and ev.n_evaluated == 5 and len(ev.genuine) == 4
    # The quote that holds the anchor title is a reference line (stage 0).
    assert ev.verdicts[0].contexts[0]["part"] == "references"
    assert ev.verdicts[1].contexts == []


def test_one_replay_scores_against_the_labels():
    judge = Judge(_seismic)
    out = replay.replay_once(ROWS, LABELS, agent=judge, model_name="PhaseNet",
                             profile="P.", anchor_titles=[ANCHOR], votes=3)
    assert out["other_accepted_by_judge"] == 2 and out["other_kept"] == 0
    assert out["same_accepted_by_judge"] == 2 and out["same_lost"] == 0
    assert out["corrected_precision"] == 2 / 5 and out["raw_precision"] == 4 / 5
    assert out["confirmed_by_anchor"] == 1 and out["calls"] == 9
    assert out["passes"] is True
    assert out["decisions"]["p1"] == "rejected" and out["decisions"]["p0"] == "anchor"


def test_agreement_and_overall_pass():
    a = {"decisions": {"x": "rejected", "y": "confirmed"}, "passes": True, "calls": 4,
         "failed_calls": 0}
    b = {"decisions": {"x": "rejected", "y": "rejected"}, "passes": True, "calls": 4,
         "failed_calls": 0}
    assert replay.agreement([a, b]) == 0.5
    summary = replay.summarize_replays([a, a, b], min_agreement=0.9, max_failed_share=0.05)
    assert summary["agreement"] == 4 / 6 and summary["passes"] is False
    summary = replay.summarize_replays([a, a, a], min_agreement=0.9, max_failed_share=0.05)
    assert summary["passes"] is True

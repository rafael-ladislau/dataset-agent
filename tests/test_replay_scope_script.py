"""Pure parts of scripts/replay_scope_check.py (fake judge, no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("replay_scope_check", ROOT / "scripts" / "replay_scope_check.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

TEXT = """# TITLE: A study

## ABSTRACT
We used the Example Program Workbench for analyses.

## BODY: Methods
The Example Program aims to enrol a million people.

## BIBLIOGRAPHY (reference list; names here do not count as mentions)
- Example Program Workbench | Journal
"""
MAP = {"dataset_name": "Example", "program_names": ["Example Program"],
       "data_names": ["Example Program Workbench"], "data_terms": []}


class Judge:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def get_structured(self, prompt, _tool):
        self.prompts.append(prompt)
        return {"scope": self.answer}


def test_parts_and_reference_list():
    parts = replay.split_parts(TEXT)
    assert [p for p, _ in parts] == ["title", "abstract", "body", "references"]


def test_contexts_follow_the_extraction_window_and_parts():
    ctx = replay.cut_contexts(TEXT, ["Example Program Workbench"])
    assert [c["part"] for c in ctx] == ["abstract", "references"]


def test_replay_scores_labels(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(TEXT)
    g = tmp_path / "b.txt"
    g.write_text(TEXT.replace("We used the Example Program Workbench for analyses.", "Policy."))
    entries = [
        {"id": "d", "label": "data", "used": False, "path": str(f)},
        {"id": "p", "label": "program_only", "used": False, "path": str(g)},
        {"id": "b", "label": "data", "used": False, "borderline": True, "path": str(g)},
        {"id": "u", "label": "data", "used": True, "path": str(g)},
        {"id": "n", "label": "not_mentioned", "used": False, "path": str(g)},
    ]
    out = replay.run(entries, MAP, Judge("program_only"), replays=2)
    r = out["replays"][0]
    assert r["by_name"] == 1 and r["calls"] == 2  # d by name, u used, n left out
    assert r["program_only_rejected"] == 1 and r["data_lost"] == 0
    assert r["data_borderline_lost"] == 1 and out["agreement"] == 1.0


def test_agreement_counts_papers_with_one_decision():
    a = {"decisions": {"x": "data", "y": "program_only"}}
    b = {"decisions": {"x": "data", "y": "unclear"}}
    assert replay.agreement([a, b]) == 0.5

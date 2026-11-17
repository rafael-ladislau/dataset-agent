"""Offline recall replay of a final FOR clause on labelled texts (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "replay_query_recall", Path(__file__).resolve().parents[1] / "scripts" / "replay_query_recall.py")
rqr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rqr)


def test_clause_semantics_follow_dimensions():
    clause = ('(\\"Example data\\" OR \\"data from the Example\\") OR '
              '((\\"Example\\") AND (\\"Workbench\\")) NOT (\\"Unrelated\\")')
    tree = rqr.parse(clause)
    assert rqr.evaluate(tree, "We used Example-data from 2020.")
    assert rqr.evaluate(tree, "the Example Workbenches were used")  # plural, any case
    assert not rqr.evaluate(tree, "the Example program enrols people")
    assert not rqr.evaluate(tree, "Example data and an Unrelated topic")  # NOT applies to the group
    assert rqr.phrases(tree) == ["Example data", "data from the Example", "Example",
                                 "Workbench", "Unrelated"]


def test_and_binds_tighter_than_or():
    tree = rqr.parse('"A" OR "B" AND "C"')
    assert rqr.evaluate(tree, "A alone")
    assert not rqr.evaluate(tree, "B alone")


def _entry(tmp_path, pid, label, text, **kw):
    p = tmp_path / f"{pid}.txt"
    p.write_text(text)
    return {"id": pid, "label": label, "path": str(p), **kw}


def test_recall_groups_and_bars(tmp_path):
    entries = [
        _entry(tmp_path, "u1", "data", "Example data were analysed.", used=True),
        _entry(tmp_path, "u2", "data", "We used the Researcher Workbench.", used=True),
        _entry(tmp_path, "d1", "data", "data from the Example were cited"),
        _entry(tmp_path, "b1", "data", "biobanks such as Example", borderline=True),
        _entry(tmp_path, "p1", "program_only", "Example aims to enrol one million."),
        _entry(tmp_path, "n1", "not_mentioned", "Example data"),
        {"id": "x", "label": "data", "path": str(tmp_path / "missing.txt")},
    ]
    out = rqr.recall(entries, '"Example data" OR "data from the Example"')
    g = out["groups"]
    assert (g["data_firm"]["matched"], g["data_firm"]["n"]) == (2, 3)
    assert (g["used"]["matched"], g["used"]["n"]) == (1, 2)
    assert g["data_borderline"]["matched"] == 0 and g["program_only_firm"]["matched"] == 0
    assert "not_mentioned" not in g
    assert out["bars"] == {"data_firm": False, "used": False, "program_only_firm": True}
    assert not out["passed"] and out["missed_data"] == ["u2"]
    assert out["papers"]["u1"]["terms"] == ["Example data"]


def test_references_can_be_left_out(tmp_path):
    e = _entry(tmp_path, "r", "data", "Body text.\n## BIBLIOGRAPHY\nExample data 2020.", used=True)
    assert rqr.recall([e], '"Example data"')["groups"]["used"]["matched"] == 1
    assert rqr.recall([e], '"Example data"', references=False)["groups"]["used"]["matched"] == 0

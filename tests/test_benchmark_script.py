from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_fulltext_refinement.py"
spec = importlib.util.spec_from_file_location("bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_normalize_and_bool():
    assert bench.normalize_pub_id("https://doi.org/10.1/ABC") == "10.1/abc"
    assert bench.parse_bool("Yes") is True and bench.parse_bool("0") is False
    assert bench.parse_bool("maybe") is None


def test_compute_metrics():
    labels = {"10.1/a": True, "10.1/b": True, "10.1/c": False, "10.1/d": False}
    m = bench.compute_metrics(["10.1/A", "10.1/c", "10.1/zzz"], labels)
    assert m["labelled_retrieved"] == 2 and m["used_retrieved"] == 1
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["label_coverage"] == 2 / 3
    assert bench.compute_metrics([], labels)["precision"] is None


def test_evaluate_end_to_end(tmp_path):
    csv_path = tmp_path / "l.csv"
    csv_path.write_text(
        "publication,dataset,is_used\n10.1/a,NHIS,yes\n10.1/b,NHIS,no\n10.1/c,NHIS,yes\n10.1/x,NHIS,?\n",
        encoding="utf-8")
    labels = bench.load_labels(csv_path)
    assert labels == {"nhis": {"10.1/a": True, "10.1/b": False, "10.1/c": True}}
    seeds = [{"dataset": "NHIS", "initial": {}, "refined": {}}]
    retrieved = {"NHIS": {"initial": ["10.1/a", "10.1/b"], "refined": ["10.1/a", "10.1/c"]}}
    res = bench.evaluate(seeds, labels, retrieved)
    assert res["micro"]["initial"]["precision"] == 0.5
    assert res["micro"]["refined"]["precision"] == 1.0 and res["micro"]["refined"]["recall"] == 1.0
    assert "MICRO" in bench.format_report(res)
    json.dumps(res)


def test_clause_from_buckets():
    c = bench.clause_from_buckets({"safe": ["Alpha Long"], "exclude_terms": ["nursing"]})
    assert "Alpha Long" in c and "NOT" in c
    assert bench.clause_from_buckets({"for_clause": "RAW"}) == "RAW"

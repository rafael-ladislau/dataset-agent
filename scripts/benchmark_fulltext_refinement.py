#!/usr/bin/env python3
"""Benchmark initial vs refined queries against human is_used labels.

Inputs
------
``--seeds seeds.json``
    A list of ``{"dataset": "<name>", "initial": <buckets>, "refined": <buckets>}``.
    ``<buckets>`` is ``{"safe": [...], "risky": [...], "flag_terms": [...],
    "exclude_terms": [...]}`` or ``{"for_clause": "<raw clause>"}``. A proposal
    returned by ``GET /proposals/{id}`` can be converted directly: use its
    ``initial`` and ``proposed`` objects.

``--labels labels.csv``
    Columns (header required): ``publication``, ``dataset``, ``is_used``.
    ``publication`` is a DOI (any case, with or without the ``https://doi.org/``
    prefix) or a Dimensions id. ``is_used`` is yes/no, true/false or 1/0.

``--retrieved retrieved.json`` (offline) or ``--fetch`` (live Dimensions)
    ``retrieved.json`` maps ``dataset -> {"initial": [ids...], "refined": [ids...]}``.
    With ``--fetch`` the script runs both clauses against Dimensions (needs
    ``DATASET_AGENT_DIMENSIONS_API_KEY``; sorted by citations, up to ``--limit``
    publications) and writes the result to ``--retrieved-out`` so reruns are offline.

Metrics (per dataset and micro-averaged), computed over human-labelled publications:

* precision = |retrieved and labelled used| / |retrieved and labelled|
* recall    = |retrieved and labelled used| / |labelled used|
* label_coverage = |retrieved and labelled| / |retrieved| (how much of the result the labels judge)

Exporting the human-validated food-security data
------------------------------------------------
Source (read-only, do not copy it into this repository)::

    digitalScience/nature/nature_review_analysis/human_validation/
      internal/validation_food_security_completes_with_LLM*.xlsx   # validator answers
      validation_master_food_security.xlsx                         # Publication ID, DOI, Dataset Name

(see ``INDEX.md`` item 8, "Human validation sampling"). Join the completed validator
sheet to the master on ``Sample ID`` and write one CSV row per (publication, dataset):
``publication`` = ``DOI`` (fallback ``Publication ID``), ``dataset`` = ``Dataset Name``,
``is_used`` = the validator's Yes/No usage column. Keep the CSV outside the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

_TRUE = {"yes", "y", "true", "1", "used"}
_FALSE = {"no", "n", "false", "0", "not used"}


def normalize_pub_id(value: str) -> str:
    v = (value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if v.startswith(prefix):
            v = v[len(prefix):]
    return v


def parse_bool(value: str) -> bool | None:
    v = (value or "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def load_labels(path: Path) -> dict[str, dict[str, bool]]:
    """``dataset(lower) -> {publication -> is_used}``; rows with unparseable labels are skipped."""
    out: dict[str, dict[str, bool]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            used = parse_bool(row.get("is_used", ""))
            pub = normalize_pub_id(row.get("publication", ""))
            ds = (row.get("dataset") or "").strip().lower()
            if used is None or not pub or not ds:
                continue
            out.setdefault(ds, {})[pub] = used
    return out


def compute_metrics(retrieved: Iterable[str], labels: dict[str, bool]) -> dict[str, Any]:
    """Precision / recall of *retrieved* against human labels (see module docstring)."""
    ret = {normalize_pub_id(r) for r in retrieved if r}
    labelled_ret = {p for p in ret if p in labels}
    used_ret = {p for p in labelled_ret if labels[p]}
    used_all = {p for p, u in labels.items() if u}
    return {
        "retrieved": len(ret),
        "labelled_retrieved": len(labelled_ret),
        "used_retrieved": len(used_ret),
        "used_total": len(used_all),
        "precision": len(used_ret) / len(labelled_ret) if labelled_ret else None,
        "recall": len(used_ret) / len(used_all) if used_all else None,
        "label_coverage": len(labelled_ret) / len(ret) if ret else None,
    }


def micro_average(per_dataset: list[dict[str, Any]]) -> dict[str, Any]:
    lr = sum(m["labelled_retrieved"] for m in per_dataset)
    ur = sum(m["used_retrieved"] for m in per_dataset)
    ut = sum(m["used_total"] for m in per_dataset)
    rt = sum(m["retrieved"] for m in per_dataset)
    return {
        "retrieved": rt,
        "labelled_retrieved": lr,
        "used_retrieved": ur,
        "used_total": ut,
        "precision": ur / lr if lr else None,
        "recall": ur / ut if ut else None,
        "label_coverage": lr / rt if rt else None,
    }


def evaluate(
    seeds: list[dict[str, Any]],
    labels: dict[str, dict[str, bool]],
    retrieved: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        name = seed["dataset"]
        lab = labels.get(name.lower(), {})
        got = retrieved.get(name, {})
        rows.append(
            {
                "dataset": name,
                "initial": compute_metrics(got.get("initial", []), lab),
                "refined": compute_metrics(got.get("refined", []), lab),
            }
        )
    return {
        "datasets": rows,
        "micro": {
            "initial": micro_average([r["initial"] for r in rows]),
            "refined": micro_average([r["refined"] for r in rows]),
        },
    }


def clause_from_buckets(b: dict[str, Any]) -> str:
    if b.get("for_clause"):
        return b["for_clause"]
    from dataset_agent.adapters.query_builder import build_for_clause

    return build_for_clause(
        safe=b.get("safe", []),
        variant="V4",
        risky=b.get("risky", []),
        flag_terms=b.get("flag_terms", []),
        exclusion_terms=b.get("exclude_terms", []),
    )


async def _fetch_one(dsl: Any, clause: str, limit: int) -> list[str]:
    from dataset_agent.adapters.dimensions_dsl import publications_from_result

    q = (
        f'search publications in full_data for "{clause}" '
        f"return publications[id+doi] sort by times_cited limit {limit}"
    )
    res = await dsl.execute_dsl(q)
    ids: list[str] = []
    for p in publications_from_result(res):
        ids.append(str(p.get("doi") or p.get("id") or ""))
    return ids


async def fetch_retrieved(seeds: list[dict[str, Any]], limit: int) -> dict[str, dict[str, list[str]]]:
    from dataset_agent.adapters.dimensions_dsl import AsyncThrottledDimensionsDsl
    from dataset_agent.settings import Settings

    dsl = AsyncThrottledDimensionsDsl(Settings())
    out: dict[str, dict[str, list[str]]] = {}
    for s in seeds:
        out[s["dataset"]] = {
            "initial": await _fetch_one(dsl, clause_from_buckets(s["initial"]), limit),
            "refined": await _fetch_one(dsl, clause_from_buckets(s["refined"]), limit),
        }
    return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def format_report(result: dict[str, Any]) -> str:
    lines = [f"{'dataset':40} {'variant':8} {'prec':>6} {'recall':>6} {'labelled':>8} {'retrieved':>9}"]
    for r in result["datasets"]:
        for v in ("initial", "refined"):
            m = r[v]
            lines.append(
                f"{r['dataset'][:40]:40} {v:8} {_fmt(m['precision']):>6} {_fmt(m['recall']):>6} "
                f"{m['labelled_retrieved']:>8} {m['retrieved']:>9}"
            )
    for v in ("initial", "refined"):
        m = result["micro"][v]
        lines.append(
            f"{'MICRO':40} {v:8} {_fmt(m['precision']):>6} {_fmt(m['recall']):>6} "
            f"{m['labelled_retrieved']:>8} {m['retrieved']:>9}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--retrieved", type=Path, help="offline retrieved-ids JSON")
    ap.add_argument("--fetch", action="store_true", help="query Dimensions live")
    ap.add_argument("--retrieved-out", type=Path, help="where --fetch writes the retrieved ids")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args(argv)

    seeds = json.loads(args.seeds.read_text(encoding="utf-8"))
    labels = load_labels(args.labels)
    if args.fetch:
        retrieved = asyncio.run(fetch_retrieved(seeds, args.limit))
        if args.retrieved_out:
            args.retrieved_out.write_text(json.dumps(retrieved, indent=2), encoding="utf-8")
    elif args.retrieved:
        retrieved = json.loads(args.retrieved.read_text(encoding="utf-8"))
    else:
        ap.error("provide --retrieved or --fetch")
    result = evaluate(seeds, labels, retrieved)
    print(json.dumps(result, indent=2) if args.json else format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

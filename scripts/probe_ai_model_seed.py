#!/usr/bin/env python3
"""Probe one hand-built AI-model seed on extraction full text and print its precision.

Slice-1 measurement tool for the AI-model full-text probe (no pipeline wiring). It
starts a probe task on ``POST /ai-models/start``, tops up open full text batch by batch
like the dataset probe (``ExtractionClient.run_probe_topup(entity="ai_model")``), builds
the evidence from ``llm-results`` and prints one JSON object. The probe task is always
cleaned up and the cleanup items are reported.

``--from-result <result.json>`` probes the shipped ``ai_model_seed`` of an
``optimize-model`` result verbatim (names, flags, excludes, years) and records the seed
clause; a withheld seed or a file that is not JSON is refused.

Seed JSON: at least ``{"model_name": ..., "description": ...}``. Missing start fields
are defaulted (``model_family`` = model_name, ``hugging_face_id`` null, empty
alias/flag/exclude lists, ``years_range`` from ``DATASET_AGENT_DEFAULT_YEARS_START`` /
``_END``, ``dimensions_limit`` null, ``filter_us_affiliation`` false). Unknown keys and
``seed_type`` / ``probe`` / ``probe_pull_cap`` are refused (the server answers 422).

Extraction URL and credentials come from the dataset probe's settings:
``DATASET_AGENT_EXTRACTION_{DEV,PROD}_{URL,USER,PASSWORD}`` (``.env`` is read), with
``--extraction dev|prod`` (default ``DATASET_AGENT_PROBE_TARGET``).

Metrics: precision = papers that mention or use the model / evaluated (a name found only
in the reference list counts as a mention); cited-only = genuine papers whose names occur
only in the reference list, by name location (``n_cited_only`` / genuine, unknown counted
apart); not run = genuine papers the judge labelled ``cited`` (``n_not_run`` / genuine);
used_rate = used / evaluated; Wilson 95% interval on precision; error rows = delivered,
analyzed papers whose validation errored (no verdict); ``n_no_verdict`` = analyzed papers
whose rows hold no judgement; ``n_incoherent`` = papers judged not this model whose row
still claims a use or usage type (never counted genuine). Each paper row carries the raw
``is_mentioned`` / ``is_used`` flags. The sample is the top-cited head
of the search result, not a random sample.

``--referent`` runs the referent check (``referent_*`` settings; the profile is the seed
description) on every genuine paper: ``precision`` is then the corrected precision and
``raw_precision`` the judge's (an upper bound); each paper row carries ``referent`` and
``judge_verdict``.

A failed model probe can stay WAITING on extraction, so keep ``--poll-timeout`` short
(it applies to the search and to each batch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from dataset_agent.adapters.extraction_client import ExtractionClient, ExtractionError
from dataset_agent.adapters.fulltext_evidence import Evidence, build_evidence
from dataset_agent.adapters.referent_check import check_referents
from dataset_agent.settings import Settings

#: Keys ``POST /ai-models/start`` accepts besides the probe fields the client adds.
SEED_KEYS = (
    "model_name", "model_family", "description", "hugging_face_id", "model_aliases",
    "flag_terms", "exclude_terms", "years_range", "dimensions_limit", "filter_us_affiliation",
)


def build_seed(raw: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Full ``/ai-models/start`` body (without the probe fields) from a hand-built seed."""
    unknown = sorted(set(raw) - set(SEED_KEYS))
    if unknown:
        raise ValueError(f"seed keys the server would refuse: {unknown}")
    name = str(raw.get("model_name") or "").strip()
    if not name:
        raise ValueError("seed needs a model_name")
    description = str(raw.get("description") or "").strip()
    if not description:
        raise ValueError("seed needs a description (the judge reads it)")
    years = raw.get("years_range") or {
        "start_year": int(settings.default_years_start),
        "end_year": int(settings.default_years_end),
    }
    return {
        "model_name": name,
        "model_family": raw.get("model_family") or name,
        "description": description,
        "hugging_face_id": raw.get("hugging_face_id") or None,
        "model_aliases": list(raw.get("model_aliases") or []),
        "flag_terms": list(raw.get("flag_terms") or []),
        "exclude_terms": list(raw.get("exclude_terms") or []),
        "years_range": dict(years),
        "dimensions_limit": raw.get("dimensions_limit"),
        "filter_us_affiliation": bool(raw.get("filter_us_affiliation", False)),
    }


def seed_from_result(path: Path, settings: Settings) -> tuple[dict[str, Any], str | None]:
    """The shipped seed of an ``optimize-model`` result file, verbatim (start defaults
    added for the keys it lacks), and the seed clause it was counted with."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    shipped = record.get("ai_model_seed")
    if not isinstance(shipped, dict):
        raise ValueError(f"{path} has no ai_model_seed (the seed was withheld)")
    seed = build_seed(shipped, settings)
    seed.update({k: shipped[k] for k in shipped})
    return seed, record.get("ai_model_seed_clause")


def anchors_from_result(path: Path) -> list[str]:
    """The admitted paper titles (anchors) of an ``optimize-model`` result file."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        str(d["term"]) for d in record.get("term_decisions") or []
        if d.get("kind") == "paper_title" and d.get("decision") == "anchor"
    ]


def write_seed_file(seed: dict[str, Any], out_dir: Path) -> tuple[Path, str]:
    """Write ``<out_dir>/<model>.json`` holding *seed* verbatim; return the path and its
    sha256."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", str(seed["model_name"]).casefold()).strip("-")
    path = out_dir / f"{slug}.json"
    path.write_text(json.dumps(seed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval of *k*/*n*; (None, None) when *n* is 0."""
    if n <= 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def _truncate(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rstrip() + "…"


def summarize(ev: Evidence, *, quote_chars: int = 200) -> dict[str, Any]:
    """Probe metrics and per-paper rows from model :class:`Evidence`."""
    n = ev.n_evaluated
    genuine = len(ev.genuine)
    used = sum(1 for v in ev.verdicts if v.used)
    cited = ev.n_cited_only
    lo, hi = wilson_interval(genuine, n)
    raw = ev.raw_precision
    return {
        "n_sampled": ev.n_sampled,
        "n_delivered": ev.n_delivered,
        "n_evaluated": n,
        "n_genuine": genuine,
        "n_used": used,
        "precision": _r(genuine / n) if n else None,
        "raw_precision": _r(raw),
        "n_referent_rejected": ev.n_referent_rejected,
        "n_referent_unclear": ev.n_referent_unclear,
        "used_rate": _r(used / n) if n else None,
        "n_cited_only": cited,
        "cited_only_share": _r(cited / genuine) if genuine and cited is not None else None,
        "n_cited_only_unknown": ev.n_cited_only_unknown,
        "n_not_run": ev.n_not_run,
        "not_run_share": (
            _r(ev.n_not_run / genuine) if genuine and ev.n_not_run is not None else None
        ),
        "n_error_rows": ev.n_error_rows,
        "error_row_share": _r(ev.n_error_rows / ev.n_delivered) if ev.n_delivered else None,
        "wilson_95": [_r(lo), _r(hi)],
        "n_overridden": sum(1 for v in ev.verdicts if v.overridden_from is not None),
        "n_incoherent": ev.n_incoherent,
        "n_no_verdict": ev.n_no_verdict,
        "papers": [
            {
                "publication_id": v.publication_id,
                "title": v.title,
                "verdict": v.verdict,
                "usage_type": v.usage_type,
                "match_type": v.match_type,
                "cited_only": v.cited_only,
                "not_run": v.not_run,
                "mention_location": v.mention_location,
                "matched_name": v.matched_name,
                "is_mentioned": v.is_mentioned,
                "is_used": v.is_used,
                "referent": v.referent,
                "judge_verdict": v.overridden_from or v.verdict,
                "text_quote": _truncate(v.quote, quote_chars),
            }
            for v in ev.verdicts
        ],
    }


def _cleanup_report(client: Any, task_id: str | None) -> dict[str, Any] | None:
    log = [e for e in getattr(client, "cleanup_log", []) if task_id is None or e.get("task_id") == task_id]
    if not log:
        return None
    last = log[-1]
    return {"ok": last.get("ok"), "items": last.get("items"), "error": last.get("error")}


def run_probe(
    client: Any,
    seed: dict[str, Any],
    *,
    target: int,
    batch_size: int,
    max_batches: int,
    pull_cap: int,
    quote_chars: int = 200,
    referent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the model probe (cleaned up by the client) and build the report. With
    *referent* (``agent``, ``model_name``, ``profile``, ``anchors``, ``votes``) every
    genuine paper gets the referent check before the metrics."""
    report: dict[str, Any] = {
        "seed": seed,
        "sample": "top-cited head of the search result",
        "probe_task_id": None,
        "error": None,
    }
    try:
        topup = client.run_probe_topup(
            seed, target=target, batch_size=batch_size, max_batches=max_batches,
            pull_cap=pull_cap, entity="ai_model",
        )
    except ExtractionError as exc:
        report["error"] = str(exc)
        log = getattr(client, "cleanup_log", [])
        report["probe_task_id"] = log[-1]["task_id"] if log else None
        report["cleanup"] = _cleanup_report(client, report["probe_task_id"])
        return report
    ev = build_evidence(topup.results, entity="ai_model")
    if referent is not None:
        res = check_referents(
            ev, agent=referent["agent"], model_name=referent["model_name"],
            profile=referent["profile"], anchor_titles=list(referent.get("anchors") or []),
            votes=int(referent.get("votes") or 1),
        )
        report["referent"] = {
            "profile": referent["profile"], "anchors": list(referent.get("anchors") or []),
            "votes": int(referent.get("votes") or 1), "checked": res.n_checked,
            "confirmed_by_anchor": res.n_confirmed_by_anchor, "confirmed": res.n_confirmed,
            "rejected": res.n_rejected, "unclear": res.n_unclear, "calls": res.calls,
            "failed_calls": res.failed_calls,
        }
    report["probe_task_id"] = topup.task_id or None
    report.update(
        target=topup.target,
        batches=topup.batches,
        stop_reason=topup.stop_reason,
        budget_exhausted=topup.budget_exhausted,
    )
    report.update(summarize(ev, quote_chars=quote_chars))
    report["cleanup"] = _cleanup_report(client, topup.task_id or None)
    return report


def parse_args(argv: list[str] | None, settings: Settings) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seed", nargs="?",
                    help="hand-built seed JSON file ({model_name, description, ...})")
    ap.add_argument("--from-result", type=Path,
                    help="probe the shipped ai_model_seed of an optimize-model result file")
    ap.add_argument("--target", type=int, default=settings.fulltext_target_delivered,
                    help="papers with delivered full text to aim for")
    ap.add_argument("--max-batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=settings.fulltext_batch_size)
    ap.add_argument("--pull-cap", type=int, default=settings.probe_pull_cap)
    ap.add_argument("--poll-timeout", type=float, default=900.0,
                    help="seconds for the search and for each batch (a failed model probe "
                    "can stay WAITING)")
    ap.add_argument("--extraction", choices=("dev", "prod"), default=settings.probe_target,
                    help="which DATASET_AGENT_EXTRACTION_<TARGET>_* settings to use")
    ap.add_argument("--quote-chars", type=int, default=200)
    ap.add_argument("--referent", action="store_true",
                    help="run the referent check (referent_* settings) on genuine papers")
    ap.add_argument("--anchor", action="append", default=[],
                    help="anchor paper title for the referent check (default with "
                    "--from-result: the result's admitted paper titles)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    args = parse_args(argv, settings)
    clause = None
    if args.from_result:
        seed, clause = seed_from_result(args.from_result, settings)
    elif args.seed:
        seed = build_seed(json.loads(Path(args.seed).read_text(encoding="utf-8")), settings)
    else:
        raise SystemExit("give a seed file or --from-result")
    settings = settings.model_copy(update={"probe_poll_timeout_seconds": float(args.poll_timeout)})
    referent = None
    if args.referent:
        from dataset_agent.bootstrap import build_referent_agent

        anchors = list(args.anchor) or (
            anchors_from_result(args.from_result) if args.from_result else []
        )
        referent = {"agent": build_referent_agent(settings), "model_name": seed["model_name"],
                    "profile": seed["description"], "anchors": anchors,
                    "votes": settings.referent_votes}
    client = ExtractionClient.for_target(settings, args.extraction)
    try:
        report = run_probe(
            client, seed, target=args.target, batch_size=args.batch_size,
            max_batches=args.max_batches, pull_cap=args.pull_cap, quote_chars=args.quote_chars,
            referent=referent,
        )
    finally:
        client.close()
    if args.from_result:
        report["seed_source"] = str(args.from_result)
        report["clause_probed"] = clause
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["error"]:
        return 1
    cleanup = report.get("cleanup") or {}
    return 0 if cleanup.get("ok") else 3


if __name__ == "__main__":
    sys.exit(main())

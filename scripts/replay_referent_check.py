#!/usr/bin/env python3
"""Replay the AI-model referent check offline on stored probe rows against a label file.

No probe and no extraction stack: the rows come from a probe report written by
``scripts/probe_ai_model_seed.py`` (``papers``: publication_id, title, verdict,
text_quote). Each paper's stored quote stands in for its full-text contexts: a quote
holding an anchor title (punctuation folded) is a reference line (stage 0); every other
quote is the stage-1 passage. The judge is the configured LLM (``DATASET_AGENT_*``
settings, ``.env`` is read).

Label file (JSON object per set): ``rows`` (probe report path), ``model_name``,
``description``, ``anchors`` ([[title, year], ...]), ``other`` (0-based row indices
labelled as another referent), optional ``wrong`` (rows that do not refer to the model
for another reason: counted against precision, outside both scored groups) and
``criteria`` (``other_kept_max``, ``same_lost_max``, ``corrected_precision_max`` /
``_min``). Scored groups count only papers the judge accepted.

The whole replay runs ``--replays`` times; a set passes when every replay meets its
criteria, failed calls are at most ``--max-failed-share`` of calls and per-paper
decisions agree across replays at least ``--min-agreement``. One JSON report is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

from dataset_agent.adapters.fulltext_evidence import Evidence, PubVerdict, _fold
from dataset_agent.adapters.referent_check import check_referents, model_profile


def evidence_from_rows(rows: list[dict], anchor_titles: list[str], *, model_name: str) -> Evidence:
    """Model :class:`Evidence` from stored probe rows (the quote is the only context)."""
    keys = [k for k in (_fold(t) for t in anchor_titles) if k]
    ev = Evidence(entity="ai_model")
    for r in rows:
        quote = str(r.get("text_quote") or "")
        contexts = (
            [{"term": model_name, "role": "alias", "part": "references", "text": quote}]
            if quote and any(k in _fold(quote) for k in keys) else []
        )
        ev.verdicts.append(PubVerdict(
            publication_id=str(r.get("publication_id") or ""), doi=None,
            title=str(r.get("title") or ""), citation_count=0, verdict=str(r["verdict"]),
            quote=quote, contexts=contexts, usage_type=r.get("usage_type"),
            names_in_references_only=None,
        ))
    ev.n_delivered = ev.n_sampled = len(rows)
    return ev


def replay_once(rows: list[dict], labels: dict, *, agent: Any, model_name: str, profile: str,
                anchor_titles: list[str], votes: int = 3) -> dict[str, Any]:
    ev = evidence_from_rows(rows, anchor_titles, model_name=model_name)
    accepted = {i for i, v in enumerate(ev.verdicts) if v.genuine}
    res = check_referents(ev, agent=agent, model_name=model_name, profile=profile,
                          anchor_titles=anchor_titles, votes=votes)
    other = set(labels.get("other") or [])
    wrong = set(labels.get("wrong") or [])
    same = set(range(len(rows))) - other - wrong
    kept = {i for i, v in enumerate(ev.verdicts) if v.genuine}
    out = {
        "n": ev.n_evaluated,
        "other_accepted_by_judge": len(other & accepted),
        "other_kept": len(other & kept),
        "wrong_accepted_by_judge": len(wrong & accepted),
        "wrong_kept": len(wrong & kept),
        "same_accepted_by_judge": len(same & accepted),
        "same_lost": len((same & accepted) - kept),
        "raw_precision": ev.raw_precision,
        "corrected_precision": ev.corrected_precision,
        "label_precision": len(same) / len(rows) if rows else None,
        "confirmed_by_anchor": res.n_confirmed_by_anchor,
        "rejected": res.n_rejected,
        "unclear": res.n_unclear,
        "calls": res.calls,
        "failed_calls": res.failed_calls,
        "decisions": {v.publication_id: v.referent or "not_checked" for v in ev.verdicts},
        "votes": {pid: p["votes"] for pid, p in res.papers.items()},
    }
    c = labels.get("criteria") or {}
    checks = []
    if "other_kept_max" in c:
        checks.append(out["other_kept"] <= c["other_kept_max"])
    if "same_lost_max" in c:
        checks.append(out["same_lost"] <= c["same_lost_max"])
    if "corrected_precision_max" in c:
        checks.append((out["corrected_precision"] or 0) <= c["corrected_precision_max"])
    if "corrected_precision_min" in c:
        checks.append((out["corrected_precision"] or 0) >= c["corrected_precision_min"])
    out["passes"] = all(checks)
    return out


def agreement(replays: list[dict]) -> float | None:
    """Share of (paper, replay pair) decisions that agree."""
    same = total = 0
    for a, b in combinations(replays, 2):
        for pid, d in a["decisions"].items():
            total += 1
            same += d == b["decisions"].get(pid)
    return same / total if total else None


def summarize_replays(replays: list[dict], *, min_agreement: float, max_failed_share: float) -> dict:
    calls = sum(r["calls"] for r in replays)
    failed = sum(r["failed_calls"] for r in replays)
    agree = agreement(replays)
    failed_share = failed / calls if calls else 0.0
    return {
        "agreement": agree,
        "failed_share": failed_share,
        "every_replay_passes": all(r["passes"] for r in replays),
        "passes": all(r["passes"] for r in replays)
        and failed_share <= max_failed_share
        and (agree is None or agree >= min_agreement),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", help="label file (JSON)")
    ap.add_argument("--set", action="append", dest="sets", help="label set(s) to replay")
    ap.add_argument("--replays", type=int, default=3)
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--min-agreement", type=float, default=0.90)
    ap.add_argument("--max-failed-share", type=float, default=0.05)
    args = ap.parse_args(argv)

    from dataset_agent.bootstrap import _build_agent
    from dataset_agent.settings import Settings

    agent = _build_agent(Settings())
    all_labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    report: dict[str, Any] = {}
    for name in args.sets or list(all_labels):
        lab = all_labels[name]
        rows = json.loads(Path(lab["rows"]).read_text(encoding="utf-8"))["papers"]
        anchors = [(t, y) for t, y in lab.get("anchors") or []]
        profile = model_profile(lab["description"], anchors)
        runs = []
        for i in range(args.replays):
            runs.append(replay_once(rows, lab, agent=agent, model_name=lab["model_name"],
                                    profile=profile, anchor_titles=[t for t, _y in anchors],
                                    votes=args.votes))
            print(f"{name} replay {i + 1}: " + json.dumps(
                {k: v for k, v in runs[-1].items() if k not in ("decisions", "votes")}),
                file=sys.stderr, flush=True)
        report[name] = {"profile": profile, "replays": runs,
                        **summarize_replays(runs, min_agreement=args.min_agreement,
                                            max_failed_share=args.max_failed_share)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(r["passes"] for r in report.values()) else 2


if __name__ == "__main__":
    sys.exit(main())

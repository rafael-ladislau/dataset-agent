"""Offline replay of the scope check on hand-labelled papers (program versus data).

Passages are cut from stored plain-text full texts with the extraction window (200
characters either side, 3 per term, 12 per paper); sections after the BIBLIOGRAPHY marker
are the reference list. A paper labelled ``used`` enters as judged ``used``; every other
``data`` or ``program_only`` paper enters as judged ``mentioned``. ``not_mentioned`` labels
are left out (the scope check only sees papers the judge accepted). No judge quote is
passed: the labellers' quotes would leak the label. Entries without a text (guard sets)
use their quote as one body passage.

Usage:
  replay_scope_check.py --labels labels.json --map map.json [--set aou|midrc|gnss]
                        [--replays 3] [--out out.json]
map.json: {"dataset_name", "description", "program_names", "data_names", "data_terms"}.
The LLM comes from the DATASET_AGENT_* settings; thinking uses the referent settings.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dataset_agent.adapters.fulltext_evidence import Evidence, PubVerdict
from dataset_agent.adapters.scope_check import check_scope

WINDOW, PER_TERM, MAX_PER_PAPER = 200, 3, 12
_BIB = re.compile(r"^## BIBLIOGRAPHY", re.M)
_HEAD = re.compile(r"^#{1,2} (TITLE|ABSTRACT|[A-Z ]+?)(?::|$)", re.M)


def split_parts(text: str) -> list[tuple[str, str]]:
    """(part, text) for title / abstract / body sections and the reference list."""
    m = _BIB.search(text)
    main, refs = (text[: m.start()], text[m.end():]) if m else (text, "")
    out: list[tuple[str, str]] = []
    heads = list(_HEAD.finditer(main))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(main)
        kind = h.group(1)
        part = "title" if kind == "TITLE" else "abstract" if kind == "ABSTRACT" else "body"
        chunk = main[h.end(): end] if part != "title" else main[h.start(): end].split(":", 1)[-1]
        if chunk.strip():
            out.append((part, chunk))
    if refs.strip():
        out.append(("references", refs))
    return out


def _pattern(term: str) -> re.Pattern:
    words = [re.escape(w) for w in re.split(r"[\W_]+", term) if w]
    return re.compile(r"(?<![\w])" + r"[\W_]{1,5}".join(words) + r"(?![\w])", re.I)


def cut_contexts(text: str, terms: list[str]) -> list[dict]:
    """Snippets {term, role, part, text} around each term occurrence, extraction style."""
    parts = split_parts(text)
    out: list[dict] = []
    for term in terms:
        pat, kept = _pattern(term), 0
        for part, chunk in parts:
            last_end = -1
            for m in pat.finditer(chunk):
                if kept >= PER_TERM or len(out) >= MAX_PER_PAPER:
                    break
                if m.start() < last_end:
                    continue
                lo, hi = max(0, m.start() - WINDOW), min(len(chunk), m.end() + WINDOW)
                out.append({"term": term, "role": "alias", "part": part,
                            "text": " ".join(chunk[lo:hi].split())})
                kept, last_end = kept + 1, hi
    return out


def evidence_from_labels(entries: list[dict], terms: list[str]) -> Evidence:
    ev = Evidence(entity="dataset")
    for e in entries:
        if e["label"] not in ("data", "program_only"):
            continue
        path = e.get("path")
        if path and Path(path).exists():
            ctx = cut_contexts(Path(path).read_text(errors="replace"), terms)
        else:
            ctx = [{"term": terms[0] if terms else "", "role": "alias", "part": "body",
                    "text": e.get("quote") or ""}]
        ev.verdicts.append(PubVerdict(
            publication_id=e["id"], doi=e.get("doi"), title=e.get("title") or "",
            citation_count=0, verdict="used" if e.get("used") else "mentioned", quote="",
            contexts=ctx,
        ))
    ev.n_sampled = ev.n_delivered = len(ev.verdicts)
    return ev


def score(entries: list[dict], ev: Evidence) -> dict:
    """Per label group (firm, borderline, close): papers labelled program-only or no longer
    genuine (rejected for program_only labels, lost for data labels)."""
    lab = {e["id"]: e for e in entries}
    rows: dict[str, list[int]] = {}
    decisions = {}
    for v in ev.verdicts:
        e = lab[v.publication_id]
        decisions[v.publication_id] = v.scope or ("references_only" if not v.genuine else None)
        groups = [e["label"] + ("_borderline" if e.get("borderline") else "")]
        if e.get("close"):
            groups.append(e["label"] + "_close")
        for g in groups:
            row = rows.setdefault(g, [0, 0])
            row[0] += 1
            row[1] += (not v.genuine) or v.scope == "program_only"
    out: dict = {"decisions": decisions}
    for g in ("program_only", "data", "program_only_borderline", "data_borderline",
              "program_only_close", "data_close"):
        n, k = rows.get(g, [0, 0])
        word = "rejected" if g.startswith("program_only") else "lost"
        out[f"{g}_n"], out[f"{g}_{word}"] = n, k
    return out


def agreement(replays: list[dict]) -> float | None:
    """Share of papers with the same decision in every replay."""
    if not replays:
        return None
    ids = list(replays[0]["decisions"])
    same = sum(1 for i in ids if len({r["decisions"].get(i) for r in replays}) == 1)
    return same / len(ids) if ids else None


def run(entries, mp, agent, replays: int, votes: int = 1) -> dict:
    terms = list(dict.fromkeys([*mp["program_names"], *mp["data_names"], *mp.get("data_terms", [])]))
    outs = []
    for _ in range(replays):
        ev = evidence_from_labels(entries, terms)
        res = check_scope(ev, agent=agent, dataset_name=mp["dataset_name"],
                          description=mp.get("description", ""), program_names=mp["program_names"],
                          data_names=mp["data_names"], data_terms=mp.get("data_terms"),
                          votes=votes)
        s = score(entries, ev)
        s.update(calls=res.calls, failed_calls=res.failed_calls, by_name=res.n_data_by_name,
                 unclear=res.n_unclear, references_only=res.n_references_only, votes={k: v["votes"] for k, v in res.papers.items()})
        outs.append(s)
        print(json.dumps({k: v for k, v in s.items() if k not in ("decisions", "votes")}),
              file=sys.stderr)
    return {"replays": outs, "agreement": agreement(outs)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--set", default="aou")
    ap.add_argument("--replays", type=int, default=3)
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from dataset_agent.bootstrap import build_referent_agent
    from dataset_agent.settings import Settings

    data = json.loads(Path(a.labels).read_text())
    entries = (list(data["labels"].values()) if a.set == "aou"
               else next(g["labels"] for g in data["guard"] if g["set"] == a.set))
    mp = json.loads(Path(a.map).read_text())
    out = run(entries, mp, build_referent_agent(Settings()), a.replays, a.votes)
    out.update(set=a.set, map=mp, votes=a.votes)
    text = json.dumps(out, indent=1)
    Path(a.out).write_text(text + "\n") if a.out else print(text)


if __name__ == "__main__":
    main()

"""Offline recall replay: which labelled papers would a final FOR clause match?

The clause (Dimensions full-data syntax: quoted phrases, AND, OR, NOT, parentheses) is
phrase-matched against the stored plain texts of the labelled papers. A phrase matches
case-insensitively with any run of up to five non-word characters between its words and
an optional plural ``s`` or ``es`` on the last word (close to the Dimensions phrase match). ``A NOT
B`` excludes B from everything before it in the same group, as Lucene does.

The whole text is searched (Dimensions indexes the reference list too); ``--no-references``
searches only the text before the BIBLIOGRAPHY marker.

Groups: firm data, used, borderline data, firm program-only, borderline program-only.
Bars (``--bars``): firm data matched >= 0.85, used >= 0.90, firm program-only matched <= 0.15.

Usage:
  replay_query_recall.py --labels labels.json (--clause TEXT | --result result.json)
                         [--no-references] [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

BARS = {"data_firm": (">=", 0.85), "used": (">=", 0.90), "program_only_firm": ("<=", 0.15)}
_BIB = re.compile(r"^## BIBLIOGRAPHY", re.M)
_TOKEN = re.compile(r'\\?"((?:[^"\\]|\\(?!"))*)\\?"|\(|\)|\bAND\b|\bOR\b|\bNOT\b')


def tokenize(clause: str) -> list[tuple[str, str]]:
    """[(kind, value)] with kind in phrase, (, ), AND, OR, NOT."""
    out: list[tuple[str, str]] = []
    for m in _TOKEN.finditer(clause):
        if m.group(1) is not None:
            out.append(("phrase", " ".join(m.group(1).split())))
        else:
            out.append((m.group(0), m.group(0)))
    return out


@lru_cache(maxsize=4096)
def phrase_pattern(phrase: str) -> re.Pattern:
    words = [re.escape(w) for w in re.split(r"[\W_]+", phrase) if w]
    if not words:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<![\w])" + r"[\W_]{1,5}".join(words) + r"(?:e?s)?(?![\w])", re.I)


def phrase_in(phrase: str, text: str) -> bool:
    return bool(phrase_pattern(phrase).search(text))


class _Parser:
    """group := operand (op operand)*; AND binds tighter than OR; NOT excludes from the
    group so far."""

    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def parse(self):
        node = self._group()
        if self.i != len(self.t):
            raise ValueError(f"unexpected token {self.t[self.i]!r}")
        return node

    def _group(self):
        ors: list[list] = [[self._operand()]]
        nots: list = []
        while self.i < len(self.t) and self.t[self.i][0] in ("AND", "OR", "NOT"):
            op = self.t[self.i][0]
            self.i += 1
            node = self._operand()
            if op == "AND":
                ors[-1].append(node)
            elif op == "OR":
                ors.append([node])
            else:
                nots.append(node)
        return ("group", ors, nots)

    def _operand(self):
        if self.i >= len(self.t):
            raise ValueError("clause ends early")
        kind, val = self.t[self.i]
        self.i += 1
        if kind == "phrase":
            return ("phrase", val)
        if kind == "(":
            node = self._group()
            if self.i >= len(self.t) or self.t[self.i][0] != ")":
                raise ValueError("missing )")
            self.i += 1
            return node
        raise ValueError(f"unexpected token {val!r}")


def parse(clause: str):
    return _Parser(tokenize(clause)).parse()


def evaluate(node, text: str) -> bool:
    if node[0] == "phrase":
        return phrase_in(node[1], text)
    _, ors, nots = node
    hit = any(all(evaluate(n, text) for n in conj) for conj in ors)
    return hit and not any(evaluate(n, text) for n in nots)


def phrases(node) -> list[str]:
    if node[0] == "phrase":
        return [node[1]]
    _, ors, nots = node
    return [p for conj in ors for n in conj for p in phrases(n)] + [p for n in nots for p in phrases(n)]


def group_of(e: dict) -> list[str]:
    lab = e.get("label")
    if lab == "data":
        if e.get("borderline"):
            return ["data_borderline"]
        return ["data_firm"] + (["used"] if e.get("used") else [])
    if lab == "program_only":
        return ["program_only_borderline" if e.get("borderline") else "program_only_firm"]
    return []


def recall(entries: list[dict], clause: str, *, references: bool = True) -> dict:
    tree = parse(clause)
    terms = list(dict.fromkeys(phrases(tree)))
    rows: dict[str, dict] = {}
    papers: dict[str, dict] = {}
    for e in entries:
        groups = group_of(e)
        path = e.get("path")
        if not groups or not path or not Path(path).exists():
            continue
        text = Path(path).read_text(errors="replace")
        if not references:
            m = _BIB.search(text)
            text = text[: m.start()] if m else text
        hit = evaluate(tree, text)
        papers[e["id"]] = {"groups": groups, "matched": hit,
                           "terms": [t for t in terms if phrase_in(t, text)]}
        for g in groups:
            row = rows.setdefault(g, {"n": 0, "matched": 0})
            row["n"] += 1
            row["matched"] += hit
    for row in rows.values():
        row["share"] = row["matched"] / row["n"] if row["n"] else None
    bars = {}
    for g, (op, bar) in BARS.items():
        share = rows.get(g, {}).get("share")
        bars[g] = None if share is None else (share >= bar if op == ">=" else share <= bar)
    missed = sorted(i for i, p in papers.items() if not p["matched"]
                    and any(g in ("data_firm", "used") for g in p["groups"]))
    return {"groups": rows, "bars": bars, "passed": all(v is not False for v in bars.values()),
            "missed_data": missed, "papers": papers}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clause")
    src.add_argument("--result", help="optimize result.json; its for_clause is replayed")
    ap.add_argument("--no-references", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    data = json.loads(Path(a.labels).read_text())
    entries = list(data["labels"].values()) if isinstance(data.get("labels"), dict) else data
    clause = a.clause or json.loads(Path(a.result).read_text())["for_clause"]
    out = recall(entries, clause, references=not a.no_references)
    out["clause"] = clause
    summary = {g: f"{r['matched']}/{r['n']}" for g, r in out["groups"].items()}
    print(json.dumps({"groups": summary, "bars": out["bars"], "passed": out["passed"]}),
          file=sys.stderr)
    text = json.dumps(out, indent=1)
    Path(a.out).write_text(text + "\n") if a.out else print(text)


if __name__ == "__main__":
    main()

"""Scope check: does a genuine dataset paper refer to the DATA resource, or only to the program?

Runs only for datasets whose resource map found a program separate from its data. Owner
rule: a paper counts when its text outside the reference list refers to the data resource
(the data, the access platform, the repository, a tier, a release, an existing data type,
data availability; a bare listing among data resources counts too). Use is not required.
A paper that only discusses the program (goals, plans to collect data, recruitment, ethics,
policy, funding) is ``program_only`` and is not genuine for precision.

Only papers judged ``mentioned`` are checked; ``used`` papers are ``data`` without a call.
A checked paper with no name context outside the reference list falls under the
references-only rule: it is turned to ``not_mentioned`` without a call (``scope`` stays None).

* stage 0, no LLM call: a data resource name that carries the program's own phrase
  (e.g. "<program> Researcher Workbench") in a name context outside the reference list
  means ``data``;
* stage 1: one narrow question on up to three passages outside the reference list
  (passages holding a data term first) plus the judge's quote, asked ``votes`` times.
  A majority of ``program_only`` turns the paper to ``not_mentioned`` (``overridden_from``
  keeps the judge's verdict); a majority of ``data`` confirms it; anything else (failed
  calls, ``unclear``, no majority) keeps the paper as ``unclear``.

General rules only: the names come from the resource map, never from constants here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from dataset_agent.adapters.fulltext_evidence import Evidence, PubVerdict, _phrase_in

logger = logging.getLogger(__name__)

_TEXT_PARTS = frozenset({"title", "abstract", "body"})
SCOPES = ("data", "program_only", "unclear")

EMIT_SCOPE_CLASSIFICATION = {
    "name": "emit_scope_classification",
    "description": "Say whether the paper refers to the data resource or only to the program.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scope": {"type": "string", "enum": list(SCOPES)},
            "reason": {"type": "string"},
        },
        "required": ["scope"],
    },
}


def _contains_phrase(name: str, phrase: str) -> bool:
    return bool(phrase.strip()) and _phrase_in(phrase, name, plural_ok=False)


def program_owned_data_names(program_names: list[str], data_names: list[str]) -> list[str]:
    """Data resource names that carry one of the program's own phrases."""
    return [d for d in data_names if any(_contains_phrase(d, p) for p in program_names)]


def _text_contexts(v: PubVerdict) -> list[dict]:
    return [c for c in v.contexts
            if c.get("part") in _TEXT_PARTS and str(c.get("text") or "").strip()]


def stage0_data(v: PubVerdict, owned_data_names: list[str]) -> bool:
    """A program-owned data resource name in a passage outside the reference list."""
    for c in _text_contexts(v):
        text = str(c["text"])
        if any(_phrase_in(n, text, plural_ok=False) for n in owned_data_names):
            return True
    return False


def scope_passages(v: PubVerdict, data_terms: list[str], *, limit: int = 3) -> list[str]:
    """Up to *limit* distinct passages outside the reference list, data-term passages first."""
    texts: list[str] = []
    for c in _text_contexts(v):
        t = " ".join(str(c["text"]).split())
        if t not in texts:
            texts.append(t)
    with_data = [t for t in texts if any(_phrase_in(d, t) for d in data_terms)]
    rest = [t for t in texts if t not in with_data]
    return (with_data + rest)[:limit]


def scope_prompt(*, dataset_name: str, description: str, program_names: list[str],
                 data_names: list[str], title: str, passages: list[str], quote: str) -> str:
    lines = "\n".join(f"{i}. {p!r}" for i, p in enumerate(passages, 1)) or "(none)"
    return f"""Dataset: {dataset_name!r}. {" ".join((description or "").split())}
Program names: {", ".join(program_names) or "(none)"}.
Data resource names: {", ".join(data_names) or "(none)"}.
Publication title: {title!r}
Passages from the paper (outside the reference list):
{lines}
Validator quote: {quote!r}

The paper counts as DATA when its text refers to the data resource itself: its data, the
access platform, the repository, a tier, a release or version, an existing data type, data
availability; or when it names the resource in a list of biobanks, cohorts, datasets,
repositories or resources; or names it as a data source for possible future analyses.
Use is not required.
It is PROGRAM_ONLY when it discusses only the program: its goals, history or renaming,
recruitment targets, the program's own plans to collect data, ethics, consent, policy or
funding, a definition quoted from it, or when it names the program as an example in a
list of initiatives, programs, efforts or projects.
Noun test for lists: biobanks, cohorts, datasets, repositories or resources mean DATA;
initiatives, programs, efforts or projects mean PROGRAM_ONLY.
Answer unclear only when the passages do not let you decide.

Use emit_scope_classification exactly once; scope is data, program_only or unclear."""


def classify_scope(agent, prompt: str) -> str | None:
    """One scope answer, or None when the call failed or returned no valid scope."""
    try:
        raw = agent.get_structured(prompt, EMIT_SCOPE_CLASSIFICATION)
    except Exception as exc:
        logger.warning("scope check call failed: %s", exc)
        return None
    val = raw.get("scope") if isinstance(raw, dict) else None
    return val if val in SCOPES else None


@dataclass
class ScopeResult:
    n_checked: int = 0
    n_data_by_name: int = 0
    n_data: int = 0
    n_program_only: int = 0
    n_unclear: int = 0
    n_references_only: int = 0
    calls: int = 0
    failed_calls: int = 0
    #: publication id -> {"stage", "votes", "decision"}
    papers: dict[str, dict] = field(default_factory=dict)


def check_scope(
    ev: Evidence,
    *,
    agent,
    dataset_name: str,
    description: str,
    program_names: list[str],
    data_names: list[str],
    data_terms: list[str] | None = None,
    votes: int = 1,
) -> ScopeResult:
    """Run the scope check on *ev* in place; return the counts. AI models never enter."""
    res = ScopeResult()
    if ev.entity != "dataset" or not program_names or not data_names:
        return res
    owned = program_owned_data_names(program_names, data_names)
    terms = list(dict.fromkeys([*data_names, *(data_terms or [])]))
    for v in ev.verdicts:
        if v.verdict == "used":
            v.scope = "data"
            continue
        if v.verdict != "mentioned":
            continue
        res.n_checked += 1
        if not _text_contexts(v):
            v.overridden_from, v.verdict = v.verdict, "not_mentioned"
            res.n_references_only += 1
            res.papers[v.publication_id] = {"stage": 0, "votes": [], "decision": "references_only"}
            continue
        if stage0_data(v, owned):
            v.scope = "data"
            res.n_data_by_name += 1
            res.papers[v.publication_id] = {"stage": 0, "votes": [], "decision": "data"}
            continue
        prompt = scope_prompt(
            dataset_name=dataset_name, description=description, program_names=program_names,
            data_names=data_names, title=v.title, passages=scope_passages(v, terms),
            quote=v.quote,
        )
        cast: list[str | None] = []
        for _ in range(max(1, votes)):
            res.calls += 1
            answer = classify_scope(agent, prompt)
            if answer is None:
                res.failed_calls += 1
            cast.append(answer)
        need = max(1, votes) // 2 + 1
        if cast.count("program_only") >= need:
            v.overridden_from, v.verdict, v.scope = v.verdict, "not_mentioned", "program_only"
            res.n_program_only += 1
        elif cast.count("data") >= need:
            v.scope = "data"
            res.n_data += 1
        else:
            v.scope = "unclear"
            res.n_unclear += 1
        res.papers[v.publication_id] = {"stage": 1, "votes": cast, "decision": v.scope}
    return res

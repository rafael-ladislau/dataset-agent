"""Search-term research for an AI model (the model counterpart of dataset research).

Papers cite a model by its family name ("Prithvi-EO", "FourCastNet"), rarely by a
checkpoint ("Prithvi-EO-2.0-600M-TL"), and many family names are ordinary words: bare
"Aurora" matches the polar aurora, Aurora kinase and Amazon Aurora long before it matches
Microsoft's atmosphere model. This module gathers the candidates; the optimizer measures
each of them in Dimensions before any reaches a query.

Evidence comes from the Hugging Face Hub (owner, tags, model card), the arXiv papers the
card links, one web search, and one structured LLM call (no tool loop). Everything the
LLM returns is a candidate only.

Candidate kinds:

* names: the family name, version strings, qualified phrases ("Microsoft Aurora",
  "Aurora foundation model"), former names.
* paper titles: titles of the papers that introduce the model. They are anchors (papers
  quoting one certainly refer to the model).
* context terms: proper names that co-occur with the model in papers (peer models,
  benchmarks, training data, the developer lab). They gate an ambiguous name.
* other meanings: phrases specific to other senses of the name (exclude candidates).
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import httpx

from dataset_agent.adapters.name_grounding import content_tokens
from dataset_agent.adapters.query_heuristics import is_short_acronym
from dataset_agent.adapters.term_sanitizer import sanitize_term, sanitize_terms
from dataset_agent.adapters.text_processing import dedupe_strings_ci_preserve_order

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co"
ARXIV_API = "https://export.arxiv.org/api/query"
_HTTP_TIMEOUT = 20.0
_HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_LINK = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)
#: ``title = {...}`` of a BibTeX entry in the model card ("how to cite this model").
_BIBTEX_TITLE = re.compile(r"^\s*title\s*=\s*[{\"]+(.+?)[}\"]+\s*,?\s*$", re.I | re.M)

MODEL_TERMS_TOOL: dict = {
    "name": "emit_model_search_terms",
    "description": "Names, paper titles and context terms for finding papers about an AI model.",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "One or two sentences: what the model is, who built it, its domain.",
            },
            "developer": {
                "type": "string",
                "description": "Organization that develops the model, as written in papers.",
            },
            "names": {
                "type": "array",
                "description": "Names of the model as authors write them in papers.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["family", "version", "checkpoint", "qualified", "former"],
                        },
                    },
                    "required": ["name", "kind"],
                },
            },
            "paper_titles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact titles of the papers that introduce the model.",
            },
            "context_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Proper names that appear in papers about this model: peer models it is "
                    "compared with, benchmarks, training datasets, the developer lab."
                ),
            },
            "other_meanings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Phrases specific to OTHER meanings of the bare name.",
            },
        },
        "required": ["description", "names", "paper_titles", "context_terms", "other_meanings"],
    },
}


@dataclass
class HubEvidence:
    """What the Hugging Face Hub and arXiv say about the model (all best effort)."""

    hugging_face_id: str | None = None
    found: bool = False
    author: str = ""
    developer: str = ""
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str = ""
    created_year: int | None = None
    card_excerpt: str = ""
    arxiv_ids: list[str] = field(default_factory=list)
    paper_titles: list[str] = field(default_factory=list)
    paper_years: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ModelCandidates:
    """Unmeasured candidates, each list in priority order."""

    description: str = ""
    developer: str = ""
    names: list[tuple[str, str]] = field(default_factory=list)  # (name, kind)
    paper_titles: list[str] = field(default_factory=list)
    #: Case-folded titles that come from the model card or its arXiv links.
    hub_title_keys: set[str] = field(default_factory=set)
    context_terms: list[str] = field(default_factory=list)
    other_meanings: list[str] = field(default_factory=list)
    #: Candidates dropped before any Dimensions call.
    rejected: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, term, reason)


def _get_json(client: httpx.Client, url: str) -> Any:
    r = client.get(url)
    r.raise_for_status()
    return r.json()


def arxiv_titles(arxiv_ids: list[str], client: httpx.Client | None = None) -> list[tuple[str, int | None]]:
    """``(title, year)`` for each arXiv id (one request); unknown ids are skipped."""
    ids = [i for i in arxiv_ids if _ARXIV_ID.match(i)]
    if not ids:
        return []
    own = client is None
    client = client or httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True)
    try:
        r = client.get(ARXIV_API, params={"id_list": ",".join(ids), "max_results": len(ids)})
        r.raise_for_status()
        root = ET.fromstring(r.text)
    finally:
        if own:
            client.close()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: list[tuple[str, int | None]] = []
    for entry in root.findall("a:entry", ns):
        title = " ".join((entry.findtext("a:title", "", ns) or "").split())
        published = entry.findtext("a:published", "", ns) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        if title and title.casefold() != "error":
            out.append((title, year))
    return out


def fetch_hub_evidence(hugging_face_id: str | None, client: httpx.Client | None = None) -> HubEvidence:
    """Hub metadata, model card excerpt and linked arXiv titles; never raises."""
    ev = HubEvidence(hugging_face_id=hugging_face_id)
    hf_id = (hugging_face_id or "").strip()
    if not hf_id:
        return ev
    if not _HF_ID.match(hf_id):
        ev.errors.append(f"not a Hugging Face model id: {hf_id!r}")
        return ev
    if ".." in hf_id:
        ev.errors.append(f"not a Hugging Face model id: {hf_id!r}")
        return ev
    own = client is None
    client = client or httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True)
    try:
        try:
            info = _get_json(client, f"{HF_API}/api/models/{hf_id}")
            if not isinstance(info, dict):
                raise ValueError("unexpected Hub response")
        except Exception as exc:
            ev.errors.append(f"Hub lookup failed: {exc}")
            return ev
        ev.found = True
        ev.author = str(info.get("author") or hf_id.split("/")[0])
        ev.tags = [t for t in (info.get("tags") or []) if isinstance(t, str)]
        ev.pipeline_tag = str(info.get("pipeline_tag") or "")
        created = str(info.get("createdAt") or "")
        ev.created_year = int(created[:4]) if created[:4].isdigit() else None
        ev.arxiv_ids = [t.split(":", 1)[1] for t in ev.tags if t.startswith("arxiv:")]
        try:
            org = _get_json(client, f"{HF_API}/api/organizations/{ev.author}/overview")
            ev.developer = str(org.get("fullname") or "").strip()
        except Exception:  # individual accounts have no organization overview
            ev.developer = ""
        try:
            r = client.get(f"{HF_API}/{hf_id}/raw/main/README.md")
            if r.status_code == 200:
                # Drop the YAML front matter; the prose is what describes the model.
                card = re.sub(r"\A---\n.*?\n---\n", "", r.text, flags=re.S)
                ev.card_excerpt = card[:3000]
                ev.arxiv_ids = dedupe_strings_ci_preserve_order(
                    ev.arxiv_ids + _ARXIV_LINK.findall(card)
                )
                for m in _BIBTEX_TITLE.finditer(card):
                    title = " ".join(re.sub(r"[{}]", "", m.group(1)).split())
                    if title:
                        ev.paper_titles.append(title)
        except Exception as exc:
            ev.errors.append(f"model card fetch failed: {exc}")
        if ev.arxiv_ids:
            try:
                for title, year in arxiv_titles(ev.arxiv_ids[:5], client):
                    ev.paper_titles.append(title)
                    if year:
                        ev.paper_years.append(year)
            except Exception as exc:
                ev.errors.append(f"arXiv lookup failed: {exc}")
    finally:
        if own:
            client.close()
    return ev


def model_terms_prompt(
    model_name: str,
    family: str,
    hub: HubEvidence,
    given_aliases: list[str],
    web_results: str,
) -> str:
    hub_lines = "(no Hugging Face id given)"
    if hub.hugging_face_id:
        hub_lines = (
            f"id: {hub.hugging_face_id}\n"
            f"owner: {hub.developer or hub.author or '(unknown)'}\n"
            f"task: {hub.pipeline_tag or '(none)'}\n"
            f"tags: {', '.join(hub.tags[:25]) or '(none)'}\n"
            f"linked papers: {'; '.join(hub.paper_titles) or '(none)'}"
        )
    return f"""We search the scientific literature for papers that mention or use the AI model
{model_name!r} (model family {family!r}). Papers cite a model by its family name, sometimes
with a version, and almost never by a checkpoint file name.

=== HUGGING FACE ===
{hub_lines}

=== MODEL CARD (excerpt) ===
{hub.card_excerpt or "(none)"}

=== NAMES ALREADY KNOWN ===
{", ".join(given_aliases) or "(none)"}

=== WEB RESULTS ===
{(web_results or "(none)")[:5000]}

=== YOUR TASK ===
Call emit_model_search_terms exactly once:
- description: what the model is, who built it and for which domain.
- developer: the organization that develops it.
- names: how authors write this model in papers. kind "family" for the family name,
  "version" for a family name with a version ("Prithvi-EO-2.0"), "checkpoint" for a
  specific release, "qualified" for the name with its developer or type
  ("Microsoft Aurora", "Aurora foundation model"), "former" for an earlier name.
  If the bare name is also an ordinary word, a place or a product in another field,
  give several qualified forms.
- paper_titles: exact titles of the papers that introduce the model (at most 3).
- context_terms: at most 10 COINED NAMES that papers about this model also contain.
  Best are the peer models it is benchmarked against and its benchmarks or training
  datasets (for a weather model: "GraphCast", "Pangu-Weather", "WeatherBench", "ERA5").
  Never a lower-case topic word ("weather", "deep learning"), a journal, an author, or
  a company so large that it appears everywhere.
- other_meanings: at most 8 short phrases that occur only with OTHER meanings of the bare
  name (for "Aurora": "aurora borealis", "Aurora kinase"); empty if the name is unique.
Only give what the sources above or well-established knowledge support; never invent."""


def _is_checkpoint_shaped(name: str) -> bool:
    """A file or size-tagged release name ("aurora-0.25-finetuned.ckpt", "TerraMind-1.0-tiny")
    still goes to the count; this only orders candidates, it never rejects one."""
    return bool(re.search(r"\.(ckpt|pt|pth|bin|safetensors|onnx)$", name, re.I))


def is_plain_word(name: str) -> bool:
    """One ordinary-looking word: letters only, in one case pattern ("Aurora", "ALBERT").

    Dimensions matches case-insensitively, so "ALBERT" also finds every Albert. "Aurora",
    "Prithvi", "Whisper" and "Llama" are plain; "GraphCast", "FourCastNet", "BioCLIP" and
    "Pangu-Weather" are coined forms. A plain word is assumed to have other meanings
    until the counts show otherwise."""
    s = (name or "").strip()
    if not s or not s.isalpha():
        return False
    return s.isupper() or not any(ch.isupper() for ch in s[1:])


def _coined_token(token: str) -> bool:
    core = token.strip("()[],.;:")
    if len(core) < 5:
        return False
    parts = [p for p in core.split("-") if p]
    worded = [p for p in parts if any(ch.isalpha() for ch in p)]
    if len(worded) >= 2 and all(p[0].isupper() or p[0].isdigit() for p in worded):
        return True  # "Pangu-Weather", "Prithvi-EO-2.0" (not "Aurora-0.25", "AI-based")
    if any(ch.isalpha() for ch in core) and any(ch.isdigit() for ch in core) and "-" not in core:
        return core[0].isalpha() and not core.replace(".", "").isdigit()  # "FourCastNet3"
    letters = [ch for ch in core if ch.isalpha()]
    inner_upper = any(ch.isupper() for ch in core[1:])
    return inner_upper and any(ch.islower() for ch in letters)  # "GraphCast", "BioCLIP"


def is_distinctive(name: str) -> bool:
    """The name has a coined token (camel case, a hyphenated compound, letters joined
    to digits), so the phrase is unlikely to mean anything but the model."""
    return any(_coined_token(t) for t in (name or "").split())


def classify_model_name(
    name: str,
    *,
    count: int,
    confirmation: float | None,
    has_titles: bool = False,
    developer: str,
    safe_share: float,
    context_share: float,
    plain_word_max_hits: int,
) -> tuple[str, str]:
    """``(verdict, reason)`` for one counted name; verdict is ``safe``, ``gate`` or
    ``unconfirmed``.

    *confirmation* is the share of the name's hits that also quote one of the model's
    paper titles (``None`` when no title is known, or when *has_titles* is set and the
    share could not be measured, which never counts in the name's favor). ``gate`` means the name may only be
    searched together with a context flag; ``unconfirmed`` means nothing shows the phrase
    refers to this model."""
    pct = f"{confirmation:.0%}" if confirmation is not None else "n/a"
    if is_short_acronym(name):
        return "gate", "short acronym: never searched on its own"
    single_word = " " not in name.strip() and name.strip().isalpha()
    if single_word and count > plain_word_max_hits:
        # Dimensions ignores case: "LLaMA" finds every llama and "GloVe" every glove.
        return "gate", f"single word with {count} hits: other meanings dominate"
    if has_titles and confirmation is None:
        return "gate" if single_word else "unconfirmed", "citing share not measured"
    if " " not in name.strip() and is_plain_word(name):
        if confirmation is None:
            return "gate", "plain word and no paper title to confirm it"
        if confirmation >= safe_share:
            return "safe", f"{pct} of its hits cite the model's papers"
        return "gate", f"plain word; only {pct} of its hits cite the model's papers"
    if is_distinctive(name):
        if confirmation is None:
            return "safe", "coined name"
        if confirmation >= max(context_share, safe_share / 2):
            return "safe", f"coined name; {pct} of its hits cite the model's papers"
        return (
            "gate" if single_word else "unconfirmed",
            f"only {pct} of its hits cite the model's papers",
        )
    by_developer = bool(developer) and contains_word(name, developer)
    if confirmation is not None:
        # "<developer> <name>" is how authors disambiguate a common name themselves, so
        # half the usual share of citing papers is enough.
        bar = safe_share / 2 if by_developer else safe_share
        if confirmation >= bar:
            return "safe", f"{pct} of its hits cite the model's papers"
        return "unconfirmed", f"only {pct} of its hits cite the model's papers"
    if by_developer:
        return "safe", "name qualified by its developer"
    if name.casefold().endswith(" foundation model"):
        return "safe", "name qualified by its type"
    return "unconfirmed", "qualified phrase with no paper title to confirm it"


def is_proper_name(term: str) -> bool:
    """Every word starts with a capital or a digit, apart from joiners such as "of"."""
    from dataset_agent.adapters.fulltext_refiner import _generic_word

    return not _generic_word(term)


def _acronym_token(token: str) -> bool:
    core = token.strip("()[],.;:")
    letters = [ch for ch in core if ch.isalpha()]
    return (
        2 <= len(core) <= 6
        and len(letters) >= 2
        and all(ch.isupper() for ch in letters)
        and all(ch.isalnum() for ch in core)
    )


def is_context_name(term: str) -> bool:
    """A flag must carry a coined name or an acronym ("GraphCast", "Pangu-Weather",
    "ERA5", "ECMWF", "Google DeepMind").

    Dimensions ignores case, so capital letters alone prove nothing: "Weather
    Forecasting" and "Earth System" search as the ordinary words they are."""
    tokens = [t for t in (term or "").split() if t.strip("()[],.;:") not in _COMMON_ACRONYMS]
    return any(_coined_token(t) or _acronym_token(t) for t in tokens)


def contains_word(phrase: str, word: str) -> bool:
    """*word* occurs in *phrase* as a whole word sequence (case-insensitive)."""
    return bool(
        re.search(rf"(?<![\w]){re.escape(word.casefold())}(?![\w])", (phrase or "").casefold())
    )


_CTX_TOKEN = re.compile(r"(?<![\w-])([A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)(?![\w-])")
#: Acronyms found in any machine-learning abstract; they never single out one model.
_COMMON_ACRONYMS = frozenset(
    """AI ML DL NLP LLM LLMs GPU GPUs CPU CNN CNNs RNN GNN LSTM MLP GAN GANs API USA UK EU
    RMSE MSE MAE ACC SOTA IEEE DOI PDF CO2 NWP ANN ANNs SVM PCA IoT GIS UAV RGB SAR""".split()
)


def _context_shaped(token: str) -> bool:
    """A coined or acronym token worth testing as a context term ("GraphCast", "ERA5",
    "ECMWF", "Pangu-Weather"); ordinary capitalized words are not."""
    if len(token) < 3 or token in _COMMON_ACRONYMS:
        return False
    if any(p and p[0].islower() for p in token.split("-")):
        return False  # "AI-based", "Data-driven"
    if _coined_token(token):
        return True
    letters = [ch for ch in token if ch.isalpha()]
    if len(letters) < 2 or not all(ch.isupper() for ch in letters):
        return False
    return True  # all-caps acronym, with or without digits ("ECMWF", "ERA5", "CMIP6")


def mine_context_terms(
    texts: list[str], *, model_names: list[str], min_docs: int, limit: int
) -> list[tuple[str, int]]:
    """Coined names and acronyms frequent in papers that certainly refer to the model.

    Returns ``(term, number of texts containing it)``, most frequent first. These are
    candidates only: the optimizer still measures how precisely each one gates the
    ambiguous name. Terms containing a model name are skipped."""
    df: Counter[str] = Counter()
    spelling: dict[str, Counter[str]] = {}
    # Words seen in ordinary case anywhere in the sample: an upper-case title turns
    # "FORECASTING" into something acronym-shaped, but it is still that word.
    ordinary = {
        w.casefold()
        for text in texts
        for w in re.findall(r"[A-Za-z][A-Za-z-]*", text or "")
        if not w.isupper() and not any(ch.isupper() for ch in w[1:])
    }
    for text in texts:
        seen: set[str] = set()
        for token in _CTX_TOKEN.findall(text or ""):
            if _context_shaped(token) and token.casefold() not in seen:
                seen.add(token.casefold())
                spelling.setdefault(token.casefold(), Counter())[token] += 1
        df.update(seen)
    out: list[tuple[str, int]] = []
    for key, n in sorted(df.items(), key=lambda kv: (-kv[1], kv[0])):
        if n < min_docs or len(out) >= limit:
            break
        term = spelling[key].most_common(1)[0][0]
        # "MEDIUM-RANGE" and "Deep-Learning" are ordinary words when every part is one.
        parts = [p for p in key.split("-") if p]
        compound = len(parts) > 1 and all(p in ordinary for p in parts)
        shouted = term.isupper() and any(p in ordinary for p in parts if len(p) > 3)
        if key in ordinary or compound or shouted:
            continue
        if any(contains_word(term, m) or contains_word(m, term) for m in model_names):
            continue
        out.append((term, n))
    return out


#: Words that describe any model; a "name" made only of them is not a name.
_MODEL_GENERIC = frozenset(
    """model foundation network neural pretrained checkpoint weight ai base large small
    tiny version deep learning machine language vision""".split()
)


def _only_generic(name: str) -> bool:
    return not (content_tokens(name) - _MODEL_GENERIC)


def assemble_candidates(
    *,
    model_name: str,
    family: str | None,
    given_aliases: list[str],
    hub: HubEvidence,
    llm: dict | None,
    max_names: int,
    max_context: int,
    max_excludes: int,
) -> ModelCandidates:
    """Merge deterministic and LLM candidates and screen them by shape only.

    Shape screening removes what can never be a search term (junk, generic words, a
    lower-case "name"). Whether a surviving term is safe, gated or useless is decided
    later from its Dimensions counts."""
    llm = llm if isinstance(llm, dict) else {}
    # A runaway list from the LLM must not turn into thousands of decisions.
    llm = {k: (v[:40] if isinstance(v, list) else v) for k, v in llm.items()}
    given_aliases = list(given_aliases)[:40]
    out = ModelCandidates()
    out.description = str(llm.get("description") or "").strip()
    developer = hub.developer or str(llm.get("developer") or "").strip()
    out.developer = sanitize_term(developer) or ""
    fam = (family or "").strip() or model_name

    names: list[tuple[str, str]] = [(model_name, "family")]
    if fam.casefold() != model_name.casefold():
        names.append((fam, "family"))
    names += [(a, "given") for a in given_aliases]
    for base in dedupe_strings_ci_preserve_order([model_name, fam]):
        if out.developer and not contains_word(base, out.developer):
            names.append((f"{out.developer} {base}", "qualified"))
        names.append((f"{base} foundation model", "qualified"))
        names.append((f"{base} model", "qualified"))
    raw_names = llm.get("names")
    for item in raw_names if isinstance(raw_names, list) else []:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            kind = item.get("kind") if item.get("kind") in (
                "family", "version", "checkpoint", "qualified", "former"
            ) else "version"
            names.append((item["name"], str(kind)))
    if hub.hugging_face_id and hub.found:
        names.append((hub.hugging_face_id.split("/", 1)[1], "checkpoint"))

    seen: set[str] = set()
    for raw, kind in names:
        name = sanitize_term(raw)
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        if _only_generic(name):
            out.rejected.append(("alias", name, "generic words only"))
        elif not any(ch.isupper() or ch.isdigit() for ch in name):
            out.rejected.append(("alias", name, "not a proper name (all lower case)"))
        elif len(name) > 120:
            out.rejected.append(("alias", name, "too long for a phrase search"))
        elif _is_checkpoint_shaped(name):
            out.rejected.append(("alias", name, "checkpoint file name"))
        elif len(out.names) >= max_names:
            out.rejected.append(("alias", name, "over the candidate limit"))
        else:
            out.names.append((name, kind))

    titles = list(hub.paper_titles)
    out.hub_title_keys = {t.casefold() for t in sanitize_terms(hub.paper_titles, "paper title")}
    raw_titles = llm.get("paper_titles")
    titles += [t for t in (raw_titles if isinstance(raw_titles, list) else []) if isinstance(t, str)]
    for title in sanitize_terms(dedupe_strings_ci_preserve_order(titles), "paper title"):
        if title.casefold() in seen:
            continue
        if len(title.split()) < 4:
            out.rejected.append(("paper_title", title, "fewer than 4 words: not distinctive"))
        elif len(out.paper_titles) < 4:
            seen.add(title.casefold())
            out.paper_titles.append(title)

    raw_ctx = llm.get("context_terms")
    ctx = [t for t in (raw_ctx if isinstance(raw_ctx, list) else []) if isinstance(t, str)]
    if out.developer:
        ctx.append(out.developer)
    name_keys = {n.casefold() for n, _k in out.names}
    for term in sanitize_terms(dedupe_strings_ci_preserve_order(ctx), "context term"):
        key = term.casefold()
        if key in name_keys or key in {c.casefold() for c in out.context_terms}:
            continue
        if not is_proper_name(term):
            out.rejected.append(("flag", term, "not a proper name (lower-case word)"))
        elif not content_tokens(term):
            out.rejected.append(("flag", term, "generic words only"))
        elif any(contains_word(term, n) for n, _k in out.names):
            # "Aurora 0.25" as a flag for "Aurora" would make the gate satisfy itself.
            out.rejected.append(("flag", term, "contains the model's own name"))
        elif any(contains_word(n, term) for n, _k in out.names):
            # "Microsoft" next to "Microsoft Aurora", "Diffusion" next to "Stable
            # Diffusion": the name's own papers would vouch for the flag.
            out.rejected.append(("flag", term, "is part of one of the model's names"))
        elif not is_context_name(term):
            out.rejected.append(
                ("flag", term, "ordinary words only: no coined name or acronym")
            )
        elif len(out.context_terms) >= max_context:
            out.rejected.append(("flag", term, "over the candidate limit"))
        else:
            out.context_terms.append(term)

    raw_other = llm.get("other_meanings")
    other = [t for t in (raw_other if isinstance(raw_other, list) else []) if isinstance(t, str)]
    for term in sanitize_terms(dedupe_strings_ci_preserve_order(other), "exclude term"):
        key = term.casefold()
        if key in name_keys:
            out.rejected.append(("exclude", term, "equals a model name"))
        elif is_short_acronym(term):
            out.rejected.append(("exclude", term, "too short to exclude safely"))
        elif len(out.other_meanings) >= max_excludes:
            out.rejected.append(("exclude", term, "over the candidate limit"))
        else:
            out.other_meanings.append(term)
    return out

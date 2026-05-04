"""Text cleaning and alias filtering."""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Single-token aliases that are almost never a dataset identity on their own.
_ALIAS_GENERIC_BLOCKLIST: frozenset[str] = frozenset(
    {
        "model",
        "base",
        "small",
        "large",
        "data",
        "survey",
        "dataset",
        "study",
        "program",
        "records",
        "results",
        "sample",
        "panel",
        "cohort",
        "database",
        "registry",
        "index",
    }
)

_DESCRIPTION_PREAMBLE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^based on (?:the )?(?:web |internet )?search results?[,:]?\s*",
        r"^from (?:the )?(?:web |internet )?search results?[,:]?\s*",
        r"^here(?:['\u2019]s| is) (?:a |an )?(?:brief |concise |short )?description(?: of [^:]{0,160})?:\s*",
        r"^the following (?:is (?:a )?)?(?:brief |concise )?description[^:]{0,80}:\s*",
        r"^according to (?:my |our |the )?(?:web |internet )?search[^.]{0,120}\.\s*",
        r"^after (?:using |conducting |performing )?(?:a )?(?:web |internet )?search[^.]{0,120}\.\s*",
        r"^i (?:have |'ve )?(?:used|conducted|performed) (?:a )?(?:web |internet )?search[^.]{0,120}\.\s*",
        r"^i (?:used|will use) (?:the )?(?:`?web_search`?|web search) (?:tool |function )?[^.]{0,80}\.\s*",
        r"^using (?:the )?(?:`?web_search`?|web search)(?: tool)?,?\s*[^.]{0,120}\.\s*",
        r"^(?:below|above) is (?:a )?(?:brief |concise )?description[^:]{0,80}:\s*",
    )
)


def _strip_description_search_preamble(text: str) -> str:
    """Strip meta preambles about how information was gathered (not part of the dataset description)."""
    t = text.strip()
    for _ in range(12):
        matched = False
        for pat in _DESCRIPTION_PREAMBLE_PATTERNS:
            m = pat.match(t)
            if m:
                t = t[m.end() :].strip()
                matched = True
                break
        if not matched:
            break
    return t


def clean_description(text: str) -> str:
    """Remove common model artifacts (thinking blocks, excessive whitespace)."""
    if not text:
        return ""
    # Strip fenced thinking / analysis blocks often seen in models
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:thinking)?\s*.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Bold/italic markdown the model often injects into descriptions
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _strip_description_search_preamble(text)
    return text


def hostname_from_http_url(url: str | None) -> str | None:
    """Return lowercase hostname from an http(s) URL, or None."""
    if not url or not isinstance(url, str):
        return None
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        return None
    try:
        host = urlparse(u).hostname
        return host.lower() if host else None
    except ValueError:
        return None


def remove_compound_aliases(aliases: list[str]) -> list[str]:
    """Drop an alias if it **contains** another list entry as a strict substring (case-insensitive).

    Keeps shorter canonical tokens (e.g. keep ``O*NET``, drop a long compound that embeds it).
    URLs are skipped so we do not strip hosts against path noise incorrectly.
    """
    cleaned = [a.strip() for a in aliases if a and a.strip()]
    if len(cleaned) < 2:
        return cleaned
    pairs = [(a, a.lower()) for a in cleaned]
    remove: set[int] = set()
    for i, (a, al) in enumerate(pairs):
        if a.startswith(("http://", "https://")):
            continue
        for j, (b, bl) in enumerate(pairs):
            if i == j or len(bl) < 2:
                continue
            if bl in al and len(bl) < len(al):
                remove.add(i)
                break
    return [a for i, (a, _) in enumerate(pairs) if i not in remove]


def is_alias_version_token_only(text: str) -> bool:
    """True if the string looks like a bare version label (digits, dots, v)."""
    t = text.strip()
    if not t or len(t) > 40:
        return False
    return bool(re.fullmatch(r"[\d\.\-v\s]+", t, flags=re.IGNORECASE))


def is_alias_likely_sentence(text: str) -> bool:
    """Heuristic: long prose-like lines are usually not citation aliases."""
    t = text.strip()
    if len(t) < 40:
        return False
    wc = len(t.split())
    return wc >= 10 and (t.endswith(".") or "?" in t)


def is_whole_alias_generic_token(text: str) -> bool:
    """True if the alias is a single generic token (e.g. ``data``, ``survey``)."""
    t = text.strip().lower()
    if not t:
        return True
    if " " in t or "-" in t or "–" in t:
        return False
    return t in _ALIAS_GENERIC_BLOCKLIST


def dedupe_strings_ci_preserve_order(items: list[str]) -> list[str]:
    """Case-insensitive dedupe preserving first-seen casing."""
    seen: set[str] = set()
    out: list[str] = []
    for t in items:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def filter_aliases_by_substrings(aliases: list[str]) -> list[str]:
    """Drop entries that are strict substrings of another (case-insensitive)."""
    if not aliases:
        return []
    sorted_by_len = sorted({a.strip() for a in aliases if a and len(a.strip()) > 1}, key=len, reverse=True)
    result: list[str] = []
    lowered: list[str] = []
    for a in sorted_by_len:
        low = a.lower()
        if any(low != existing and low in existing for existing in lowered):
            continue
        result.append(a)
        lowered.append(low)
    return sorted(result, key=str.lower)


def normalize_access_label(raw: str) -> str:
    low = raw.lower()
    if "restricted" in low:
        return "Restricted"
    if "open" in low:
        return "Open"
    return "Unknown"

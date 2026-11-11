"""One sanitizer for every LLM-produced search term (names, aliases, flags, excludes).

Local models sometimes emit NBSP runs, zero-width characters, non-breaking hyphens,
curly quotes or placeholder ellipses. Those reach Dimensions as lexer errors, so every
entry point that stores or queries an LLM term goes through :func:`sanitize_term`.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable

logger = logging.getLogger(__name__)

_DASHES = dict.fromkeys(
    [*range(0x2010, 0x2016), 0x2212, 0xFE58, 0xFE63, 0xFF0D], "-"
)
_ZERO_WIDTH = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF], None)
_QUOTES = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'", 0x2032: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"', 0x2033: '"',
}
_PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}
_QUOTE_CHARS = "\"'`"


def _strip_edges(s: str) -> str:
    """Strip whitespace, trailing ``.,;:`` and only unbalanced/wrapping brackets and quotes."""
    for _ in range(20):
        prev = s
        s = s.strip().rstrip(".,;:").lstrip(",;:").strip()
        if len(s) >= 2:
            if s[0] in _QUOTE_CHARS and s[-1] == s[0] and s.count(s[0]) == 2:
                s = s[1:-1]
                continue
            if s[0] in _PAIRS and s[-1] == _PAIRS[s[0]] and s.count(s[0]) == 1 and s.count(s[-1]) == 1:
                s = s[1:-1]
                continue
        if s:
            if s[-1] in _QUOTE_CHARS and s.count(s[-1]) % 2 == 1:
                s = s[:-1]
            elif s[0] in _QUOTE_CHARS and s.count(s[0]) % 2 == 1:
                s = s[1:]
            for op, cl in _PAIRS.items():
                if s and s[-1] == cl and s.count(cl) > s.count(op):
                    s = s[:-1]
                if s and s[0] == op and s.count(op) > s.count(cl):
                    s = s[1:]
        if s == prev:
            break
    return s


def sanitize_term(term: object) -> str | None:
    """Return the cleaned term, or ``None`` when it must be dropped."""
    if not isinstance(term, str):
        return None
    s = unicodedata.normalize("NFKC", term)
    s = s.translate(_DASHES).translate(_ZERO_WIDTH).translate(_QUOTES)
    s = "".join(
        " " if (ch.isspace() or unicodedata.category(ch) == "Zs")
        else ch
        for ch in s
        if unicodedata.category(ch) != "Cf"
    )
    s = re.sub(r"\s+", " ", s).strip()
    # Placeholder echoes: check before stripping trailing punctuation.
    if "..." in s or "…" in s:
        return None
    s = _strip_edges(s)
    if not s or not any(ch.isalpha() for ch in s):
        return None
    return s


def sanitize_terms(terms: Iterable[object] | None, label: str = "term") -> list[str]:
    """Sanitize a list, logging each dropped or changed term; order preserved."""
    out: list[str] = []
    for t in terms or []:
        cleaned = sanitize_term(t)
        if cleaned is None:
            logger.warning("Dropped unusable %s %r", label, t)
            continue
        if cleaned != t:
            logger.info("Sanitized %s %r -> %r", label, t, cleaned)
        out.append(cleaned)
    return out

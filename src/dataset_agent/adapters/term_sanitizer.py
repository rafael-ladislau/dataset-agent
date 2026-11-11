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
_STRIP_CHARS = " \t\r\n\"'`.,;:!?()[]{}<>"


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
    s = s.strip(_STRIP_CHARS)
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

"""Parse lists, URLs, and structured sections from model output."""

from __future__ import annotations

import ast
import re
from typing import Any

from dataset_agent.domain.ports import TextExtractorPort


class HeuristicTextExtractor(TextExtractorPort):
    def extract_list(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        # Try Python literal list
        try:
            parsed: Any = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (SyntaxError, ValueError, TypeError):
            pass

        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                parsed = ast.literal_eval(m.group(0))
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except (SyntaxError, ValueError, TypeError):
                pass

        # Line-based bullet list
        lines = []
        for line in text.splitlines():
            line = re.sub(r"^[-*]\s*", "", line.strip())
            if line and not line.lower().startswith("here"):
                lines.append(line)
        return lines[:200]

    def extract_url(self, text: str) -> str | None:
        text = text.strip()
        m = re.search(r"https?://[^\s\"'<>]+", text)
        if m:
            url = m.group(0).rstrip(").,;]")
            return url
        return None

    def extract_section(self, text: str, section_name: str) -> str:
        """Extract content between ===SECTION_NAME=== markers."""
        pattern = rf"==={re.escape(section_name)}===\s*\n?(.*?)(?====|\Z)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            if content.lower() in ("none", "null", "n/a", "not found", ""):
                return ""
            return content
        return ""

    def extract_sections(self, text: str) -> dict[str, str]:
        """Extract all ===SECTION=== blocks into a dict."""
        pattern = r"===([A-Z_]+)===\s*\n?(.*?)(?====|\Z)"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        result: dict[str, str] = {}
        for name, content in matches:
            val = content.strip()
            if val.lower() in ("none", "null", "n/a", "not found", ""):
                val = ""
            result[name.upper()] = val
        return result

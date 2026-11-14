"""Shared contract fixture 'llm-results model task v2' (copied unchanged from extraction)."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "llm-results-model-task-v2.json"

PUB_KEYS = {
    "publication_id", "doi", "title", "citation_count", "fulltext", "analyzed",
    "validation_failed", "validated_ai_models", "category_for", "term_contexts",
}
ROW_KEYS = {"name", "is_mentioned", "is_used", "text_quote", "match_type", "usage_type"}


def load_model_task_v2() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_model_task_v2_fixture_shape():
    body = load_model_task_v2()
    assert {"task_id", "probe", "seed_type", "publications", "summary"} <= set(body)
    assert body["seed_type"] == "ai_model" and body["probe"] is True
    for p in body["publications"]:
        assert PUB_KEYS <= set(p)
        assert "validated_datasets" not in p
        assert set(p["fulltext"]) == {"status", "source", "reason"}
        for row in p["validated_ai_models"]:
            assert set(row) == ROW_KEYS
        for c in p["term_contexts"]:
            assert set(c) == {"term", "role", "part", "text", "match"}

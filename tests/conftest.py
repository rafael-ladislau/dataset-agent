"""Suite-wide defaults. Production turns on the resource map, AI-model full text and the
referent thinking budget (owner decisions 2026-11-14); unit tests keep them off unless a
test enables them, so no test reaches the network or a probe by accident."""

import pytest


@pytest.fixture(autouse=True)
def _production_defaults_off_in_tests(monkeypatch):
    monkeypatch.setenv("DATASET_AGENT_RESOURCE_MAP_ENABLED", "false")
    monkeypatch.setenv("DATASET_AGENT_AI_MODEL_FULLTEXT_ENABLED", "false")
    monkeypatch.setenv("DATASET_AGENT_REFERENT_THINKING_BUDGET", "0")

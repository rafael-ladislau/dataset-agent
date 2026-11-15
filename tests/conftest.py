"""Suite-wide defaults: the resource map is on in production (owner decision 2026-11-14), but
unit tests keep it off unless they enable it explicitly, so no test builds a map against
the network by accident."""

import pytest


@pytest.fixture(autouse=True)
def _resource_map_off_in_tests(monkeypatch):
    monkeypatch.setenv("DATASET_AGENT_RESOURCE_MAP_ENABLED", "false")

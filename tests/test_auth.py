"""Unit tests for the APIKeyAuth adapter."""

from __future__ import annotations

from dataset_agent.adapters.auth import APIKeyAuth


def test_auth_disabled_when_no_keys() -> None:
    auth = APIKeyAuth("")
    assert auth.enabled is False
    # Any key (or none) is accepted when disabled.
    assert auth.is_valid_key(None) is True
    assert auth.is_valid_key("whatever") is True
    assert auth.get_client_id("whatever") == "anonymous"


def test_auth_validates_configured_keys() -> None:
    auth = APIKeyAuth("key-a, key-b ,key-c")
    assert auth.enabled is True
    assert auth.api_keys == ["key-a", "key-b", "key-c"]

    assert auth.is_valid_key("key-a") is True
    assert auth.is_valid_key("key-c") is True
    assert auth.is_valid_key("missing") is False
    assert auth.is_valid_key(None) is False
    assert auth.is_valid_key("") is False


def test_auth_client_id_is_index_based() -> None:
    auth = APIKeyAuth("key-a,key-b")
    assert auth.get_client_id("key-a") == "client_1"
    assert auth.get_client_id("key-b") == "client_2"
    assert auth.get_client_id("unknown") == "unknown"

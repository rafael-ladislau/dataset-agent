"""API key authentication helper."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class APIKeyAuth:
    """Validate API keys from a comma-separated configuration string.

    When no keys are configured, authentication is disabled (all requests are
    allowed) so local/dev usage stays frictionless.
    """

    def __init__(self, api_keys: str = "") -> None:
        self.api_keys = [k.strip() for k in (api_keys or "").split(",") if k.strip()]
        if not self.api_keys:
            logger.warning("No API keys configured; write endpoints are unsecured.")
        else:
            logger.info("API key auth initialized with %d key(s).", len(self.api_keys))

    @property
    def enabled(self) -> bool:
        """True when at least one API key is configured."""
        return bool(self.api_keys)

    def is_valid_key(self, api_key: str | None) -> bool:
        """Return True if *api_key* is valid (or auth is disabled)."""
        if not self.api_keys:
            return True
        is_valid = bool(api_key) and api_key in self.api_keys
        if not is_valid:
            preview = (api_key[:5] + "...") if api_key else "<empty>"
            logger.warning("Invalid API key attempt: %s", preview)
        return is_valid

    def get_client_id(self, api_key: str | None) -> str:
        """Return a stable client identifier derived from the API key index."""
        if not self.api_keys:
            return "anonymous"
        try:
            idx = self.api_keys.index(api_key)  # type: ignore[arg-type]
            return f"client_{idx + 1}"
        except (ValueError, TypeError):
            return "unknown"

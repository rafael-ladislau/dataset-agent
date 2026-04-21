"""Logging configuration for the API and workers (Docker / Uvicorn)."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_application_logging() -> None:
    """
    Ensure loggers under `dataset_agent` write to stderr (INFO by default).
    Idempotent; suitable for Uvicorn with reload and background tasks.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    pkg = logging.getLogger("dataset_agent")
    pkg.setLevel(level)
    if not pkg.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        pkg.addHandler(handler)
    pkg.propagate = False
    _CONFIGURED = True

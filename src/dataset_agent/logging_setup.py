"""Logging configuration for the API and workers (Docker / Uvicorn)."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False

#: Default rotation policy for file handlers.
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5


def configure_application_logging() -> None:
    """
    Ensure loggers under `dataset_agent` write to stderr (INFO by default).
    Idempotent; suitable for Uvicorn with reload and background tasks.

    When ``DATASET_AGENT_LOG_DIR`` is set, also attaches rotating file
    handlers: ``dataset_agent.log`` (all records) and
    ``dataset_agent_errors.log`` (errors only), each rotating at 10 MB
    with 5 backups.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pkg = logging.getLogger("dataset_agent")
    pkg.setLevel(level)
    if not pkg.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        pkg.addHandler(handler)

        log_dir = os.environ.get("DATASET_AGENT_LOG_DIR", "").strip()
        if log_dir:
            _attach_file_handlers(pkg, Path(log_dir), formatter)
    pkg.propagate = False
    _CONFIGURED = True


def _attach_file_handlers(
    logger: logging.Logger, log_dir: Path, formatter: logging.Formatter
) -> None:
    """Attach rotating file handlers (general + errors) under *log_dir*."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Could not create log directory %s; file logging disabled.", log_dir)
        return

    general = RotatingFileHandler(
        log_dir / "dataset_agent.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    general.setFormatter(formatter)
    logger.addHandler(general)

    errors = RotatingFileHandler(
        log_dir / "dataset_agent_errors.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    errors.setLevel(logging.ERROR)
    errors.setFormatter(formatter)
    logger.addHandler(errors)

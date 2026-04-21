"""Uvicorn entry for FastAPI."""

from __future__ import annotations

from typing import Any


def main() -> None:
    import os

    import uvicorn

    from dataset_agent.logging_setup import configure_application_logging

    configure_application_logging()

    reload = os.environ.get("UVICORN_RELOAD", "").lower() in ("1", "true", "yes")
    kwargs: dict[str, Any] = {
        "host": "0.0.0.0",
        "port": int(os.environ.get("UVICORN_PORT", "8000")),
        "factory": False,
    }
    if reload:
        kwargs["reload"] = True
        kwargs["reload_dirs"] = [os.environ.get("UVICORN_RELOAD_DIR", "/app/src")]
    log_level = os.environ.get("UVICORN_LOG_LEVEL", os.environ.get("LOG_LEVEL", "info")).lower()
    kwargs["log_level"] = log_level
    uvicorn.run("dataset_agent.interfaces.api:app", **kwargs)


if __name__ == "__main__":
    main()

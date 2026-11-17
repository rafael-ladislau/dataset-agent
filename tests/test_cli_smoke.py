"""CLI smoke tests (no LLM)."""

from __future__ import annotations

from typer.testing import CliRunner

from dataset_agent.interfaces.cli import app


def test_cli_help_exits_zero() -> None:
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "Dataset metadata" in r.stdout or "run" in r.stdout

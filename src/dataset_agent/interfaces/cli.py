"""Typer CLI."""

from __future__ import annotations

import logging
from typing import Optional

import typer

from dataset_agent.application.research import LiteratureGateFailed
from dataset_agent.domain.models import OptimizeRequest, ResearchRequest
from dataset_agent.settings import Settings

app = typer.Typer(help="Dataset metadata research (LOGICA_DO_PROJETO.md)")


@app.command("run")
def run_cmd(
    dataset_name: str = typer.Argument(..., help="Dataset name"),
    webhook: Optional[str] = typer.Option(None, "--webhook", help="Optional notification URL"),
) -> None:
    import asyncio
    from dataset_agent.bootstrap import run_full_pipeline

    settings = Settings()
    req = ResearchRequest(dataset_name=dataset_name.strip(), webhook_url=webhook)

    try:
        record, path = asyncio.run(run_full_pipeline(req, settings))
    except LiteratureGateFailed as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2) from e
    except Exception as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    typer.echo(record.model_dump_json(indent=2))
    typer.echo(f"\nSaved: {path}", err=True)


@app.command("optimize")
def optimize_cmd(
    dataset_name: str = typer.Argument(..., help="Primary dataset name (Dimensions anchor)"),
    url: Optional[str] = typer.Option(
        None,
        "--url",
        help="Official dataset page URL (aliases / platform heuristics)",
    ),
) -> None:
    """Optimize a Dimensions publications query (runs research unless --names-style inputs are added later)."""
    import asyncio

    from dataset_agent.bootstrap import build_optimize_use_case

    settings = Settings()
    uc = build_optimize_use_case(settings)
    req = OptimizeRequest(dataset_name=dataset_name.strip(), dataset_url=url.strip() if url else None)

    try:
        record = asyncio.run(uc.execute(req))
    except Exception as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    typer.echo(record.model_dump_json(indent=2))
    if not record.success:
        raise typer.Exit(code=2)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    app()


if __name__ == "__main__":
    main()

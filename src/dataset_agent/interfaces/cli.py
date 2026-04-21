"""Typer CLI."""

from __future__ import annotations

import logging
from typing import Optional

import typer

from dataset_agent.application.research import LiteratureGateFailed
from dataset_agent.domain.models import ResearchRequest
from dataset_agent.settings import Settings

app = typer.Typer(help="Dataset metadata research (LOGICA_DO_PROJETO.md)")


@app.command("run")
def run_cmd(
    dataset_name: str = typer.Argument(..., help="Dataset name"),
    webhook: Optional[str] = typer.Option(None, "--webhook", help="Optional notification URL"),
) -> None:
    from dataset_agent.bootstrap import build_use_case

    settings = Settings()
    use_case = build_use_case(settings)

    req = ResearchRequest(dataset_name=dataset_name.strip(), webhook_url=webhook)

    try:
        record, path = use_case.execute(req)
    except LiteratureGateFailed as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2) from e
    except Exception as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    typer.echo(record.model_dump_json(indent=2))
    typer.echo(f"\nSaved: {path}", err=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    app()


if __name__ == "__main__":
    main()

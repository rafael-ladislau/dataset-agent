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
    _echo_program_datasets(record)
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
    _echo_program_datasets(record)
    if not record.success:
        raise typer.Exit(code=2)


def _echo_program_datasets(record) -> None:
    """The program and its datasets, readable, on stderr (nothing when not split)."""
    from dataset_agent.adapters.resource_map import program_datasets_summary

    text = program_datasets_summary(getattr(record, "program_datasets", None))
    if text:
        typer.echo("\n" + text, err=True)


@app.command("optimize-model")
def optimize_model_cmd(
    model_name: str = typer.Argument(..., help="AI model (family) name, e.g. Aurora"),
    hf_id: Optional[str] = typer.Option(
        None, "--hf-id", help="Hugging Face model id, e.g. microsoft/aurora"
    ),
    family: Optional[str] = typer.Option(None, "--family", help="Model family (default: the name)"),
    alias: list[str] = typer.Option(
        [], "--alias", help="Known alias or version string (repeatable); measured like the rest"
    ),
    flag: list[str] = typer.Option(
        [], "--flag", help="Candidate context term (repeatable); kept only if it proves precise"
    ),
    exclude: list[str] = typer.Option(
        [], "--exclude", help="Candidate exclude phrase (repeatable); kept only if it is safe"
    ),
    start_year: Optional[int] = typer.Option(None, "--start-year"),
    end_year: Optional[int] = typer.Option(None, "--end-year"),
) -> None:
    """Build a Dimensions query (and a POST /ai-models/start seed) for an AI model."""
    import asyncio

    from dataset_agent.bootstrap import build_optimize_use_case

    settings = Settings()
    uc = build_optimize_use_case(settings)
    req = OptimizeRequest(
        dataset_name=model_name.strip(),
        entity_type="ai_model",
        hugging_face_id=hf_id,
        model_family=family,
        dataset_names=list(alias),
        flag_terms=list(flag),
        exclude_terms=list(exclude),
        years_start=start_year,
        years_end=end_year,
    )

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

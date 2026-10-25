"""CLI `trt`: кожен крок конвеєра окремо та разом."""

from __future__ import annotations

from pathlib import Path

import typer
from alembic import command as alembic_cmd
from alembic.config import Config as AlembicConfig
from sqlalchemy import text

from trt import __version__
from trt.config import get_settings
from trt.logging import configure_logging, get_logger

app = typer.Typer(help="LLM-based UI localization pipeline (PostgreSQL + pgvector + LangChain).")
log = get_logger("trt.cli")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config() -> AlembicConfig:
    cfg = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    return cfg


def _not_implemented(milestone: str) -> None:
    typer.secho(f"Not implemented yet — planned for {milestone} (see PLAN.md).", fg="yellow")
    raise typer.Exit(code=2)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Детальне логування."),
) -> None:
    settings = get_settings()
    configure_logging("DEBUG" if verbose else settings.log_level)


@app.command()
def version() -> None:
    """Показати версію."""
    typer.echo(__version__)


@app.command()
def check() -> None:
    """Перевірити з'єднання з базою та наявність pgvector."""
    from trt import db

    typer.echo(db.ping())
    if db.has_pgvector():
        typer.secho("pgvector: available", fg="green")
    else:
        typer.secho("pgvector: NOT available — install the extension on the server", fg="red")
        raise typer.Exit(code=1)


@app.command()
def migrate(revision: str = typer.Argument("head")) -> None:
    """Застосувати міграції Alembic (розширення vector, схема rag, індекси)."""
    alembic_cmd.upgrade(_alembic_config(), revision)
    typer.secho(f"migrated to {revision}", fg="green")


@app.command()
def seed(
    file: Path = typer.Option(PROJECT_ROOT / "sql" / "seed_i18n.sql", help="SQL-файл із даними."),
) -> None:
    """Залити тестові дані в i18n_*."""
    from trt.db import get_engine

    sql = file.read_text(encoding="utf-8")
    with get_engine().begin() as conn:
        conn.execute(text(sql))
    typer.secho(f"seeded from {file}", fg="green")


@app.command()
def fetch(
    source: str = typer.Option("pg", help="pg | files | docs | all"),
    since: str | None = typer.Option(None, help="ISO-дата; за замовчуванням — курсор."),
) -> None:
    """Витягнути дані з джерел і зберегти оригінали на диск."""
    _not_implemented("M1 (pg) / M3 (files, docs)")


@app.command()
def ingest() -> None:
    """Витягнути текст з оригіналів і зробити upsert у rag.document."""
    _not_implemented("M2")


@app.command()
def embed(
    all_: bool = typer.Option(False, "--all", help="Перерахувати всі документи."),
    reindex: bool = typer.Option(False, "--reindex", help="Перебудувати після зміни моделі."),
) -> None:
    """Розбити документи на фрагменти, обчислити вкладення, записати в rag.chunk."""
    _not_implemented("M4")


@app.command()
def search(
    query: str,
    n: int = typer.Option(5, "-n"),
    locale: str | None = typer.Option(None, "--locale"),
    kind: str | None = typer.Option(None, "--kind"),
) -> None:
    """kNN-пошук N найближчих фрагментів за текстом запиту."""
    _not_implemented("M5")


@app.command()
def translate(
    locale: str = typer.Option(..., "--locale"),
    limit: int = typer.Option(100, "--limit"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Перекласти нові/застарілі рядки через LLM з translation memory."""
    _not_implemented("M6")


@app.command("run-all")
def run_all() -> None:
    """fetch → ingest → embed → translate для всіх увімкнених локалей."""
    _not_implemented("M6")


@app.command()
def schedule() -> None:
    """Запустити планувальник: run-all кожні INTERVAL_MINUTES."""
    _not_implemented("M7")


if __name__ == "__main__":
    app()

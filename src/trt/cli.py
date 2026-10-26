"""CLI `trt`: кожен крок конвеєра окремо та разом."""

from __future__ import annotations

from datetime import datetime
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
    source: str = typer.Option("all", help="pg | files | docs | all"),
    since: datetime | None = typer.Option(None, help="ISO-дата; за замовчуванням — курсор."),
) -> None:
    """Витягнути дані з джерел і зберегти оригінали на диск (RawStore)."""
    from trt.pipeline import step_fetch

    for r in step_fetch(source, since):
        typer.echo(f"{r.source}: new artifacts={len(r.artifacts)} unchanged={r.skipped}")
        for a in r.artifacts:
            typer.echo(f"  {a.name} -> {a.path}")


@app.command()
def ingest() -> None:
    """Витягнути текст з оригіналів і зробити upsert у rag.document."""
    from trt.pipeline import step_ingest

    s = step_ingest()
    typer.echo(
        f"artifacts={s.artifacts} documents changed={s.upserted} unchanged={s.unchanged} "
        f"errors={len(s.errors)}"
    )
    for e in s.errors:
        typer.secho(f"  ! {e}", fg="red")


@app.command()
def embed(
    all_: bool = typer.Option(
        False, "--all", "--reindex", help="Перерахувати всі документи (після зміни моделі)."
    ),
) -> None:
    """Розбити документи на фрагменти, обчислити вкладення, записати в rag.chunk."""
    from trt.pipeline import step_embed

    s = step_embed(force_all=all_)
    typer.echo(f"documents={s.documents} chunks={s.chunks}")


@app.command()
def search(
    query: str,
    n: int = typer.Option(5, "-n"),
    locale: str | None = typer.Option(None, "--locale"),
    kind: str | None = typer.Option(None, "--kind", help="ui_string | reference_doc"),
) -> None:
    """kNN-пошук N найближчих фрагментів за текстом запиту."""
    from trt.db import session_scope
    from trt.vector_store import search_text

    with session_scope() as session:
        hits = search_text(session, query, n, locale=locale, kind=kind)
    for h in hits:
        snippet = h.text.replace("\n", " ")
        if len(snippet) > 90:
            snippet = snippet[:87] + "..."
        typer.echo(
            f"{h.score:.4f}  chunk={h.chunk_id} doc={h.document_id} [{h.doc_key}]  {snippet}"
        )


@app.command()
def translate(
    locale: list[str] = typer.Option(
        None, "--locale", help="Цільова локаль (можна кілька); за замовчуванням усі увімкнені."
    ),
    limit: int = typer.Option(100, "--limit"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Показати переклади, не писати в базу."),
) -> None:
    """Перекласти нові/застарілі рядки через LLM з translation memory."""
    from trt.pipeline import step_translate

    for r in step_translate(locale or None, limit, dry_run):
        typer.echo(
            f"[{r.locale}] requested={r.requested} accepted={r.accepted} written={r.written} "
            f"review={r.queued_for_review} llm_calls={r.llm_calls} elapsed={r.elapsed_seconds:.1f}s"
        )
        if dry_run:
            for code, src, tgt in r.results:
                typer.echo(f"  {code}: {src!r} -> {tgt!r}")
        for code, reason in r.review_reasons.items():
            typer.secho(f"  review {code}: {reason}", fg="yellow")


@app.command("run-all")
def run_all_cmd(
    limit: int = typer.Option(100, "--limit"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """fetch → ingest → embed → translate для всіх увімкнених локалей."""
    from trt.pipeline import run_all

    report = run_all(limit=limit, dry_run=dry_run)
    if report.skipped_locked:
        typer.secho("another run-all is in progress, skipped", fg="yellow")
        raise typer.Exit(code=3)
    typer.echo(
        f"ingested={report.ingest.upserted if report.ingest else 0} "
        f"chunks={report.embed.chunks if report.embed else 0} "
        f"translated={sum(r.written for r in report.translated)} "
        f"review={sum(r.queued_for_review for r in report.translated)}"
    )


@app.command()
def schedule(
    cron: str | None = typer.Option(None, "--cron", help="Crontab-вираз замість інтервалу."),
    no_initial: bool = typer.Option(False, "--no-initial", help="Не запускати одразу при старті."),
) -> None:
    """Запустити планувальник: run-all кожні INTERVAL_MINUTES (або за cron)."""
    from trt.scheduler import build_scheduler

    sched = build_scheduler(cron, run_now=not no_initial)
    typer.echo(f"scheduler started: {cron or f'every {get_settings().interval_minutes} min'}")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        typer.echo("scheduler stopped")


if __name__ == "__main__":
    app()

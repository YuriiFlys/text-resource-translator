"""Періодичний запуск run-all через APScheduler."""

from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from trt.config import get_settings
from trt.logging import get_logger
from trt.pipeline import run_all

log = get_logger(__name__)


def build_scheduler(cron: str | None = None, run_now: bool = True) -> BlockingScheduler:
    settings = get_settings()
    scheduler = BlockingScheduler(timezone="UTC")
    trigger = (
        CronTrigger.from_crontab(cron)
        if cron
        else IntervalTrigger(minutes=settings.interval_minutes)
    )
    scheduler.add_job(
        _job,
        trigger,
        id="run_all",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=300,
    )
    if run_now:
        scheduler.add_job(_job, id="run_all_initial")
    return scheduler


def _job() -> None:
    log.info("scheduler.run_all.start")
    report = run_all()
    log.info(
        "scheduler.run_all.done",
        locked=report.skipped_locked,
        ingested=report.ingest.upserted if report.ingest else None,
        chunks=report.embed.chunks if report.embed else None,
        translated=sum(r.written for r in report.translated),
    )

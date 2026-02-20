"""Scheduled data collection at 6:30 AM and 8:00 AM PST."""
import logging
from typing import Optional

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.config.settings import settings
from backend.database.db import SessionLocal
from backend.services.data_collection.collector import run_collection

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None
_tz = pytz.timezone(settings.TIMEZONE)


def _scheduled_collection_job(snapshot_type: Optional[str] = None) -> None:
    """Job run by APScheduler; uses its own DB session."""
    logger.info("Scheduled data collection started (snapshot_type=%s)", snapshot_type)
    db = SessionLocal()
    try:
        result = run_collection(
            db,
            snapshot_type_override=snapshot_type,
            capture_screenshots=getattr(
                settings, "COLLECTION_CAPTURE_SCREENSHOTS", True
            ),
        )
        logger.info(
            "Scheduled collection finished: collected=%s failed=%s type=%s",
            result["collected"],
            result["failed"],
            result["snapshot_type"],
        )
        if result["errors"]:
            for err in result["errors"]:
                logger.warning("Collection error: %s", err)
    except Exception as e:
        logger.exception("Scheduled collection failed: %s", e)
    finally:
        db.close()


def get_scheduler() -> Optional[BackgroundScheduler]:
    """Return the global scheduler instance if it exists."""
    return _scheduler


def start_scheduler() -> Optional[BackgroundScheduler]:
    """
    Start the background scheduler with two cron jobs:
    - 6:30 AM PST: before snapshot
    - 8:00 AM PST: after snapshot
    """
    global _scheduler
    if not getattr(settings, "ENABLE_SCHEDULED_COLLECTION", True):
        logger.info("Scheduled data collection is disabled (ENABLE_SCHEDULED_COLLECTION=False)")
        return None

    if _scheduler is not None:
        logger.warning("Scheduler already started")
        return _scheduler

    _scheduler = BackgroundScheduler(timezone=_tz)

    # 6:30 AM local (PST/PDT)
    _scheduler.add_job(
        _scheduled_collection_job,
        trigger=CronTrigger(hour=6, minute=30, timezone=_tz),
        id="before_snapshot",
        name="NY AM before snapshot (6:30)",
        kwargs={"snapshot_type": "before"},
        replace_existing=True,
    )
    # 8:00 AM local
    _scheduler.add_job(
        _scheduled_collection_job,
        trigger=CronTrigger(hour=8, minute=0, timezone=_tz),
        id="after_snapshot",
        name="NY AM after snapshot (8:00)",
        kwargs={"snapshot_type": "after"},
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "Scheduled data collection started: 6:30 AM and 8:00 AM %s",
        settings.TIMEZONE,
    )
    return _scheduler


def stop_scheduler() -> None:
    """Stop the background scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduled data collection stopped")

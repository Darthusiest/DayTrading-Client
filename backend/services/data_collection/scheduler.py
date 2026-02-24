"""Scheduled data collection at session open and close (e.g. 9:30 and 16:00 ET)."""
import logging
from typing import Optional

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.config.settings import settings
from backend.database.db import SessionLocal
from backend.services.data_collection.collector import run_collection, run_session_candle_capture
from backend.services.data_processing.training_data_pipeline import (
    process_training_data_from_snapshots,
    process_training_data_from_session_candles,
    process_training_data_from_bars_only,
)

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None


def _parse_session_time(time_str: str) -> tuple[int, int]:
    """Parse HH:MM to (hour, minute)."""
    parts = time_str.strip().split(":")
    h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    return h, m


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


def _scheduled_session_candle_capture_job() -> None:
    """Job run by APScheduler at session open: capture every candle from first bar to session end (e.g. 9:31–16:00)."""
    logger.info("Scheduled session candle capture started (session start–end)")
    db = SessionLocal()
    try:
        result = run_session_candle_capture(db)
        logger.info(
            "Session candle capture finished: collected=%s failed=%s",
            result["collected"],
            result["failed"],
        )
        for err in result.get("errors", [])[:20]:
            logger.warning("Session candle error: %s", err)
        if len(result.get("errors", [])) > 20:
            logger.warning("... and %s more errors", len(result["errors"]) - 20)
    except Exception as e:
        logger.exception("Session candle capture failed: %s", e)
    finally:
        db.close()


def _scheduled_process_training_data_job() -> None:
    """Job run by APScheduler: create training samples from before/after, session candles, and bar-only (Databento)."""
    logger.info("Scheduled process training data started")
    db = SessionLocal()
    try:
        result_before_after = process_training_data_from_snapshots(db)
        result_session = process_training_data_from_session_candles(db)
        result_bars = process_training_data_from_bars_only(db)
        logger.info(
            "Process training data finished: before/after=%s, session=%s, bars_only=%s",
            result_before_after["created"],
            result_session["created"],
            result_bars["created"],
        )
        for err in (
            result_before_after.get("errors", [])
            + result_session.get("errors", [])
            + result_bars.get("errors", [])
        ):
            logger.warning("Process training data error: %s", err)
    except Exception as e:
        logger.exception("Scheduled process training data failed: %s", e)
    finally:
        db.close()


def get_scheduler() -> Optional[BackgroundScheduler]:
    """Return the global scheduler instance if it exists."""
    return _scheduler


def start_scheduler() -> Optional[BackgroundScheduler]:
    """
    Start the background scheduler with jobs at session open and close (e.g. 9:30 and 16:00 ET).
    Uses SESSION_TIMEZONE, SESSION_START_TIME, SESSION_END_TIME from settings.
    """
    global _scheduler
    if not getattr(settings, "ENABLE_SCHEDULED_COLLECTION", True):
        logger.info("Scheduled data collection is disabled (ENABLE_SCHEDULED_COLLECTION=False)")
        return None

    if _scheduler is not None:
        logger.warning("Scheduler already started")
        return _scheduler

    session_tz = pytz.timezone(settings.SESSION_TIMEZONE)
    start_h, start_m = _parse_session_time(settings.SESSION_START_TIME)
    end_h, end_m = _parse_session_time(settings.SESSION_END_TIME)
    # Process training data 5 minutes after session close
    process_h, process_m = end_h, end_m + 5
    if process_m >= 60:
        process_h, process_m = process_h + 1, process_m - 60

    _scheduler = BackgroundScheduler(timezone=session_tz)

    if getattr(settings, "ENABLE_SESSION_CANDLE_CAPTURE", False):
        _scheduler.add_job(
            _scheduled_session_candle_capture_job,
            trigger=CronTrigger(hour=start_h, minute=start_m, timezone=session_tz),
            id="session_candle_capture",
            name="Session candle capture (session start–end)",
            replace_existing=True,
        )
    _scheduler.add_job(
        _scheduled_collection_job,
        trigger=CronTrigger(hour=start_h, minute=start_m, timezone=session_tz),
        id="before_snapshot",
        name="Before snapshot (session open)",
        kwargs={"snapshot_type": "before"},
        replace_existing=True,
    )
    _scheduler.add_job(
        _scheduled_collection_job,
        trigger=CronTrigger(hour=end_h, minute=end_m, timezone=session_tz),
        id="after_snapshot",
        name="After snapshot (session close)",
        kwargs={"snapshot_type": "after"},
        replace_existing=True,
    )
    _scheduler.add_job(
        _scheduled_process_training_data_job,
        trigger=CronTrigger(hour=process_h, minute=process_m, timezone=session_tz),
        id="process_training_data",
        name="Process training data (after close)",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "Scheduled data collection started: %s:%02d, %s:%02d %s",
        start_h, start_m, end_h, end_m, settings.SESSION_TIMEZONE,
    )
    return _scheduler


def stop_scheduler() -> None:
    """Stop the background scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduled data collection stopped")

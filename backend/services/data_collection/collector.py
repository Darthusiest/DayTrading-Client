"""Shared data collection logic for scheduled and manual runs."""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pytz
from sqlalchemy.orm import Session

from backend.config.settings import settings
from backend.database.models import Snapshot, SessionMinuteBar
from backend.services.data_collection.screenshot_capture import ScreenshotCapture
from backend.services.evaluation.outcome_feedback import update_predictions_with_outcomes

logger = logging.getLogger(__name__)

# Intervals (minutes) and bar-close rules for session candle capture (session start–end from config)
SESSION_CANDLE_INTERVALS = (1, 5, 15, 60)


def _parse_session_time(time_str: str) -> tuple[int, int]:
    """Parse HH:MM string to (hour, minute)."""
    parts = time_str.strip().split(":")
    h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    return h, m


def _intervals_closing_at(hour: int, minute: int, session_end_hour: int, session_end_minute: int) -> list[int]:
    """Return which of (1, 5, 15, 60) close at this (hour, minute). Session end used for 1h bar boundary."""
    out = [1]  # 1m closes every minute
    if minute % 5 == 0:
        out.append(5)
    if minute % 15 == 0:
        out.append(15)
    # 1h bars close at :00 or :30; only include up to session end
    if (minute == 0 or minute == 30) and (hour < session_end_hour or (hour == session_end_hour and minute <= session_end_minute)):
        out.append(60)
    return out


def _collect_session_minute_bars(
    db: Session,
    session_date: str,
    tz: pytz.BaseTzInfo,
) -> int:
    """
    Placeholder: minute bars are ingested from Databento (data/databento/raw/).
    Returns 0. Session minute bars must be loaded via a Databento ingestion script.
    """
    logger.debug(
        "Session minute bars for %s: use Databento ingestion (data/databento/raw/)",
        session_date,
    )
    return 0


def run_collection(
    db: Session,
    session_date: Optional[str] = None,
    snapshot_type_override: Optional[str] = None,
    capture_screenshots: bool = True,
) -> dict:
    """
    Collect before or after snapshots for configured symbols only (MNQ & MES by default).

    When run at session open (e.g. 9:30 ET), collects "before"; at session close (e.g. 16:00 ET), "after".
    If snapshot_type_override is set, use that instead of inferring from current time.
    Uses SESSION_TIMEZONE for before/after decision.

    Args:
        db: Database session (caller manages lifecycle).
        session_date: Date string YYYY-MM-DD; default is today in session timezone.
        snapshot_type_override: If "before" or "after", skip time check and use this.
        capture_screenshots: If False, skip screenshots (no Selenium).

    Returns:
        Dict with keys: collected (int), failed (int), snapshot_type (str), errors (list).
    """
    session_tz = pytz.timezone(settings.SESSION_TIMEZONE)
    now = datetime.now(session_tz)
    if session_date is None:
        session_date = now.strftime("%Y-%m-%d")

    before_time = datetime.strptime(settings.BEFORE_SNAPSHOT_TIME, "%H:%M").time()
    after_time = datetime.strptime(settings.AFTER_SNAPSHOT_TIME, "%H:%M").time()
    current_time = now.time()

    if snapshot_type_override in ("before", "after"):
        snapshot_type = snapshot_type_override
    elif current_time <= before_time:
        snapshot_type = "before"
    elif current_time >= after_time:
        snapshot_type = "after"
    else:
        logger.info(
            "Skipping collection: current time %s is between before (%s) and after (%s)",
            current_time,
            before_time,
            after_time,
        )
        return {
            "collected": 0,
            "failed": 0,
            "snapshot_type": "none",
            "errors": ["Current time is between before and after snapshot windows"],
        }

    screenshot_capture = ScreenshotCapture() if capture_screenshots else None
    collected = 0
    failed = 0
    errors = []

    try:
        for symbol in settings.SYMBOLS:
            try:
                logger.info("Collecting %s snapshot for %s...", snapshot_type, symbol)

                image_path = None
                if capture_screenshots and screenshot_capture:
                    image_path = screenshot_capture.capture_chart_screenshot(
                        symbol=symbol,
                        snapshot_type=snapshot_type,
                        session_date=session_date,
                    )
                if image_path is not None or not capture_screenshots:
                    path_str = str(image_path) if image_path else f"(price-only {symbol} {snapshot_type} {session_date})"
                    snapshot = Snapshot(
                        symbol=symbol,
                        snapshot_type=snapshot_type,
                        timestamp=now,
                        image_path=path_str,
                        session_date=session_date,
                    )
                    db.add(snapshot)
                    db.commit()
                    collected += 1
                    logger.info("Saved %s snapshot for %s", snapshot_type, symbol)
                else:
                    failed += 1
                    errors.append(f"Failed to capture screenshot for {symbol}")
            except Exception as e:
                logger.exception("Error collecting %s for %s", snapshot_type, symbol)
                db.rollback()
                failed += 1
                errors.append(f"{symbol}: {e}")

        # After close snapshot: minute bars come from Databento ingestion; update prediction outcomes
        if snapshot_type == "after":
            try:
                _collect_session_minute_bars(db, session_date, session_tz)
            except Exception as e:
                logger.exception("Session minute bars placeholder failed: %s", e)
                errors.append(f"Session minute bars: {e}")
            try:
                n = update_predictions_with_outcomes(db, session_date)
                if n:
                    logger.info("Outcome feedback: updated %s predictions for %s", n, session_date)
            except Exception as e:
                logger.exception("Outcome feedback failed: %s", e)
                errors.append(f"Outcome feedback: {e}")
    finally:
        if screenshot_capture:
            screenshot_capture.close()

    return {
        "collected": collected,
        "failed": failed,
        "snapshot_type": snapshot_type,
        "errors": errors,
    }


def run_session_candle_capture(
    db: Session,
    session_date: Optional[str] = None,
) -> dict:
    """
    Capture every candle between session start and end at 1m, 5m, 15m, 1h for each configured symbol.
    Runs from first bar after start to session end (e.g. 9:31–16:00 ET). One browser session; login once.
    Saves snapshots with snapshot_type='session_candle', interval_minutes, bar_time.
    No-op if ENABLE_SESSION_CANDLE_CAPTURE is False (long-running; disabled by default).

    Returns:
        Dict with collected (int), failed (int), errors (list).
    """
    if not getattr(settings, "ENABLE_SESSION_CANDLE_CAPTURE", False):
        return {"collected": 0, "failed": 0, "errors": ["Session candle capture is disabled (ENABLE_SESSION_CANDLE_CAPTURE=False)"]}

    session_tz = pytz.timezone(settings.SESSION_TIMEZONE)
    now = datetime.now(session_tz)
    if session_date is None:
        session_date = now.strftime("%Y-%m-%d")

    start_h, start_m = _parse_session_time(settings.SESSION_START_TIME)
    end_h, end_m = _parse_session_time(settings.SESSION_END_TIME)
    total_minutes = (end_h * 60 + end_m) - (start_h * 60 + start_m)  # e.g. 390 for 9:30–16:00

    # Build list of (bar_time, intervals) from first bar after start to session end
    base_date = datetime.strptime(session_date, "%Y-%m-%d").date()
    bar_schedule: list[tuple[datetime, list[int]]] = []
    for minute_offset in range(1, total_minutes + 1):
        dt = datetime.combine(base_date, datetime.strptime(settings.SESSION_START_TIME, "%H:%M").time()) + timedelta(minutes=minute_offset)
        dt = session_tz.localize(dt) if dt.tzinfo is None else dt
        h, m = dt.hour, dt.minute
        bar_schedule.append((dt, _intervals_closing_at(h, m, end_h, end_m)))

    screenshot_capture = ScreenshotCapture()
    collected = 0
    failed = 0
    errors = []

    try:
        screenshot_capture._init_driver()
        screenshot_capture._ensure_logged_in()

        for bar_time, intervals in bar_schedule:
            now = datetime.now(session_tz)
            if now < bar_time:
                wait_secs = (bar_time - now).total_seconds()
                if wait_secs > 0:
                    logger.debug("Session candle: waiting %.0fs until %s", wait_secs, bar_time.strftime("%H:%M"))
                    time.sleep(min(wait_secs, 65))  # cap 65s so we don't oversleep past next minute

            bar_time_naive = bar_time.replace(tzinfo=None) if bar_time.tzinfo else bar_time

            for symbol in settings.SYMBOLS:
                for interval in intervals:
                    try:
                        existing = db.query(Snapshot).filter(
                            Snapshot.symbol == symbol,
                            Snapshot.session_date == session_date,
                            Snapshot.snapshot_type == "session_candle",
                            Snapshot.interval_minutes == interval,
                            Snapshot.bar_time == bar_time_naive,
                        ).first()
                        if existing:
                            logger.debug("Session candle: skip existing %s %sm at %s", symbol, interval, bar_time_naive.strftime("%H:%M"))
                            continue
                        image_path = screenshot_capture.capture_chart_screenshot(
                            symbol=symbol,
                            snapshot_type="session_candle",
                            session_date=session_date,
                            interval_minutes=interval,
                            bar_time=bar_time_naive,
                        )
                        if image_path is None:
                            failed += 1
                            errors.append(f"{symbol} {interval}m at {bar_time_naive}: capture failed")
                            continue
                        snapshot = Snapshot(
                            symbol=symbol,
                            snapshot_type="session_candle",
                            timestamp=bar_time_naive,
                            image_path=str(image_path),
                            session_date=session_date,
                            interval_minutes=interval,
                            bar_time=bar_time_naive,
                        )
                        db.add(snapshot)
                        db.commit()
                        collected += 1
                        logger.debug("Session candle: %s %sm at %s", symbol, interval, bar_time_naive.strftime("%H:%M"))
                    except Exception as e:
                        logger.exception("Session candle %s %sm at %s: %s", symbol, interval, bar_time_naive, e)
                        db.rollback()
                        failed += 1
                        errors.append(f"{symbol} {interval}m at {bar_time_naive}: {e}")

    finally:
        screenshot_capture.close()

    return {"collected": collected, "failed": failed, "errors": errors}


def capture_snapshot_now(
    db: Session,
    symbol: str = "MNQ1!",
    interval_minutes: Optional[int] = None,
) -> dict:
    """
    Log in to TradingView (if credentials set), capture a screenshot of the current
    chart for the given symbol at the given timeframe, save the image to disk, and
    store a Snapshot row in the database.

    Args:
        db: Database session.
        symbol: MNQ1! or MES1!.
        interval_minutes: Chart timeframe in minutes (1, 5, 15, 60, 240, 1440). If None, uses settings default.

    Returns:
        Dict with snapshot_id, image_path, symbol, session_date, interval_minutes, and success (bool),
        or error message on failure.
    """
    if symbol not in settings.SYMBOLS:
        return {
            "success": False,
            "error": f"Symbol {symbol} not in configured SYMBOLS ({settings.SYMBOLS})",
        }
    tz = pytz.timezone(settings.TIMEZONE)
    now = datetime.now(tz)
    session_date = now.strftime("%Y-%m-%d")
    interval = interval_minutes if interval_minutes is not None else getattr(settings, "CHART_INTERVAL_MINUTES", 15)
    screenshot_capture = ScreenshotCapture()
    try:
        image_path = screenshot_capture.capture_chart_screenshot(
            symbol=symbol,
            snapshot_type="manual",
            session_date=session_date,
            interval_minutes=interval_minutes,
        )
        if image_path is None:
            return {
                "success": False,
                "error": f"Failed to capture screenshot for {symbol}",
            }
        path_str = str(image_path)
        snapshot = Snapshot(
            symbol=symbol,
            snapshot_type="manual",
            timestamp=now,
            image_path=path_str,
            session_date=session_date,
        )
        db.add(snapshot)
        db.commit()
        db.refresh(snapshot)
        return {
            "success": True,
            "snapshot_id": snapshot.id,
            "image_path": path_str,
            "symbol": symbol,
            "session_date": session_date,
            "interval_minutes": interval,
            "timestamp": now.isoformat(),
        }
    except Exception as e:
        logger.exception("capture_snapshot_now failed: %s", e)
        db.rollback()
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        screenshot_capture.close()

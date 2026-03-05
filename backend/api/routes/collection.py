"""Data collection API: manual trigger, scheduler status, and training data processing."""
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.config.settings import settings
from backend.services.data_collection.collector import run_collection, capture_snapshot_now, run_session_candle_capture
from backend.services.data_collection.scheduler import get_scheduler
from backend.services.data_processing.training_data_pipeline import (
    process_training_data_from_bars_only,
)
from backend.services.data_collection.databento_ingestion import run_ingestion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collection", tags=["collection"])


@router.post("/run")
def run_collection_now(
    capture_screenshots: bool = True,
    db: Session = Depends(get_db),
):
    """
    Run data collection once now (before or after snapshot based on current time).
    Optionally disable screenshot capture.
    """
    result = run_collection(
        db,
        capture_screenshots=capture_screenshots,
    )
    return result


@router.post("/capture-now")
def capture_chart_now(
    symbol: str = "MNQ1!",
    interval: int = 15,
    db: Session = Depends(get_db),
):
    """
    Fetch a screenshot of the current TradingView chart for the selected symbol and
    timeframe. The backend logs in (if credentials are set), opens the chart at the
    requested interval, and saves the image so the AI model can use the same chart
    the user is looking at.

    **Query parameters:**
    - **symbol**: `MNQ1!` (Micro E-mini Nasdaq) or `MES1!` (Micro E-mini S&P 500). Default: MNQ1!.
    - **interval**: Chart timeframe in minutes. Allowed: 1, 5, 15, 60, 240, 1440 (1m, 5m, 15m, 1h, 4h, 1D). Default: 15.

    **Example (frontend button):** `POST /api/v1/collection/capture-now?symbol=MNQ1!&interval=15`

    **Where it is saved:**
    - **Database**: Table `snapshots`. One row per capture (symbol, snapshot_type \"manual\", image_path, session_date, etc.). Price data in `price_data` when available.
    - **Filesystem**: `data/raw/<symbol>_manual_<session_date>_<time>.png`
    """
    if interval not in (1, 5, 15, 60, 240, 1440):
        raise HTTPException(
            status_code=400,
            detail="interval must be one of: 1, 5, 15, 60, 240, 1440 (minutes)",
        )
    return capture_snapshot_now(db, symbol=symbol, interval_minutes=interval)


@router.post("/run-session-candles")
def run_session_candles_now(
    session_date: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Run session candle capture from first bar after session start to session end (e.g. 9:31–16:00 ET) at 1m, 5m, 15m, 1h for all symbols.
    No-op if ENABLE_SESSION_CANDLE_CAPTURE is False. Optional query: session_date=YYYY-MM-DD (default: today).
    """
    return run_session_candle_capture(db, session_date=session_date)


@router.post("/ingest-databento")
def ingest_databento_now(
    path: str | None = None,
    dry_run: bool = False,
    db: Session = Depends(get_db),
):
    """
    Ingest Databento OHLCV-1m batch files from data/databento/raw/ (or optional path) into session_minute_bars.
    Optional query: path=<file or dir> to restrict to a single file or directory; dry_run=true to decode only.
    """
    path_override = Path(path) if path else None
    return run_ingestion(db, path_override=path_override, dry_run=dry_run)


@router.post("/process-training-data")
def process_training_data_now(db: Session = Depends(get_db)):
    """
    Build training samples **only** from bar-only (Databento SessionMinuteBar) data.

    This endpoint no longer uses screenshot-based snapshots for training; it creates
    one `TrainingSample` per (session_date, symbol) using aggregated OHLCV bars so
    the model trains purely on numerical market data.
    """
    result_bars = process_training_data_from_bars_only(db)
    return {
        "bars_only": result_bars,
        "created": result_bars["created"],
        "skipped": result_bars["skipped"],
        "errors": result_bars["errors"],
    }


@router.get("/schedule")
def get_schedule_status():
    """Return whether scheduled collection is enabled and next run times."""
    sched = get_scheduler()
    jobs = []
    if sched:
        for j in sched.get_jobs():
            jobs.append({
                "id": j.id,
                "name": j.name,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
            })
    return {
        "enabled": getattr(settings, "ENABLE_SCHEDULED_COLLECTION", True),
        "timezone": settings.TIMEZONE,
        "session_timezone": settings.SESSION_TIMEZONE,
        "session_start": settings.SESSION_START_TIME,
        "session_end": settings.SESSION_END_TIME,
        "before_time": settings.BEFORE_SNAPSHOT_TIME,
        "after_time": settings.AFTER_SNAPSHOT_TIME,
        "scheduler_running": sched is not None,
        "jobs": jobs,
    }

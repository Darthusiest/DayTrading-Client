"""Data collection API: manual trigger, scheduler status, and training data processing."""
import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.config.settings import settings
from backend.services.data_collection.collector import run_collection
from backend.services.data_collection.scheduler import get_scheduler
from backend.services.data_processing.training_data_pipeline import process_training_data_from_snapshots

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collection", tags=["collection"])


@router.post("/run")
def run_collection_now(
    capture_screenshots: bool = True,
    db: Session = Depends(get_db),
):
    """
    Run data collection once now (before or after snapshot based on current time).
    Optionally disable screenshot capture to only fetch Polygon price data.
    """
    result = run_collection(
        db,
        capture_screenshots=capture_screenshots,
    )
    return result


@router.post("/process-training-data")
def process_training_data_now(db: Session = Depends(get_db)):
    """
    Process before/after snapshot pairs into training samples (preprocess images,
    extract features, create labels). Idempotent: skips pairs that already have a sample.
    """
    result = process_training_data_from_snapshots(db)
    return result


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
        "before_time": settings.BEFORE_SNAPSHOT_TIME,
        "after_time": settings.AFTER_SNAPSHOT_TIME,
        "scheduler_running": sched is not None,
        "jobs": jobs,
    }

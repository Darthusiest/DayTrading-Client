"""Helpers for persisted job lifecycle management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from backend.database.models import JobRun


def create_job(
    db: Session,
    job_type: str,
    status: str = "pending",
    details: dict[str, Any] | None = None,
    requested_by: str | None = None,
) -> JobRun:
    started_at = datetime.utcnow() if status == "running" else None
    job = JobRun(
        job_id=str(uuid4()),
        job_type=job_type,
        status=status,
        details=details or {},
        requested_by=requested_by,
        started_at=started_at,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_job(
    db: Session,
    job_id: str,
    *,
    status: str | None = None,
    error: str | None = None,
    details: dict[str, Any] | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> JobRun | None:
    job = db.query(JobRun).filter(JobRun.job_id == job_id).first()
    if not job:
        return None
    if status is not None:
        job.status = status
    if error is not None:
        job.error = error
    if details is not None:
        merged = dict(job.details or {})
        merged.update(details)
        job.details = merged
    if started_at is not None:
        job.started_at = started_at
    if finished_at is not None:
        job.finished_at = finished_at
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> JobRun | None:
    return db.query(JobRun).filter(JobRun.job_id == job_id).first()


def count_recent_failures(db: Session, lookback_hours: int = 24) -> int:
    since = datetime.utcnow() - timedelta(hours=lookback_hours)
    return (
        db.query(JobRun)
        .filter(JobRun.status == "failed", JobRun.created_at >= since)
        .count()
    )


"""Backtest API endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.config.settings import settings
from backend.database.db import get_db
from backend.services.pipeline.orchestrator import run_backtest
from backend.services.trading.strategy import RiskConfig
from backend.api.deps.security import require_api_key
from backend.services.ops.jobs import create_job, get_job, update_job
from datetime import datetime

router = APIRouter(prefix="/backtest", tags=["backtest"])


class EventHourBacktestRequest(BaseModel):
    prefix: str = Field(default="event_hour_backtest")
    batch_size: int = Field(default=512, ge=1, le=4096)
    risk_per_trade: float = Field(default=0.01, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fee_per_trade: float = Field(default=0.10, ge=0.0)
    slippage_bps: float = Field(default=1.0, ge=0.0)


@router.post("/event-hour/run", dependencies=[Depends(require_api_key)])
def run_event_hour_backtest(req: EventHourBacktestRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    job = create_job(
        db,
        job_type="backtest_event_hour",
        status="running",
        details={"prefix": req.prefix, "batch_size": req.batch_size},
    )
    try:
        result = run_backtest(
            "event_hour",
            prefix=req.prefix,
            batch_size=req.batch_size,
            risk=RiskConfig(
                risk_per_trade=req.risk_per_trade,
                min_confidence=req.min_confidence,
                fee_per_trade=req.fee_per_trade,
                slippage_bps=req.slippage_bps,
            ),
        )
        update_job(
            db,
            job.job_id,
            status="completed",
            started_at=job.started_at or datetime.utcnow(),
            finished_at=datetime.utcnow(),
            details={
                "summary_path": result.get("summary_path"),
                "trades_path": result.get("trades_path"),
                "curves_path": result.get("curves_path"),
            },
        )
        return {"job_id": job.job_id, **result}
    except Exception as exc:
        update_job(db, job.job_id, status="failed", error=str(exc), finished_at=datetime.utcnow())
        raise


@router.get("/event-hour/summary")
def get_event_hour_backtest_summary(prefix: str = "event_hour_backtest") -> dict[str, Any]:
    path = settings.MODELS_DIR / f"{prefix}_summary.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Backtest summary not found: {path}")
    with path.open("r") as f:
        return json.load(f)


@router.get("/event-hour/trades")
def get_event_hour_backtest_trades(prefix: str = "event_hour_backtest") -> list[dict[str, Any]]:
    path = settings.MODELS_DIR / f"{prefix}_trades.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Backtest trades not found: {path}")
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="Backtest trades file is invalid.")
    return data


@router.get("/jobs/{job_id}")
def get_backtest_job(job_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Backtest job not found: {job_id}")
    return {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "status": job.status,
        "error": job.error,
        "details": job.details or {},
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


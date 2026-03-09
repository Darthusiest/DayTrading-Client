"""Main FastAPI application."""
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from backend.config.settings import settings
from backend.database.db import init_db
from backend.database.db import SessionLocal
from backend.database.models import JobRun
from backend.api.routes import predict, training, evaluation, notes, collection, event_hour, backtest
from backend.services.data_collection.scheduler import get_scheduler, start_scheduler, stop_scheduler
from backend.services.ops.jobs import count_recent_failures
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized")
    start_scheduler()
    yield
    # Shutdown
    stop_scheduler()
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Day Trading AI Agent Backend API",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    elapsed_ms = (perf_counter() - started) * 1000.0
    logger.info(
        "request path=%s method=%s status=%s latency_ms=%.2f",
        request.url.path,
        request.method,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": str(exc.detail)}},
        )
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred.",
            }
        },
    )

# Include routers
app.include_router(predict.router, prefix=settings.API_V1_PREFIX)
app.include_router(training.router, prefix=settings.API_V1_PREFIX)
app.include_router(evaluation.router, prefix=settings.API_V1_PREFIX)
app.include_router(notes.router, prefix=settings.API_V1_PREFIX)
app.include_router(collection.router, prefix=settings.API_V1_PREFIX)
app.include_router(event_hour.router, prefix=settings.API_V1_PREFIX)
app.include_router(backtest.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/diagnostics")
def diagnostics():
    """Operational diagnostics for DB, scheduler, and model artifacts."""
    db_ok = True
    db_error = None
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_ok = False
        db_error = str(exc)
    finally:
        db.close()

    model_files = ["event_hour_continuation.pt", "event_hour_reversal.pt", "next_minute_lstm.pt"]
    model_status = {}
    now = datetime.now(timezone.utc)
    for name in model_files:
        path = settings.MODELS_DIR / name
        if path.is_file():
            stat = path.stat()
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            model_status[name] = {
                "exists": True,
                "bytes": stat.st_size,
                "modified_at": modified_at.isoformat(),
                "age_minutes": round((now - modified_at).total_seconds() / 60.0, 2),
            }
        else:
            model_status[name] = {"exists": False}

    scheduler = get_scheduler()
    scheduler_jobs = []
    if scheduler is not None:
        for job in scheduler.get_jobs():
            scheduler_jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                }
            )

    def _dir_size(path):
        return sum(p.stat().st_size for p in path.glob("**/*") if p.is_file()) if path.exists() else 0

    data_usage = {
        "total_bytes": _dir_size(settings.DATA_DIR),
        "raw_bytes": _dir_size(settings.RAW_DATA_DIR),
        "processed_bytes": _dir_size(settings.PROCESSED_DATA_DIR),
        "models_bytes": _dir_size(settings.MODELS_DIR),
        "databento_raw_bytes": _dir_size(settings.DATABENTO_RAW_DIR),
    }

    recent_failed_jobs = 0
    recent_jobs = []
    if db_ok:
        db = SessionLocal()
        try:
            recent_failed_jobs = count_recent_failures(db, lookback_hours=24)
            rows = db.query(JobRun).order_by(JobRun.created_at.desc()).limit(10).all()
            for row in rows:
                recent_jobs.append(
                    {
                        "job_id": row.job_id,
                        "job_type": row.job_type,
                        "status": row.status,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
                    }
                )
        finally:
            db.close()
    return {
        "status": "ok" if db_ok else "degraded",
        "run_mode": settings.RUN_MODE,
        "db": {"ok": db_ok, "error": db_error},
        "models": model_status,
        "scheduler_running": scheduler is not None,
        "scheduler_jobs": scheduler_jobs,
        "storage": data_usage,
        "recent_failed_jobs_24h": recent_failed_jobs,
        "recent_jobs": recent_jobs,
        "api_key_required": settings.REQUIRE_API_KEY,
    }

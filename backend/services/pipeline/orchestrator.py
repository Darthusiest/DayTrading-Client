"""Orchestrates collection, processing, training, and backtesting tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from backend.services.data_collection.collector import run_collection
from backend.services.data_collection.databento_ingestion import run_ingestion
from backend.services.data_processing.training_data_pipeline import (
    process_training_data_from_bars_only,
    process_training_data_from_snapshots,
)

logger = logging.getLogger(__name__)


def process_session(
    db: Session,
    capture_screenshots: bool = True,
    snapshot_type: str | None = None,
) -> dict[str, Any]:
    """Collect a session snapshot using the shared collector pipeline."""
    return run_collection(
        db,
        snapshot_type_override=snapshot_type,
        capture_screenshots=capture_screenshots,
    )


def build_datasets(
    db: Session,
    mode: str = "bars_only",
) -> dict[str, Any]:
    """Build training dataset artifacts using a unified entrypoint."""
    if mode == "bars_only":
        return process_training_data_from_bars_only(db)
    if mode == "snapshots":
        return process_training_data_from_snapshots(db)
    raise ValueError(f"Unsupported dataset build mode: {mode}")


def ingest_market_data(
    db: Session,
    path_override: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest raw market data files into minute bars."""
    return run_ingestion(db, path_override=path_override, dry_run=dry_run)


def train_model(model_name: str) -> dict[str, Any]:
    """Run model training script by high-level model name."""
    if model_name == "event_hour":
        from scripts.train_event_hour_models import main as event_main

        event_main()
        return {"status": "ok", "model_name": model_name}
    if model_name == "next_minute":
        from scripts.train_next_minute_model import main as nm_main

        nm_main()
        return {"status": "ok", "model_name": model_name}
    raise ValueError(f"Unsupported model_name: {model_name}")


def run_backtest(strategy_name: str, **kwargs: Any) -> dict[str, Any]:
    """Run strategy backtest by strategy alias."""
    if strategy_name == "event_hour":
        from scripts.backtest_event_hour import run_backtest as run_event_hour_backtest

        return run_event_hour_backtest(**kwargs)
    raise ValueError(f"Unsupported strategy_name: {strategy_name}")


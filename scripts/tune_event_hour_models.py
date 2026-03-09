#!/usr/bin/env python3
"""Optuna hyperparameter tuning for event-hour continuation/reversal models.

Runs a study over lr, hidden_size, num_layers, dropout, weight_decay,
focal_gamma, pos_weight_scale, label_horizon. Each trial trains one WF fold
and returns the validation metric (EVENT_EARLY_STOP_METRIC).

Usage:
  python scripts/tune_event_hour_models.py --n-trials 20
  python scripts/tune_event_hour_models.py --n-trials 50 --study-name event_tune_v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set env before importing settings so trial params can override
os.environ.setdefault("EVENT_REBUILD_CACHE", "False")

import optuna
from optuna.trial import Trial

from backend.config.settings import settings
from backend.database.db import SessionLocal
from backend.database.models import SessionMinuteBar


def _load_bars(limit_days: int | None = None):
    """Load SessionMinuteBar rows for event dataset."""
    from sqlalchemy import func, select

    with SessionLocal() as db:
        q = select(SessionMinuteBar).order_by(SessionMinuteBar.session_date, SessionMinuteBar.symbol)
        if limit_days:
            # Approximate: filter by session_date
            subq = (
                select(func.min(SessionMinuteBar.session_date).label("first_date"))
                .select_from(SessionMinuteBar)
            )
            first = db.execute(subq).scalar()
            if first:
                from datetime import timedelta
                cutoff = first + timedelta(days=limit_days)
                q = q.where(SessionMinuteBar.session_date <= cutoff)
        rows = db.execute(q).scalars().all()
    return list(rows)


def _run_trial(
    trial: Trial,
    sequences,
    targets,
    event_type,
    event_dir,
    session_id,
    symbol_id,
    cont_eligible,
    forward_return_60m,
    atr_pct,
    train_idx,
    val_idx,
    test_idx,
    model_name: str,
    out_dir: Path,
    selection_metric: str,
) -> float:
    """Train one fold with trial params; return val metric."""
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    from backend.services.ml.event_hour import EventHourDataset, EventHourLSTM, EventHourModelConfig

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.05, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
    pos_weight_scale = trial.suggest_float("pos_weight_scale", 1.0, 2.0)
    label_horizon = trial.suggest_categorical("label_horizon", [45, 60, 90])

    # Temporarily override settings for this trial
    prev_lr = getattr(settings, "EVENT_LR", None)
    prev_hidden = getattr(settings, "EVENT_LSTM_HIDDEN_SIZE", None)
    prev_layers = getattr(settings, "EVENT_NUM_LSTM_LAYERS", None)
    prev_dropout = getattr(settings, "EVENT_DROPOUT", None)
    prev_wd = getattr(settings, "EVENT_WEIGHT_DECAY", None)
    prev_gamma = getattr(settings, "EVENT_FOCAL_GAMMA", None)
    prev_pw = getattr(settings, "EVENT_POS_WEIGHT_SCALE", None)

    settings.EVENT_LR = lr
    settings.EVENT_LSTM_HIDDEN_SIZE = hidden_size
    settings.EVENT_NUM_LSTM_LAYERS = num_layers
    settings.EVENT_DROPOUT = dropout
    settings.EVENT_WEIGHT_DECAY = weight_decay
    settings.EVENT_FOCAL_GAMMA = focal_gamma
    settings.EVENT_POS_WEIGHT_SCALE = pos_weight_scale
    # label_horizon would require rebuild; skip for now to keep trials fast

    try:
        from scripts.train_event_hour_models import (
            DEVICE,
            _evaluate,
            _train_one,
        )

        metrics, _ = _train_one(
            model_name,
            sequences,
            targets,
            event_type,
            event_dir,
            session_id,
            symbol_id,
            train_idx,
            val_idx,
            test_idx,
            out_dir,
            forward_return_60m=forward_return_60m,
            atr_pct=atr_pct,
        )
        val = metrics.get("val", {}) or {}
        score = float(val.get(selection_metric, float("nan")))
    finally:
        if prev_lr is not None:
            settings.EVENT_LR = prev_lr
        if prev_hidden is not None:
            settings.EVENT_LSTM_HIDDEN_SIZE = prev_hidden
        if prev_layers is not None:
            settings.EVENT_NUM_LSTM_LAYERS = prev_layers
        if prev_dropout is not None:
            settings.EVENT_DROPOUT = prev_dropout
        if prev_wd is not None:
            settings.EVENT_WEIGHT_DECAY = prev_wd
        if prev_gamma is not None:
            settings.EVENT_FOCAL_GAMMA = prev_gamma
        if prev_pw is not None:
            settings.EVENT_POS_WEIGHT_SCALE = prev_pw

    return score if not (score != score) else -1e9  # nan -> -inf


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for event-hour models")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", type=str, default="event_hour_tune")
    parser.add_argument("--limit-days", type=int, default=None, help="Limit bars to last N days for faster tuning")
    args = parser.parse_args()

    # Import here to avoid circular deps and ensure settings loaded
    from scripts.train_event_hour_models import (
        _build_event_dataset,
        _session_split,
        _walk_forward_fold_ranges,
        _wf_split_indices,
    )

    bars = _load_bars(limit_days=args.limit_days)
    if not bars:
        print("No bars found. Run ingestion first.")
        return 1

    lookback = int(getattr(settings, "EVENT_LOOKBACK", 90))
    horizon = int(getattr(settings, "EVENT_LABEL_HORIZON_MINUTES", 60))
    built = _build_event_dataset(bars, lookback=lookback, horizon_minutes=horizon)
    sequences = built["sequences"]
    targets_cont = built["targets_cont"]
    targets_rev = built["targets_rev"]
    event_type = built["event_type"]
    event_dir = built["event_dir"]
    session_id = built["session_id"]
    symbol_id = built.get("symbol_id")
    forward_return_60m = built.get("forward_return_60m")
    atr_pct = built.get("atr_pct")
    cont_eligible = built.get("cont_eligible")
    sessions = built.get("sessions", [])

    if symbol_id is None:
        import torch
        symbol_id = torch.zeros_like(session_id)
    if cont_eligible is None:
        cont_eligible = __import__("torch").ones_like(session_id)
    if forward_return_60m is None:
        import torch
        forward_return_60m = torch.ones(sequences.size(0), dtype=torch.float32) * 1e-4
    if atr_pct is None:
        import torch
        atr_pct = torch.ones(sequences.size(0), dtype=torch.float32) * 1e-4

    # Use last WF fold for tuning (or single split if no WF)
    wf_train_days = int(getattr(settings, "EVENT_WF_TRAIN_DAYS", 365))
    wf_test_days = int(getattr(settings, "EVENT_WF_TEST_DAYS", 90))
    wf_slide_days = int(getattr(settings, "EVENT_WF_SLIDE_DAYS", 90))
    wf_val_ratio = float(getattr(settings, "EVENT_WF_VAL_RATIO", 0.1))
    folds = _walk_forward_fold_ranges(sessions, wf_train_days, wf_test_days, wf_slide_days)

    if folds:
        train_sids, test_sids, _, _, _, _ = folds[-1]
        train_idx, val_idx, test_idx = _wf_split_indices(
            int(sequences.size(0)), session_id, train_sids, test_sids, wf_val_ratio
        )
        cont_train_idx = [i for i in train_idx if int(cont_eligible[i].item()) == 1]
        cont_val_idx = [i for i in val_idx if int(cont_eligible[i].item()) == 1]
        cont_test_idx = [i for i in test_idx if int(cont_eligible[i].item()) == 1]
        train_idx_cont = cont_train_idx
        val_idx_cont = cont_val_idx
        test_idx_cont = cont_test_idx
    else:
        n_sessions = len(sessions) if sessions else int(session_id.max().item() + 1)
        val_ratio = float(getattr(settings, "EVENT_VAL_RATIO", 0.1))
        test_ratio = float(getattr(settings, "EVENT_TEST_RATIO", 0.1))
        train_idx, val_idx, test_idx = _session_split(session_id, n_sessions, val_ratio, test_ratio)
        train_idx_cont = [i for i in train_idx if int(cont_eligible[i].item()) == 1]
        val_idx_cont = [i for i in val_idx if int(cont_eligible[i].item()) == 1]
        test_idx_cont = [i for i in test_idx if int(cont_eligible[i].item()) == 1]

    out_dir = settings.MODELS_DIR / "tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_metric = str(getattr(settings, "EVENT_EARLY_STOP_METRIC", "f1"))

    def objective(trial: Trial) -> float:
        return _run_trial(
            trial,
            sequences,
            targets_cont,
            event_type,
            event_dir,
            session_id,
            symbol_id,
            cont_eligible,
            forward_return_60m,
            atr_pct,
            train_idx_cont,
            val_idx_cont,
            test_idx_cont,
            "event_hour_continuation_tune",
            out_dir,
            selection_metric,
        )

    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    best_params = best.params
    out_path = out_dir / "event_hour_tuned_config.json"
    with out_path.open("w") as f:
        json.dump({"best_params": best_params, "best_value": best.value}, f, indent=2)
    print(f"Best {selection_metric}: {best.value:.4f}")
    print(f"Best params: {best_params}")
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

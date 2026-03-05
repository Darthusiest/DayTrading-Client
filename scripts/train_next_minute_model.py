"""Train a 1m next-minute LSTM model on SessionMinuteBar data and report metrics.

Usage (from project root):

  python scripts/train_next_minute_model.py

This script:
  1. Loads all SessionMinuteBar rows for symbols in settings.SYMBOLS.
  2. Builds sliding windows of length LOOKBACK (per session, per symbol).
  3. Predicts the NEXT bar's close price from each window.
  4. Splits chronologically into train/val/test.
  5. Trains an LSTM (NextMinuteBarLSTM) on train, reports MAE/RMSE/direction accuracy
     on val and test.
  6. Saves the trained weights and metrics for reuse and evaluation:
       - models/next_minute_lstm.pt
       - models/next_minute_metrics.json
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config.settings import settings  # noqa: E402
from backend.database.db import SessionLocal  # noqa: E402
from backend.database.models import SessionMinuteBar  # noqa: E402
from backend.services.ml.bar_next_minute import (  # noqa: E402
    NextMinuteBarDataset,
    NextMinuteBarLSTM,
    NextMinuteModelConfig,
)


LOOKBACK = getattr(settings, "BAR_LOOKBACK", 60)  # number of minutes in input window
BATCH_SIZE = getattr(settings, "BAR_BATCH_SIZE", 256)
EPOCHS = getattr(settings, "BAR_NUM_EPOCHS", 10)
LEARNING_RATE = getattr(settings, "BAR_LEARNING_RATE", settings.LEARNING_RATE)
VAL_RATIO = getattr(settings, "BAR_VAL_SPLIT", 0.1)
TEST_RATIO = getattr(settings, "BAR_TEST_SPLIT", 0.1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_bars() -> List[SessionMinuteBar]:
    """Load all SessionMinuteBar rows for the configured symbols, ordered chronologically."""
    db = SessionLocal()
    try:
        q = (
            db.query(SessionMinuteBar)
            .filter(SessionMinuteBar.symbol.in_(settings.SYMBOLS))
            .order_by(
                SessionMinuteBar.symbol,
                SessionMinuteBar.session_date,
                SessionMinuteBar.bar_time,
            )
        )
        rows = q.all()
        return rows
    finally:
        db.close()


def _build_sequences(
    bars: List[SessionMinuteBar],
    lookback: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build sliding windows of enriched bar features and next-bar close targets.

    Returns:
        sequences: [N, T, F] float32  (F >= 5: OHLCV + derived indicators + time features)
        targets:   [N] float32        (next-bar close price)
    """
    by_key: Dict[Tuple[str, str], List[SessionMinuteBar]] = defaultdict(list)
    for b in bars:
        by_key[(b.symbol, b.session_date)].append(b)

    sequences = []
    targets = []

    for (symbol, session_date), group in sorted(by_key.items(), key=lambda x: x[0][1]):
        if len(group) <= lookback:
            continue

        # Ensure bars are strictly ordered within the session
        group = sorted(group, key=lambda g: g.bar_time)

        import numpy as np

        n_bars = len(group)
        # Base OHLCV features
        feat = np.zeros((n_bars, 5), dtype="float32")
        for i, g in enumerate(group):
            feat[i, 0] = g.open_price
            feat[i, 1] = g.high_price
            feat[i, 2] = g.low_price
            feat[i, 3] = g.close_price
            feat[i, 4] = float(g.volume or 0.0)

        closes = feat[:, 3].copy()
        vols = feat[:, 4].copy()

        # === 1) Price-based derived indicators (session-local) ===
        returns = np.zeros(n_bars, dtype="float32")
        log_returns = np.zeros(n_bars, dtype="float32")
        if n_bars > 1:
            prev_close = closes[:-1].copy()
            curr_close = closes[1:].copy()
            valid = prev_close > 0
            ret = np.zeros_like(curr_close)
            ret[valid] = (curr_close[valid] / prev_close[valid]) - 1.0
            returns[1:] = ret

            log_ret = np.zeros_like(curr_close)
            log_ret[valid] = np.log(curr_close[valid] / prev_close[valid])
            log_returns[1:] = log_ret

        # Rolling volatility of returns (e.g. 20-bar window)
        vol_window = min(20, max(2, n_bars))
        rolling_vol = np.zeros(n_bars, dtype="float32")
        for i in range(n_bars):
            start = max(0, i - vol_window + 1)
            window = returns[start : i + 1]
            if window.size >= 2:
                rolling_vol[i] = float(window.std(ddof=1))

        # VWAP (cumulative within session)
        vwap = np.zeros(n_bars, dtype="float32")
        pv = closes * vols
        cum_pv = np.cumsum(pv)
        cum_v = np.cumsum(vols)
        nonzero = cum_v > 0
        vwap[nonzero] = cum_pv[nonzero] / cum_v[nonzero]
        if not nonzero[0]:
            vwap[0] = closes[0]

        # RSI (14-period, simple approximation)
        rsi = np.full(n_bars, 50.0, dtype="float32")
        period = min(14, max(2, n_bars // 4))  # adapt for very short sessions
        if n_bars > period:
            deltas = np.diff(closes)
            gains = np.clip(deltas, 0, None)
            losses = -np.clip(deltas, None, 0)
            for i in range(period, n_bars):
                g_win = gains[i - period : i]
                l_win = losses[i - period : i]
                avg_gain = g_win.mean()
                avg_loss = l_win.mean()
                if avg_loss == 0:
                    rsi[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        # MACD (12, 26, 9) on closes
        macd = np.zeros(n_bars, dtype="float32")
        fast_period, slow_period, signal_period = 12, 26, 9
        if n_bars >= slow_period + signal_period:
            def ema(x: np.ndarray, span: int) -> np.ndarray:
                alpha = 2.0 / (span + 1.0)
                out = np.zeros_like(x, dtype="float32")
                out[0] = x[0]
                for j in range(1, x.size):
                    out[j] = alpha * x[j] + (1 - alpha) * out[j - 1]
                return out

            ema_fast = ema(closes, fast_period)
            ema_slow = ema(closes, slow_period)
            macd_line = ema_fast - ema_slow
            signal_line = ema(macd_line, signal_period)
            macd[:] = macd_line - signal_line

        # Momentum: close[t] - close[t-n]
        mom_period = min(10, max(1, n_bars // 6))
        momentum = np.zeros(n_bars, dtype="float32")
        if n_bars > mom_period:
            momentum[mom_period:] = closes[mom_period:] - closes[:-mom_period]

        # === 2) Time features from bar_time and session config ===
        from datetime import datetime as _dt

        hours = np.zeros(n_bars, dtype="float32")
        minutes = np.zeros(n_bars, dtype="float32")
        day_of_week = np.zeros(n_bars, dtype="float32")
        minutes_since_open = np.zeros(n_bars, dtype="float32")
        is_ny_open = np.zeros(n_bars, dtype="float32")
        is_power_hour = np.zeros(n_bars, dtype="float32")

        # Parse session start/end once
        def _parse_hm(s: str) -> tuple[int, int]:
            parts = s.strip().split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            return h, m

        start_h, start_m = _parse_hm(settings.SESSION_START_TIME)
        end_h, end_m = _parse_hm(settings.SESSION_END_TIME)
        start_total = start_h * 60 + start_m
        end_total = end_h * 60 + end_m

        for i, g in enumerate(group):
            bt = g.bar_time
            # bt is naive in session TZ; use its components directly.
            h = bt.hour
            m = bt.minute
            hours[i] = float(h)
            minutes[i] = float(m)
            day_of_week[i] = float(bt.weekday())

            total_min = h * 60 + m
            ms_open = max(0, total_min - start_total)
            minutes_since_open[i] = float(ms_open)

            # Example flags: first 60 minutes of session as "NY open", last 60 as "power hour"
            is_ny_open[i] = 1.0 if 0 <= ms_open <= 60 else 0.0
            mins_to_close = end_total - total_min
            is_power_hour[i] = 1.0 if 0 <= mins_to_close <= 60 else 0.0

        # Stack all features: OHLCV + derived + time
        derived_cols = np.stack(
            [
                returns,
                log_returns,
                rolling_vol,
                vwap,
                rsi,
                macd,
                momentum,
                hours,
                minutes,
                day_of_week,
                minutes_since_open,
                is_ny_open,
                is_power_hour,
            ],
            axis=1,
        ).astype("float32")

        all_feat = np.concatenate([feat, derived_cols], axis=1)

        # Sliding window: predict bar i close from previous `lookback` bars
        # Window covers indices [i-lookback, i-1], target is close[i]
        n = n_bars
        for i in range(lookback, n):
            window = all_feat[i - lookback : i]
            target = closes[i]
            sequences.append(window)
            targets.append(target)

    if not sequences:
        raise RuntimeError(
            f"No sequences built. Check that SessionMinuteBar has data and LOOKBACK={lookback} is not too large."
        )

    import numpy as np

    seq_arr = np.stack(sequences).astype("float32")
    tgt_arr = np.asarray(targets, dtype="float32")

    return torch.from_numpy(seq_arr), torch.from_numpy(tgt_arr)


def _time_split_indices(n: int, val_ratio: float, test_ratio: float) -> Tuple[range, range, range]:
    """Return (train_idx, val_idx, test_idx) ranges for chronological split."""
    n_val = max(1, int(n * val_ratio)) if n >= 10 else max(0, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio)) if n >= 10 else max(0, int(n * test_ratio))
    n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = max(1, n - max(1, n_test))
        n_val = max(0, n - n_train - n_test)
    train_idx = range(0, n_train)
    val_idx = range(n_train, n_train + n_val)
    test_idx = range(n_train + n_val, n)
    return train_idx, val_idx, test_idx


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> Dict[str, float]:
    """Compute MAE, RMSE, and directional accuracy on a dataloader."""
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n_total = 0
    dir_correct = 0
    dir_total = 0

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(DEVICE)
            tgt = batch["target"].to(DEVICE)

            preds = model(seq)

            diff = preds - tgt
            mae_sum += torch.sum(torch.abs(diff)).item()
            mse_sum += torch.sum(diff * diff).item()
            n_total += tgt.numel()

            # Direction accuracy: sign(pred_next - last_input_close) vs sign(tgt_next - last_input_close)
            last_close = seq[:, -1, 3]  # close feature index = 3
            pred_dir = torch.sign(preds - last_close)
            tgt_dir = torch.sign(tgt - last_close)
            # Map small moves to 0 (sideways)
            thresh = 0.0
            pred_dir = torch.where(torch.abs(preds - last_close) <= thresh, torch.zeros_like(pred_dir), pred_dir)
            tgt_dir = torch.where(torch.abs(tgt - last_close) <= thresh, torch.zeros_like(tgt_dir), tgt_dir)

            same = (pred_dir == tgt_dir).float()
            dir_correct += same.sum().item()
            dir_total += same.numel()

    if n_total == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "direction_accuracy": float("nan")}

    mae = mae_sum / n_total
    rmse = math.sqrt(mse_sum / n_total)
    direction_accuracy = dir_correct / dir_total if dir_total else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "direction_accuracy": direction_accuracy,
        "samples": n_total,
    }


def main() -> None:
    print("Loading SessionMinuteBar data...")
    bars = _load_bars()
    if not bars:
        print("No SessionMinuteBar rows found. Run Databento ingestion first.")
        return
    print(f"Loaded {len(bars)} minute bars.")

    print(f"Building sequences with LOOKBACK={LOOKBACK}...")
    sequences, targets = _build_sequences(bars, LOOKBACK)
    n = sequences.size(0)
    print(f"Built {n} sequences.")

    train_idx, val_idx, test_idx = _time_split_indices(n, VAL_RATIO, TEST_RATIO)
    print(
        f"Split sizes -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
        f"(VAL_RATIO={VAL_RATIO}, TEST_RATIO={TEST_RATIO})"
    )

    train_dataset = NextMinuteBarDataset(sequences[train_idx], targets[train_idx])
    val_dataset = NextMinuteBarDataset(sequences[val_idx], targets[val_idx]) if len(val_idx) else None
    test_dataset = NextMinuteBarDataset(sequences[test_idx], targets[test_idx]) if len(test_idx) else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if test_dataset else None

    input_size = sequences.size(2)
    config = NextMinuteModelConfig(input_size=input_size)
    model = NextMinuteBarLSTM(config).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(
        f"Training NextMinuteBarLSTM on device={DEVICE} for {EPOCHS} epochs "
        f"(batch_size={BATCH_SIZE}, lr={LEARNING_RATE})"
    )

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_seen = 0

        for batch in train_loader:
            seq = batch["sequence"].to(DEVICE)
            tgt = batch["target"].to(DEVICE)

            optimizer.zero_grad()
            preds = model(seq)
            loss = criterion(preds, tgt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * tgt.numel()
            n_seen += tgt.numel()

        train_mse = epoch_loss / max(1, n_seen)
        train_rmse = math.sqrt(train_mse)

        val_metrics = {"mae": float("nan"), "rmse": float("nan"), "direction_accuracy": float("nan")}
        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| train_rmse={train_rmse:.5f} "
            f"| val_rmse={val_metrics['rmse']:.5f} "
            f"| val_mae={val_metrics['mae']:.5f} "
            f"| val_dir_acc={val_metrics['direction_accuracy']:.4f}"
        )

        if not math.isnan(val_metrics["rmse"]) and val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = model.state_dict()

    # Use best validation model (if we found one)
    if best_state is not None:
        model.load_state_dict(best_state)

    print("Evaluating on validation and test sets...")
    val_summary = _evaluate(model, val_loader) if val_loader is not None else None
    test_summary = _evaluate(model, test_loader) if test_loader is not None else None

    models_dir = settings.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "next_minute_lstm.pt"
    torch.save(model.state_dict(), ckpt_path)

    metrics = {
        "lookback": LOOKBACK,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "val": val_summary,
        "test": test_summary,
    }
    metrics_path = models_dir / "next_minute_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved next-minute model to: {ckpt_path}")
    print(f"Saved metrics JSON to: {metrics_path}")
    print("Validation metrics:", val_summary)
    print("Test metrics:", test_summary)


if __name__ == "__main__":
    main()


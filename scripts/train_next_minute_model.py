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
    Build sliding windows of OHLCV features and next-bar close targets.

    Returns:
        sequences: [N, T, F] float32  (F=5: open, high, low, close, volume)
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

        feat = np.zeros((len(group), 5), dtype="float32")
        for i, g in enumerate(group):
            feat[i, 0] = g.open_price
            feat[i, 1] = g.high_price
            feat[i, 2] = g.low_price
            feat[i, 3] = g.close_price
            feat[i, 4] = float(g.volume or 0.0)

        closes = feat[:, 3]

        # Sliding window: predict bar i close from previous `lookback` bars
        # Window covers indices [i-lookback, i-1], target is close[i]
        n = len(group)
        for i in range(lookback, n):
            window = feat[i - lookback : i]
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


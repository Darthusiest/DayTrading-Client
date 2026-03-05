"""Train a 1m next-minute LSTM model on SessionMinuteBar data and report metrics.

Usage (from project root):

  python scripts/train_next_minute_model.py

This script:
  1. Loads all SessionMinuteBar rows for symbols in settings.SYMBOLS.
  2. Builds sliding windows of length BAR_LOOKBACK (per session, per symbol), or loads
     them from cache (BAR_CACHE_DATASET=True) if a valid next_minute_dataset.pt exists.
  3. Predicts next-bar 1m return (not raw price), plus 5m direction, 10m volatility, and 10m breakout.
  4. Splits chronologically into train/val/test.
  5. Trains an LSTM (NextMinuteBarLSTM) with configurable multi-task loss weights;
     early stopping on validation return_rmse (reported as price_rmse). Reports
     return MAE/RMSE, direction_5m_accuracy, volatility_10m_rmse, breakout_10m_accuracy on val and test.
  6. Saves the trained weights and metrics:
       - models/next_minute_lstm.pt
       - models/next_minute_metrics.json

Targets and normalization:
  - Price head predicts return_1m (optionally z-scored per session via BAR_NORMALIZE_RETURN_TARGET).
  - Inputs are per-session standardized when BAR_NORMALIZE_INPUTS=True.
  - Vol target is realized vol of returns (optionally z-scored via BAR_NORMALIZE_VOL_TARGET).
  - Breakout is defined relative to ATR (BAR_BREAKOUT_ATR_K, BAR_ATR_WINDOW).

Lookback: BAR_LOOKBACK (default 30). Shorter lookbacks (15–30) are recommended for
next-minute microstructure; 60 can add noise.

Cache: Set BAR_REBUILD_CACHE=true or delete models/next_minute_dataset.pt to force
rebuilding sequences after config or data changes.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
from backend.services.ml.bar_features import build_session_feature_matrix  # noqa: E402


LOOKBACK = settings.BAR_LOOKBACK
BATCH_SIZE = getattr(settings, "BAR_BATCH_SIZE", 256)
EPOCHS = getattr(settings, "BAR_NUM_EPOCHS", 10)
LEARNING_RATE = getattr(settings, "BAR_LEARNING_RATE", settings.LEARNING_RATE)
VAL_RATIO = getattr(settings, "BAR_VAL_SPLIT", 0.1)
TEST_RATIO = getattr(settings, "BAR_TEST_SPLIT", 0.1)
USE_CACHE = getattr(settings, "BAR_CACHE_DATASET", True)
REBUILD_CACHE = getattr(settings, "BAR_REBUILD_CACHE", False)
CACHE_PATH = settings.MODELS_DIR / "next_minute_dataset.pt"
EARLY_STOP_PATIENCE = getattr(settings, "BAR_EARLY_STOP_PATIENCE", 5)
EARLY_STOP_MIN_DELTA = getattr(settings, "BAR_EARLY_STOP_MIN_DELTA", 0.0)
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


def _cache_metadata_matches(data: dict, expected: dict) -> bool:
    """Return True if cache metadata matches expected (lookback, symbols, target_type, normalization, etc.)."""
    for key, expected_val in expected.items():
        if data.get(key) != expected_val:
            return False
    return True


def _load_cached_sequences(
    cache_path: Path,
    expected_lookback: int,
    expected_symbols: List[str],
    num_bars_expected: Optional[int] = None,
    expected_target_type: str = "return",
    expected_normalize_inputs: Optional[bool] = None,
    expected_normalize_return: Optional[bool] = None,
    expected_normalize_vol: Optional[bool] = None,
    expected_dir5_threshold: Optional[float] = None,
    expected_breakout_atr_k: Optional[float] = None,
) -> Optional[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
]:
    """
    Load sequences and targets from a .pt cache if it exists and matches config.
    Returns None if cache is missing, invalid, or stale (e.g. different lookback/symbols/target_type).
    """
    if not cache_path.is_file():
        return None
    try:
        data = torch.load(cache_path, weights_only=False)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("sequences", "targets_price", "targets_dir5", "targets_vol10", "targets_breakout"):
        if key not in data:
            return None
    expected_meta = {
        "lookback": expected_lookback,
        "symbols": expected_symbols,
        "target_type": expected_target_type,
    }
    if expected_normalize_inputs is not None:
        expected_meta["normalize_inputs"] = expected_normalize_inputs
    if expected_normalize_return is not None:
        expected_meta["normalize_return"] = expected_normalize_return
    if expected_normalize_vol is not None:
        expected_meta["normalize_vol"] = expected_normalize_vol
    if expected_dir5_threshold is not None:
        expected_meta["dir5_threshold"] = expected_dir5_threshold
    if expected_breakout_atr_k is not None:
        expected_meta["breakout_atr_k"] = expected_breakout_atr_k
    if not _cache_metadata_matches(data, expected_meta):
        return None
    if num_bars_expected is not None and data.get("num_bars") != num_bars_expected:
        return None
    n = data["sequences"].size(0)
    if data["targets_price"].size(0) != n:
        return None
    return (
        data["sequences"],
        data["targets_price"],
        data["targets_dir5"],
        data["targets_vol10"],
        data["targets_breakout"],
    )


def _save_sequences_cache(
    cache_path: Path,
    sequences: torch.Tensor,
    targets_price: torch.Tensor,
    targets_dir5: torch.Tensor,
    targets_vol10: torch.Tensor,
    targets_breakout: torch.Tensor,
    lookback: int,
    symbols: List[str],
    num_bars: int,
    min_session_date: str,
    max_session_date: str,
    target_type: str = "return",
    normalize_inputs: bool = False,
    normalize_return: bool = False,
    normalize_vol: bool = False,
    dir5_threshold: float = 0.002,
    breakout_atr_k: float = 1.0,
) -> None:
    """Save built sequences and metadata to a .pt file for future runs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sequences": sequences,
            "targets_price": targets_price,
            "targets_dir5": targets_dir5,
            "targets_vol10": targets_vol10,
            "targets_breakout": targets_breakout,
            "lookback": lookback,
            "symbols": symbols,
            "num_samples": sequences.size(0),
            "num_bars": num_bars,
            "min_session_date": min_session_date,
            "max_session_date": max_session_date,
            "target_type": target_type,
            "normalize_inputs": normalize_inputs,
            "normalize_return": normalize_return,
            "normalize_vol": normalize_vol,
            "dir5_threshold": dir5_threshold,
            "breakout_atr_k": breakout_atr_k,
        },
        cache_path,
    )


def _build_sequences(
    bars: List[SessionMinuteBar],
    lookback: int,
    normalize_inputs: bool = True,
    normalize_return_target: bool = False,
    normalize_vol_target: bool = False,
    dir5_threshold: float = 0.002,
    breakout_atr_k: float = 1.0,
    atr_window: int = 14,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build sliding windows of enriched bar features and multi-horizon targets.

    Returns:
        sequences:        [N, T, F] float32  (F >= 5: OHLCV + derived; optionally per-session normalized)
        targets_price:    [N]       float32  (next-bar 1m return, optionally z-scored per session)
        targets_dir5:     [N]       int64    (direction next 5m: 0=down,1=sideways,2=up)
        targets_vol10:    [N]       float32  (realized vol next 10m; optionally z-scored per session)
        targets_breakout: [N]       float32  (0/1 breakout next 10m, vol-relative)
    """
    by_key: Dict[Tuple[str, str], List[SessionMinuteBar]] = defaultdict(list)
    for b in bars:
        by_key[(b.symbol, b.session_date)].append(b)

    sequences: List[torch.Tensor] = []
    targets_price: List[float] = []
    targets_dir5: List[int] = []
    targets_vol10: List[float] = []
    targets_breakout: List[float] = []

    items = sorted(by_key.items(), key=lambda x: x[0][1])
    total_sessions = len(items)
    if total_sessions:
        print(f"Building sequences across {total_sessions} sessions...")

    for idx, ((symbol, session_date), group) in enumerate(items, start=1):
        if len(group) <= lookback:
            continue

        # Compute feature matrix for this session (centralized implementation)
        all_feat, closes, highs, lows = build_session_feature_matrix(group)
        n_bars = all_feat.shape[0]

        # Per-session input normalization
        if normalize_inputs and n_bars > 0:
            feat_mean = all_feat.mean(axis=0)
            feat_std = all_feat.std(axis=0)
            eps = 1e-8
            all_feat = (all_feat - feat_mean) / (feat_std + eps)

        # Precompute session 1m returns and vol10 for optional z-scoring
        session_returns_1m = np.zeros(n_bars, dtype="float32")
        for j in range(1, n_bars):
            if closes[j - 1] > 0:
                session_returns_1m[j] = (closes[j] - closes[j - 1]) / closes[j - 1]
        max_ahead = 10
        vol10_per_i: List[float] = []
        for ii in range(lookback, n_bars - max_ahead):
            fw = closes[ii : ii + max_ahead + 1]
            rets = np.zeros(max_ahead, dtype="float32")
            for k in range(max_ahead):
                if fw[k] > 0:
                    rets[k] = (fw[k + 1] - fw[k]) / fw[k]
            vol10_per_i.append(float(rets.std(ddof=1)) if rets.size >= 2 else 0.0)
        ret_slice = session_returns_1m[lookback : n_bars - max_ahead]
        ret_mean = float(ret_slice.mean()) if ret_slice.size else 0.0
        ret_std = float(ret_slice.std()) + 1e-8 if ret_slice.size else 1.0
        vol_mean = float(np.mean(vol10_per_i)) if vol10_per_i else 0.0
        vol_std = float(np.std(vol10_per_i)) + 1e-8 if vol10_per_i else 1.0

        # ATR (average true range) for vol-relative breakout
        tr = np.zeros(n_bars, dtype="float32")
        for j in range(1, n_bars):
            hl = highs[j] - lows[j]
            hc = abs(highs[j] - closes[j - 1])
            lc = abs(lows[j] - closes[j - 1])
            tr[j] = max(hl, hc, lc)
        atr = np.zeros(n_bars, dtype="float32")
        w = min(atr_window, n_bars)
        for j in range(n_bars):
            start = max(0, j - w + 1)
            atr[j] = float(tr[start : j + 1].mean())

        # Sliding window with future horizons:
        #   - predict next-bar 1m return
        #   - direction next 5m (i -> i+5)
        #   - volatility next 10m (i+1..i+10)
        #   - breakout in next 10m: 1 iff move beyond recent range by > k*ATR
        n = n_bars
        if n <= lookback + max_ahead:
            continue

        for i in range(lookback, n - max_ahead):
            # History window for input sequence
            window = all_feat[i - lookback : i]

            # 1) Next-bar return target (1m forward return from close at i-1 to close at i)
            prev_close = closes[i - 1]
            curr_close = closes[i]
            if prev_close > 0:
                return_1m = (curr_close - prev_close) / prev_close
            else:
                return_1m = 0.0
            if normalize_return_target:
                return_1m = (return_1m - ret_mean) / ret_std

            # 2) Direction next 5m based on 5-bar ahead close
            idx_5 = i + 5
            if idx_5 >= n:
                continue
            c_now = closes[i]
            c_5 = closes[idx_5]
            if c_now > 0:
                ret_5 = (c_5 - c_now) / c_now
            else:
                ret_5 = 0.0
            if ret_5 > dir5_threshold:
                dir_class = 2  # up
            elif ret_5 < -dir5_threshold:
                dir_class = 0  # down
            else:
                dir_class = 1  # sideways

            # 3) Volatility next 10m: std of 1m returns over [i..i+9] -> prices [i..i+10]
            future_window = closes[i : i + max_ahead + 1]  # length 11
            rets_10 = np.zeros(max_ahead, dtype="float32")
            for k in range(max_ahead):
                c0 = future_window[k]
                c1 = future_window[k + 1]
                if c0 > 0:
                    rets_10[k] = (c1 - c0) / c0
            if rets_10.size >= 2:
                vol10 = float(rets_10.std(ddof=1))
            else:
                vol10 = 0.0
            if normalize_vol_target:
                vol10 = (vol10 - vol_mean) / vol_std

            # 4) Breakout: 1 iff future range breaks out by more than k*ATR (vol-relative)
            L_back = min(30, i + 1)
            start_hist = i - L_back + 1
            recent_high = float(highs[start_hist : i + 1].max())
            recent_low = float(lows[start_hist : i + 1].min())
            future_high = float(highs[i + 1 : i + max_ahead + 1].max())
            future_low = float(lows[i + 1 : i + max_ahead + 1].min())
            atr_i = float(atr[i])
            threshold = breakout_atr_k * atr_i if atr_i > 0 else 0.0
            up_break = (future_high - recent_high) > threshold
            down_break = (recent_low - future_low) > threshold
            breakout = 1.0 if (up_break or down_break) else 0.0

            sequences.append(window)
            targets_price.append(return_1m)
            targets_dir5.append(dir_class)
            targets_vol10.append(vol10)
            targets_breakout.append(breakout)

        # Lightweight progress indicator for sequence-building phase
        if total_sessions and (idx % max(1, total_sessions // 20) == 0 or idx == total_sessions):
            pct = idx / total_sessions * 100.0
            print(f"Sequence build progress: {idx}/{total_sessions} sessions ({pct:.1f}%)", end="\r", flush=True)

    if total_sessions:
        print()  # move to next line after progress output

    if not sequences:
        raise RuntimeError(
            f"No sequences built. Check that SessionMinuteBar has data and LOOKBACK={lookback} is not too large."
        )

    seq_arr = np.stack(sequences).astype("float32")
    tgt_price_arr = np.asarray(targets_price, dtype="float32")
    tgt_dir5_arr = np.asarray(targets_dir5, dtype="int64")
    tgt_vol10_arr = np.asarray(targets_vol10, dtype="float32")
    tgt_brk_arr = np.asarray(targets_breakout, dtype="float32")

    return (
        torch.from_numpy(seq_arr),
        torch.from_numpy(tgt_price_arr),
        torch.from_numpy(tgt_dir5_arr),
        torch.from_numpy(tgt_vol10_arr),
        torch.from_numpy(tgt_brk_arr),
    )


def _plot_training_curves(
    history_epochs: List[int],
    history_train_rmse: List[float],
    history_val: List[Dict[str, float]],
    save_path: Path,
) -> None:
    """Plot training and validation metrics across epochs and save to save_path."""
    if not history_epochs or not history_val:
        return
    n_epochs = len(history_epochs)
    val_price_rmse = [m["price_rmse"] for m in history_val]
    val_price_mae = [m["price_mae"] for m in history_val]
    val_direction_5m = [m["direction_5m_accuracy"] for m in history_val]
    val_volatility_10m = [m["volatility_10m_rmse"] for m in history_val]
    val_breakout_10m = [m["breakout_10m_accuracy"] for m in history_val]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Next-minute LSTM training curves", fontsize=12)

    axes[0, 0].plot(history_epochs, history_train_rmse, "b-o", markersize=4)
    axes[0, 0].set_title("Train RMSE (return)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history_epochs, val_price_rmse, "g-o", markersize=4)
    axes[0, 1].set_title("Validation price RMSE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history_epochs, val_price_mae, "g-s", markersize=4)
    axes[0, 2].set_title("Validation price MAE")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history_epochs, val_direction_5m, "m-o", markersize=4)
    axes[1, 0].set_title("Validation 5m direction accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history_epochs, val_volatility_10m, "c-o", markersize=4)
    axes[1, 1].set_title("Validation 10m volatility RMSE")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(history_epochs, val_breakout_10m, "orange", marker="o", markersize=4)
    axes[1, 2].set_title("Validation 10m breakout accuracy")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to: {save_path}")


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
    """
    Compute metrics for all tasks on a dataloader:
      - price_mae / price_rmse
      - direction_5m_accuracy (5m price direction: down/sideways/up)
      - volatility_10m_rmse (10m realized volatility)
      - breakout_10m_accuracy (10m ATR breakout)
    """
    model.eval()
    price_mae_sum = 0.0
    price_mse_sum = 0.0
    price_n = 0

    dir5_correct = 0
    dir5_total = 0

    vol10_mse_sum = 0.0
    vol10_n = 0

    brk_correct = 0
    brk_total = 0

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(DEVICE)
            tgt_price = batch["target_price"].to(DEVICE)
            tgt_dir5 = batch["target_dir5"].to(DEVICE)
            tgt_vol10 = batch["target_vol10"].to(DEVICE)
            tgt_brk = batch["target_breakout"].to(DEVICE)

            outputs = model(seq)

            # Price metrics
            pred_price = outputs["price"]
            diff_price = pred_price - tgt_price
            price_mae_sum += torch.sum(torch.abs(diff_price)).item()
            price_mse_sum += torch.sum(diff_price * diff_price).item()
            price_n += tgt_price.numel()

            # Direction next 5m accuracy
            logits = outputs["dir5_logits"]
            pred_dir5 = torch.argmax(logits, dim=1)
            matches = (pred_dir5 == tgt_dir5).float()
            dir5_correct += matches.sum().item()
            dir5_total += matches.numel()

            # Volatility next 10m RMSE
            pred_vol10 = outputs["vol10"]
            diff_vol = pred_vol10 - tgt_vol10
            vol10_mse_sum += torch.sum(diff_vol * diff_vol).item()
            vol10_n += tgt_vol10.numel()

            # Breakout accuracy (0/1 threshold at 0.5; model outputs logits)
            pred_brk_prob = torch.sigmoid(outputs["breakout"])
            pred_brk = (pred_brk_prob >= 0.5).float()
            brk_matches = (pred_brk == tgt_brk).float()
            brk_correct += brk_matches.sum().item()
            brk_total += brk_matches.numel()

    if price_n == 0:
        return {
            "price_mae": float("nan"),
            "price_rmse": float("nan"),
            "direction_5m_accuracy": float("nan"),
            "volatility_10m_rmse": float("nan"),
            "breakout_10m_accuracy": float("nan"),
            "samples": 0,
        }

    price_mae = price_mae_sum / price_n
    price_rmse = math.sqrt(price_mse_sum / price_n)
    vol10_rmse = math.sqrt(vol10_mse_sum / vol10_n) if vol10_n else float("nan")
    dir5_acc = dir5_correct / dir5_total if dir5_total else float("nan")
    brk_acc = brk_correct / brk_total if brk_total else float("nan")

    return {
        "price_mae": price_mae,
        "price_rmse": price_rmse,
        "direction_5m_accuracy": dir5_acc,
        "volatility_10m_rmse": vol10_rmse,
        "breakout_10m_accuracy": brk_acc,
        "samples": price_n,
    }


def main() -> None:
    print("Loading SessionMinuteBar data...")
    bars = _load_bars()
    if not bars:
        print("No SessionMinuteBar rows found. Run Databento ingestion first.")
        return
    print(f"Loaded {len(bars)} minute bars.")

    # Use cached dataset if enabled and valid; otherwise build and optionally save cache
    cached = None
    normalize_inputs = getattr(settings, "BAR_NORMALIZE_INPUTS", True)
    normalize_return = getattr(settings, "BAR_NORMALIZE_RETURN_TARGET", False)
    normalize_vol = getattr(settings, "BAR_NORMALIZE_VOL_TARGET", False)
    dir5_threshold = getattr(settings, "BAR_DIR5_THRESHOLD", 0.002)
    breakout_atr_k = getattr(settings, "BAR_BREAKOUT_ATR_K", 1.0)

    if USE_CACHE and not REBUILD_CACHE and CACHE_PATH.is_file():
        cached = _load_cached_sequences(
            CACHE_PATH,
            expected_lookback=LOOKBACK,
            expected_symbols=settings.SYMBOLS,
            num_bars_expected=len(bars),
            expected_target_type="return",
            expected_normalize_inputs=normalize_inputs,
            expected_normalize_return=normalize_return,
            expected_normalize_vol=normalize_vol,
            expected_dir5_threshold=dir5_threshold,
            expected_breakout_atr_k=breakout_atr_k,
        )
        if cached is not None:
            sequences, targets_price, targets_dir5, targets_vol10, targets_breakout = cached
            n = sequences.size(0)
            print(f"Using cached sequences from {CACHE_PATH} ({n} samples, built from {len(bars)} bars).")
        else:
            if CACHE_PATH.is_file():
                print("Cache exists but is stale or incompatible (lookback/symbols/bar count). Rebuilding...")
    atr_window = getattr(settings, "BAR_ATR_WINDOW", 14)
    loss_w_price = getattr(settings, "BAR_LOSS_WEIGHT_PRICE", 1.0)
    loss_w_dir5 = getattr(settings, "BAR_LOSS_WEIGHT_DIR5", 2.0)
    loss_w_vol = getattr(settings, "BAR_LOSS_WEIGHT_VOL", 0.5)
    loss_w_brk = getattr(settings, "BAR_LOSS_WEIGHT_BREAKOUT", 2.0)
    train_phase = getattr(settings, "BAR_TRAIN_PHASE", "all")
    if cached is None:
        print(f"Building sequences with LOOKBACK={LOOKBACK}...")
        (
            sequences,
            targets_price,
            targets_dir5,
            targets_vol10,
            targets_breakout,
        ) = _build_sequences(
            bars,
            LOOKBACK,
            normalize_inputs=normalize_inputs,
            normalize_return_target=normalize_return,
            normalize_vol_target=normalize_vol,
            dir5_threshold=dir5_threshold,
            breakout_atr_k=breakout_atr_k,
            atr_window=atr_window,
        )
        n = sequences.size(0)
        print(f"Built {n} sequences.")
        if USE_CACHE:
            min_sd = min(b.session_date for b in bars)
            max_sd = max(b.session_date for b in bars)
            _save_sequences_cache(
                CACHE_PATH,
                sequences,
                targets_price,
                targets_dir5,
                targets_vol10,
                targets_breakout,
                lookback=LOOKBACK,
                symbols=settings.SYMBOLS,
                num_bars=len(bars),
                min_session_date=str(min_sd),
                max_session_date=str(max_sd),
                target_type="return",
                normalize_inputs=normalize_inputs,
                normalize_return=normalize_return,
                normalize_vol=normalize_vol,
                dir5_threshold=dir5_threshold,
                breakout_atr_k=breakout_atr_k,
            )
            print(f"Saved dataset cache to {CACHE_PATH}.")

    train_idx, val_idx, test_idx = _time_split_indices(n, VAL_RATIO, TEST_RATIO)
    print(
        f"Split sizes -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
        f"(VAL_RATIO={VAL_RATIO}, TEST_RATIO={TEST_RATIO})"
    )

    train_dataset = NextMinuteBarDataset(
        sequences[train_idx],
        targets_price[train_idx],
        targets_dir5[train_idx],
        targets_vol10[train_idx],
        targets_breakout[train_idx],
    )
    val_dataset = (
        NextMinuteBarDataset(
            sequences[val_idx],
            targets_price[val_idx],
            targets_dir5[val_idx],
            targets_vol10[val_idx],
            targets_breakout[val_idx],
        )
        if len(val_idx)
        else None
    )
    test_dataset = (
        NextMinuteBarDataset(
            sequences[test_idx],
            targets_price[test_idx],
            targets_dir5[test_idx],
            targets_vol10[test_idx],
            targets_breakout[test_idx],
        )
        if len(test_idx)
        else None
    )

    # 5m direction class distribution (train) and balanced weights
    train_dir5 = targets_dir5[train_idx]
    if hasattr(train_dir5, "numpy"):
        train_dir5_np = train_dir5.numpy()
    else:
        train_dir5_np = np.asarray(train_dir5)
    dir5_counts = np.bincount(train_dir5_np.astype(int), minlength=3)
    dir5_total = int(dir5_counts.sum())
    print(
        f"5m direction class distribution (train): down={dir5_counts[0]}, sideways={dir5_counts[1]}, up={dir5_counts[2]} "
        f"(total={dir5_total})"
    )
    # Inverse frequency weights so minority classes get higher weight
    dir5_weights = np.ones(3, dtype="float32")
    for c in range(3):
        if dir5_counts[c] > 0:
            dir5_weights[c] = dir5_total / (3.0 * dir5_counts[c])
    dir5_weight_tensor = torch.from_numpy(dir5_weights).to(DEVICE)

    # Breakout class balance (train) and pos_weight for BCE
    train_brk = targets_breakout[train_idx]
    if hasattr(train_brk, "numpy"):
        train_brk_np = train_brk.numpy()
    else:
        train_brk_np = np.asarray(train_brk)
    brk_pos = int((train_brk_np >= 0.5).sum())
    brk_neg = int(train_brk_np.size - brk_pos)
    print(f"10m breakout class balance (train): pos={brk_pos}, neg={brk_neg} (pos_rate={brk_pos / max(1, train_brk_np.size):.3f})")
    brk_pos_weight = torch.tensor([brk_neg / max(1, brk_pos)], dtype=torch.float32, device=DEVICE) if brk_pos > 0 else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if test_dataset else None

    input_size = sequences.size(2)
    config = NextMinuteModelConfig(input_size=input_size)
    model = NextMinuteBarLSTM(config).to(DEVICE)

    if train_phase == "heads_only":
        for p in model.lstm.parameters():
            p.requires_grad = False
        for p in model.trunk.parameters():
            p.requires_grad = False
        for p in model.price_head.parameters():
            p.requires_grad = False
        for p in model.vol10_head.parameters():
            p.requires_grad = False
        print("Staged training: frozen trunk, price_head, vol10_head; training dir5 and breakout heads only.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_price = nn.MSELoss()
    criterion_dir5 = nn.CrossEntropyLoss(weight=dir5_weight_tensor)
    criterion_vol10 = nn.MSELoss()
    criterion_brk = (
        nn.BCEWithLogitsLoss(pos_weight=brk_pos_weight) if brk_pos_weight is not None else nn.BCEWithLogitsLoss()
    )

    print(
        f"Training NextMinuteBarLSTM on device={DEVICE} for {EPOCHS} epochs "
        f"(batch_size={BATCH_SIZE}, lr={LEARNING_RATE})"
    )

    best_val_rmse = float("inf")
    best_state = None
    epochs_without_improve = 0

    # Per-epoch history for training curves
    history_epochs: List[int] = []
    history_train_rmse: List[float] = []
    history_val: List[Dict[str, float]] = []

    num_batches = len(train_loader)
    total_steps = max(1, EPOCHS * num_batches)
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        print(f"Epoch {epoch + 1}/{EPOCHS} starting...")

        for batch_idx, batch in enumerate(train_loader, start=1):
            seq = batch["sequence"].to(DEVICE)
            tgt_price = batch["target_price"].to(DEVICE)
            tgt_dir5 = batch["target_dir5"].to(DEVICE)
            tgt_vol10 = batch["target_vol10"].to(DEVICE)
            tgt_brk = batch["target_breakout"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(seq)

            loss_price = criterion_price(outputs["price"], tgt_price)
            loss_dir5 = criterion_dir5(outputs["dir5_logits"], tgt_dir5)
            loss_vol10 = criterion_vol10(outputs["vol10"], tgt_vol10)
            loss_brk = criterion_brk(outputs["breakout"], tgt_brk)

            # Combined multi-task loss with configurable weights
            loss = (
                loss_w_price * loss_price
                + loss_w_dir5 * loss_dir5
                + loss_w_vol * loss_vol10
                + loss_w_brk * loss_brk
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * tgt_price.numel()
            n_seen += tgt_price.numel()

            # Global training progress across all epochs/batches
            global_step += 1
            if num_batches:
                if batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches:
                    pct = global_step / total_steps * 100.0
                    print(
                        f"Training progress: epoch {epoch + 1}/{EPOCHS}, "
                        f"batch {batch_idx}/{num_batches} "
                        f"({pct:.1f}% overall)",
                        end="\r",
                        flush=True,
                    )

        print()  # newline after epoch's batch-level progress

        train_mse = epoch_loss / max(1, n_seen)
        train_rmse = math.sqrt(train_mse)

        val_metrics = {
            "price_mae": float("nan"),
            "price_rmse": float("nan"),
            "direction_5m_accuracy": float("nan"),
            "volatility_10m_rmse": float("nan"),
            "breakout_10m_accuracy": float("nan"),
            "samples": 0,
        }
        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| train_rmse={train_rmse:.5f} "
            f"| val_price_rmse={val_metrics['price_rmse']:.5f} "
            f"| val_price_mae={val_metrics['price_mae']:.5f} "
            f"| val_direction_5m_acc={val_metrics['direction_5m_accuracy']:.4f} "
            f"| val_volatility_10m_rmse={val_metrics['volatility_10m_rmse']:.5f} "
            f"| val_breakout_10m_acc={val_metrics['breakout_10m_accuracy']:.4f}"
        )

        # Record history for plots
        history_epochs.append(epoch + 1)
        history_train_rmse.append(train_rmse)
        history_val.append(dict(val_metrics))

        # Track best model by validation price RMSE and early stopping
        val_rmse = val_metrics["price_rmse"]
        if not math.isnan(val_rmse) and (val_rmse + EARLY_STOP_MIN_DELTA) < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(no improvement in val return_rmse for {EARLY_STOP_PATIENCE} epochs; best={best_val_rmse:.5f})."
            )
            break

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

    if history_epochs and history_val:
        curves_path = models_dir / "training_curves.png"
        _plot_training_curves(history_epochs, history_train_rmse, history_val, curves_path)

    print(f"Saved next-minute model to: {ckpt_path}")
    print(f"Saved metrics JSON to: {metrics_path}")
    print("Validation metrics:", val_summary)
    print("Test metrics:", test_summary)


if __name__ == "__main__":
    main()


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
     early stopping and best checkpoint on validation direction metric (default: direction_5m_macro_f1).
     Reports return MAE/RMSE, direction_5m_accuracy, direction_5m_macro_f1, direction_5m_balanced_accuracy,
     volatility_10m_rmse, breakout_10m_accuracy on val and test.
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
rebuilding sequences after config or data changes (e.g. after changing BAR_DIR5_THRESHOLD).
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

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
BATCH_SIZE = getattr(settings, "BAR_BATCH_SIZE", 128)
EPOCHS = getattr(settings, "BAR_NUM_EPOCHS", 10)
LEARNING_RATE = getattr(settings, "BAR_LEARNING_RATE", settings.LEARNING_RATE)
VAL_RATIO = getattr(settings, "BAR_VAL_SPLIT", 0.1)
TEST_RATIO = getattr(settings, "BAR_TEST_SPLIT", 0.1)
USE_CACHE = getattr(settings, "BAR_CACHE_DATASET", True)
REBUILD_CACHE = getattr(settings, "BAR_REBUILD_CACHE", False)
CACHE_PATH = settings.MODELS_DIR / "next_minute_dataset.pt"
# Bump when bar feature set changes (e.g. 15m return, regime, microstructure) so old caches are invalidated.
FEATURE_VERSION = 4
EARLY_STOP_PATIENCE = getattr(settings, "BAR_EARLY_STOP_PATIENCE", 5)
EARLY_STOP_MIN_DELTA = getattr(settings, "BAR_EARLY_STOP_MIN_DELTA", 0.0)
EARLY_STOP_METRIC = getattr(settings, "BAR_EARLY_STOP_METRIC", "direction_5m_accuracy")
# For these metrics lower is better; we negate so "best" is still max.
EARLY_STOP_LOWER_IS_BETTER = frozenset({"price_rmse", "volatility_10m_rmse"})
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    """Return a compact ASCII progress bar."""
    if total <= 0:
        return "[" + ("-" * width) + "]"
    done = max(0, min(done, total))
    filled = int(round(width * (done / total)))
    return "[" + ("=" * filled) + ("-" * (width - filled)) + "]"


def _direction_predict_with_confidence(
    logits: torch.Tensor,
    confidence_threshold: float,
    num_classes: int = 3,
) -> torch.Tensor:
    """
    Predict direction. For 3-class (0=down, 1=sideways, 2=up): if confidence_threshold > 0 and
    max(prob) >= threshold and argmax is down or up, predict that; otherwise predict sideways.
    For 2-class (0=down, 1=up): always argmax. If threshold <= 0, use raw argmax.
    """
    if num_classes == 2 or confidence_threshold <= 0:
        return torch.argmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    max_prob, argmax = torch.max(probs, dim=1)
    # Predict down/up only when confident; else sideways (1)
    pred = torch.where(
        (max_prob >= confidence_threshold) & (argmax != 1),
        argmax,
        torch.full_like(argmax, 1, device=logits.device, dtype=argmax.dtype),
    )
    return pred


class _FocalLoss(nn.Module):
    """Multi-class focal loss: -alpha_c * (1 - p_t)^gamma * log(p_t). Reduces weight on easy examples."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], target: [B] long
        log_probs = nn.functional.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.weight is not None:
            w = self.weight[target]  # [B]
            loss = loss * w
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _macro_f1_and_balanced_accuracy(
    pred: np.ndarray,
    true: np.ndarray,
    num_classes: int = 3,
) -> Tuple[float, float, List[float], List[float], List[float]]:
    """Compute macro-F1, balanced accuracy, and per-class precision, recall, F1."""
    macro_f1 = 0.0
    balanced_acc = 0.0
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for c in range(num_classes):
        tp = ((pred == c) & (true == c)).sum()
        pred_c = (pred == c).sum()
        true_c = (true == c).sum()
        precision = tp / pred_c if pred_c > 0 else 0.0
        recall = tp / true_c if true_c > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        macro_f1 += f1
        balanced_acc += recall
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
    macro_f1 /= num_classes
    balanced_acc /= num_classes
    return float(macro_f1), float(balanced_acc), precisions, recalls, f1s


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
    expected_dir5_threshold_atr_k: Optional[float] = None,
    expected_dir5_min_band: Optional[float] = None,
    expected_dir5_two_class: Optional[bool] = None,
    expected_breakout_atr_k: Optional[float] = None,
    expected_feature_version: Optional[int] = None,
    expected_normalize_inputs_expanding: Optional[bool] = None,
    expected_max_abs_ret_1m: Optional[float] = None,
    expected_max_abs_ret_5m: Optional[float] = None,
) -> Optional[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[Tuple[str, str]]],
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
    if expected_dir5_threshold_atr_k is not None:
        expected_meta["dir5_threshold_atr_k"] = expected_dir5_threshold_atr_k
    if expected_dir5_min_band is not None:
        expected_meta["dir5_min_band"] = expected_dir5_min_band
    if expected_dir5_two_class is not None:
        expected_meta["dir5_two_class"] = expected_dir5_two_class
    if expected_breakout_atr_k is not None:
        expected_meta["breakout_atr_k"] = expected_breakout_atr_k
    if expected_feature_version is not None:
        expected_meta["feature_version"] = expected_feature_version
    if expected_normalize_inputs_expanding is not None:
        expected_meta["normalize_inputs_expanding"] = expected_normalize_inputs_expanding
    if expected_max_abs_ret_1m is not None:
        expected_meta["max_abs_ret_1m"] = expected_max_abs_ret_1m
    if expected_max_abs_ret_5m is not None:
        expected_meta["max_abs_ret_5m"] = expected_max_abs_ret_5m
    if not _cache_metadata_matches(data, expected_meta):
        return None
    if num_bars_expected is not None and data.get("num_bars") != num_bars_expected:
        return None
    n = data["sequences"].size(0)
    if data["targets_price"].size(0) != n:
        return None
    session_id_per_sample = data.get("session_id_per_sample")
    sessions = data.get("sessions")
    return (
        data["sequences"],
        data["targets_price"],
        data["targets_dir5"],
        data["targets_vol10"],
        data["targets_breakout"],
        session_id_per_sample,
        sessions,
    )


def _save_sequences_cache(
    cache_path: Path,
    sequences: torch.Tensor,
    targets_price: torch.Tensor,
    targets_dir5: torch.Tensor,
    targets_vol10: torch.Tensor,
    targets_breakout: torch.Tensor,
    session_id_per_sample: torch.Tensor,
    sessions: List[Tuple[str, str]],
    lookback: int,
    symbols: List[str],
    num_bars: int,
    min_session_date: str,
    max_session_date: str,
    target_type: str = "return",
    normalize_inputs: bool = False,
    normalize_return: bool = False,
    normalize_vol: bool = False,
    normalize_inputs_expanding: bool = False,
    dir5_threshold: float = 0.002,
    dir5_threshold_atr_k: float = 0.0,
    dir5_min_band: float = 0.0001,
    dir5_two_class: bool = False,
    breakout_atr_k: float = 1.0,
    feature_version: int = FEATURE_VERSION,
    max_abs_ret_1m: float = 0.10,
    max_abs_ret_5m: float = 0.20,
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
            "session_id_per_sample": session_id_per_sample,
            "sessions": sessions,
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
            "normalize_inputs_expanding": normalize_inputs_expanding,
            "dir5_threshold": dir5_threshold,
            "dir5_threshold_atr_k": dir5_threshold_atr_k,
            "dir5_min_band": dir5_min_band,
            "dir5_two_class": dir5_two_class,
            "breakout_atr_k": breakout_atr_k,
            "feature_version": feature_version,
            "max_abs_ret_1m": max_abs_ret_1m,
            "max_abs_ret_5m": max_abs_ret_5m,
        },
        cache_path,
    )


def _build_sequences(
    bars: List[SessionMinuteBar],
    lookback: int,
    normalize_inputs: bool = True,
    normalize_return_target: bool = False,
    normalize_vol_target: bool = False,
    normalize_inputs_expanding: bool = False,
    dir5_threshold: float = 0.002,
    dir5_threshold_atr_k: float = 0.0,
    dir5_min_band: float = 0.0001,
    dir5_two_class: bool = False,
    breakout_atr_k: float = 1.0,
    atr_window: int = 14,
    max_abs_ret_1m: float = 0.10,
    max_abs_ret_5m: float = 0.20,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[Tuple[str, str]],
]:
    """
    Build sliding windows of enriched bar features and multi-horizon targets.

    Returns:
        sequences:        [N, T, F] float32  (F >= 5: OHLCV + derived; optionally per-session normalized)
        targets_price:    [N]       float32  (next-bar 1m return, optionally z-scored per session)
        targets_dir5:     [N]       int64    (direction next 5m: 0=down,1=sideways,2=up or 2-class 0/1)
        targets_vol10:    [N]       float32  (realized vol next 10m; optionally z-scored per session)
        targets_breakout: [N]       float32  (0/1 breakout next 10m, vol-relative)
        session_id_per_sample: [N] int64 (session index for each sample; for session-based split)
        sessions: list of (symbol, session_date) in chronological order
    """
    by_key: Dict[Tuple[str, str], List[SessionMinuteBar]] = defaultdict(list)
    for b in bars:
        by_key[(b.symbol, b.session_date)].append(b)

    sequences: List[torch.Tensor] = []
    targets_price: List[float] = []
    targets_dir5: List[int] = []
    targets_vol10: List[float] = []
    targets_breakout: List[float] = []
    session_ids: List[int] = []

    items = sorted(by_key.items(), key=lambda x: x[0][1])
    total_sessions = len(items)
    sessions_list: List[Tuple[str, str]] = [key for key, _ in items]
    if total_sessions:
        print(f"Building sequences across {total_sessions} sessions...")

    for session_idx, ((symbol, session_date), group) in enumerate(items):
        if len(group) <= lookback:
            continue

        # Compute feature matrix for this session (centralized implementation)
        all_feat, closes, highs, lows = build_session_feature_matrix(group)
        n_bars = all_feat.shape[0]

        # Per-session input normalization (full-session or expanding-window to avoid lookahead)
        if normalize_inputs and n_bars > 0:
            eps = 1e-8
            if normalize_inputs_expanding:
                for j in range(n_bars):
                    mean_j = all_feat[: j + 1].mean(axis=0)
                    std_j = all_feat[: j + 1].std(axis=0)
                    all_feat[j] = (all_feat[j] - mean_j) / (std_j + eps)
            else:
                feat_mean = all_feat.mean(axis=0)
                feat_std = all_feat.std(axis=0)
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
            # Skip windows containing non-finite values from noisy upstream bars/features.
            if not np.all(np.isfinite(window)):
                continue

            # 1) Next-bar return target (1m forward return from close at i-1 to close at i)
            prev_close = closes[i - 1]
            curr_close = closes[i]
            if prev_close > 0:
                return_1m = (curr_close - prev_close) / prev_close
            else:
                return_1m = 0.0
            if abs(float(session_returns_1m[i])) > max_abs_ret_1m:
                continue
            if normalize_return_target:
                return_1m = (return_1m - ret_mean) / ret_std

            # 2) Direction next 5m based on 5-bar ahead close (optional volatility-adjusted band)
            idx_5 = i + 5
            if idx_5 >= n:
                continue
            c_now = closes[i]
            c_5 = closes[idx_5]
            if c_now > 0:
                ret_5 = (c_5 - c_now) / c_now
            else:
                ret_5 = 0.0
            if dir5_threshold_atr_k > 0 and c_now > 0 and atr[i] >= 0:
                atr_pct = float(atr[i]) / c_now
                band = max(dir5_min_band, dir5_threshold_atr_k * atr_pct)
            else:
                band = dir5_threshold
            if ret_5 > band:
                dir_class = 2  # up
            elif ret_5 < -band:
                dir_class = 0  # down
            else:
                dir_class = 1  # sideways
            if dir5_two_class and dir_class == 1:
                continue  # drop sideways for 2-class; only keep down (0) and up (2 -> remap to 1)
            if abs(float(ret_5)) > max_abs_ret_5m:
                continue

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
            # For 2-class: 0=down, 1=up (remap original 2 -> 1)
            targets_dir5.append(1 if dir_class == 2 else 0 if dir5_two_class else dir_class)
            targets_vol10.append(vol10)
            targets_breakout.append(breakout)
            session_ids.append(session_idx)

        # Lightweight progress indicator for sequence-building phase
        if total_sessions and (session_idx % max(1, total_sessions // 20) == 0 or session_idx == total_sessions - 1):
            pct = (session_idx + 1) / total_sessions * 100.0
            bar = _progress_bar(session_idx + 1, total_sessions)
            print(
                f"Sequence build {bar} {pct:5.1f}%  ({session_idx + 1}/{total_sessions} sessions)",
                end="\r",
                flush=True,
            )

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
    session_id_arr = np.asarray(session_ids, dtype="int64")

    return (
        torch.from_numpy(seq_arr),
        torch.from_numpy(tgt_price_arr),
        torch.from_numpy(tgt_dir5_arr),
        torch.from_numpy(tgt_vol10_arr),
        torch.from_numpy(tgt_brk_arr),
        torch.from_numpy(session_id_arr),
        sessions_list,
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
    val_price_rmse = [m["price_rmse"] for m in history_val]
    val_price_mae = [m["price_mae"] for m in history_val]
    val_direction_5m = [m["direction_5m_accuracy"] for m in history_val]
    val_direction_5m_macro_f1 = [m.get("direction_5m_macro_f1", float("nan")) for m in history_val]
    val_volatility_10m = [m["volatility_10m_rmse"] for m in history_val]
    val_breakout_10m = [m["breakout_10m_accuracy"] for m in history_val]

    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Next-minute LSTM — training curves", fontsize=14, fontweight="bold", y=1.02)

    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    axes[0, 0].plot(history_epochs, history_train_rmse, "b-o", markersize=5, linewidth=1.5)
    axes[0, 0].set_title("Train RMSE (return)", fontsize=11)

    axes[0, 1].plot(history_epochs, val_price_rmse, "g-o", markersize=5, linewidth=1.5)
    axes[0, 1].set_title("Validation price RMSE", fontsize=11)

    axes[0, 2].plot(history_epochs, val_price_mae, "g-s", markersize=5, linewidth=1.5)
    axes[0, 2].set_title("Validation price MAE", fontsize=11)

    axes[1, 0].plot(history_epochs, val_direction_5m, "m-o", markersize=5, linewidth=1.5, label="accuracy")
    axes[1, 0].plot(history_epochs, val_direction_5m_macro_f1, "c-s", markersize=5, linewidth=1.5, label="macro-F1")
    axes[1, 0].set_title("Validation 5m direction (accuracy + macro-F1)", fontsize=11)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(loc="lower right", fontsize=8)

    axes[1, 1].plot(history_epochs, val_volatility_10m, "c-o", markersize=5, linewidth=1.5)
    axes[1, 1].set_title("Validation 10m volatility RMSE", fontsize=11)

    axes[1, 2].plot(history_epochs, val_breakout_10m, color="darkorange", marker="o", markersize=5, linewidth=1.5)
    axes[1, 2].set_title("Validation 10m breakout accuracy", fontsize=11)
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved training curves to: {save_path}")


def _plot_walk_forward_summary(
    fold_results: List[Dict],
    save_path: Path,
    metric_key: str = "direction_5m_accuracy",
) -> None:
    """Plot per-fold test metrics (e.g. direction accuracy) and summary table for walk-forward."""
    if not fold_results:
        return
    n_folds = len(fold_results)
    fold_labels = [f"Fold {r['fold_idx']+1}\n{r.get('test_start', '')[:7]}" for r in fold_results]
    accs = [r.get("test_summary", {}).get(metric_key, float("nan")) for r in fold_results]
    accs = [a if not math.isnan(a) else 0.0 for a in accs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Walk-forward: per-fold test metrics", fontsize=12, fontweight="bold")
    axes[0].bar(range(n_folds), accs, color="steelblue", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(range(n_folds))
    axes[0].set_xticklabels(fold_labels, fontsize=8, rotation=15)
    axes[0].set_ylabel(metric_key)
    axes[0].set_title(f"Test {metric_key} per fold")
    axes[0].set_ylim(0, 1.0)
    axes[0].axhline(y=0.8, color="gray", linestyle="--", alpha=0.7, label="80% target")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Table: fold | train_end | test_start | test_end | direction_acc | ...
    table_data = []
    for r in fold_results:
        ts = r.get("test_summary") or {}
        table_data.append([
            str(r.get("fold_idx", "") + 1),
            r.get("train_end", "")[:10],
            r.get("test_start", "")[:10],
            r.get("test_end", "")[:10],
            f"{ts.get('direction_5m_accuracy', float('nan')):.3f}",
            f"{ts.get('direction_5m_macro_f1', float('nan')):.3f}",
            f"{ts.get('breakout_10m_accuracy', float('nan')):.3f}",
        ])
    axes[1].axis("off")
    table = axes[1].table(
        cellText=table_data,
        colLabels=["Fold", "Train end", "Test start", "Test end", "Dir acc", "Macro F1", "Breakout acc"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    axes[1].set_title("Per-fold test summary")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved walk-forward summary to: {save_path}")


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


def _nan_to_none(obj):
    """Recursively replace float nan with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _parse_session_date(s: str):
    """Parse session_date string YYYY-MM-DD to date."""
    return datetime.strptime(s.strip()[:10], "%Y-%m-%d").date()


def _walk_forward_fold_ranges(
    sessions: List[Tuple[str, str]],
    train_days: int,
    test_days: int,
    slide_days: int,
) -> List[Tuple[Set[int], Set[int], str, str, str, str]]:
    """
    Return list of (train_session_ids, test_session_ids, train_start, train_end, test_start, test_end) per fold.
    Sessions are (symbol, session_date_str) in chronological order; session index i = sessions[i].
    """
    if not sessions or train_days <= 0 or test_days <= 0 or slide_days <= 0:
        return []
    dates = [_parse_session_date(s[1]) for s in sessions]
    min_date = min(dates)
    max_date = max(dates)
    folds: List[Tuple[Set[int], Set[int], str, str, str, str]] = []
    f = 0
    while True:
        train_start = min_date + timedelta(days=f * slide_days)
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        if test_end > max_date:
            break
        train_sids = {i for i in range(len(sessions)) if train_start <= dates[i] < train_end}
        test_sids = {i for i in range(len(sessions)) if test_start <= dates[i] < test_end}
        if not train_sids or not test_sids:
            f += 1
            continue
        folds.append(
            (
                train_sids,
                test_sids,
                train_start.isoformat(),
                train_end.isoformat(),
                test_start.isoformat(),
                test_end.isoformat(),
            )
        )
        f += 1
    return folds


def _wf_split_indices(
    n: int,
    session_id_per_sample: torch.Tensor,
    train_session_ids: Set[int],
    test_session_ids: Set[int],
    val_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Return train_idx, val_idx, test_idx for one WF fold. Val = last val_ratio of train sessions."""
    sid = session_id_per_sample.detach().cpu().numpy()
    ordered_train = sorted(train_session_ids)
    n_val_sess = max(0, int(round(val_ratio * len(ordered_train))))
    val_sids = set(ordered_train[-n_val_sess:]) if n_val_sess else set()
    train_only_sids = set(ordered_train[: len(ordered_train) - n_val_sess]) if n_val_sess else set(ordered_train)
    train_idx = [i for i in range(n) if sid[i] in train_only_sids]
    val_idx = [i for i in range(n) if sid[i] in val_sids]
    test_idx = [i for i in range(n) if sid[i] in test_session_ids]
    return train_idx, val_idx, test_idx


def _session_split_indices(
    n: int,
    session_id_per_sample: torch.Tensor,
    sessions: List[Tuple[str, str]],
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Return (train_idx, val_idx, test_idx) by assigning whole sessions to splits (no leakage)."""
    num_sessions = len(sessions)
    if num_sessions == 0:
        return list(range(n)), [], []
    # Session order is already chronological (sorted by session_date); split by session index
    n_train_sess = max(1, int(round((1.0 - val_ratio - test_ratio) * num_sessions)))
    n_val_sess = max(0, int(round(val_ratio * num_sessions)))
    n_test_sess = num_sessions - n_train_sess - n_val_sess
    if n_test_sess < 0:
        n_test_sess = 0
        n_val_sess = num_sessions - n_train_sess
    train_session_ids = set(range(n_train_sess))
    val_session_ids = set(range(n_train_sess, n_train_sess + n_val_sess))
    test_session_ids = set(range(n_train_sess + n_val_sess, num_sessions))
    sid = session_id_per_sample.detach().cpu().numpy()
    train_idx = [i for i in range(n) if sid[i] in train_session_ids]
    val_idx = [i for i in range(n) if sid[i] in val_session_ids]
    test_idx = [i for i in range(n) if sid[i] in test_session_ids]
    return train_idx, val_idx, test_idx


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    direction_confidence_threshold: float = 0.0,
    num_dir_classes: int = 3,
) -> Dict[str, float]:
    """
    Compute metrics for all tasks on a dataloader:
      - price_mae / price_rmse
      - direction_5m_accuracy, direction_5m_macro_f1, direction_5m_balanced_accuracy
      - volatility_10m_rmse (10m realized volatility)
      - breakout_10m_accuracy (10m ATR breakout)
    Direction predictions use confidence threshold when direction_confidence_threshold > 0 (3-class only).
    num_dir_classes: 2 for binary up/down, 3 for down/sideways/up.
    """
    model.eval()
    price_mae_sum = 0.0
    price_mse_sum = 0.0
    price_n = 0

    dir5_pred_list: List[torch.Tensor] = []
    dir5_true_list: List[torch.Tensor] = []

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

            # Direction next 5m: use confidence threshold then accumulate for macro-F1
            logits = outputs["dir5_logits"]
            pred_dir5 = _direction_predict_with_confidence(
                logits, direction_confidence_threshold, num_classes=num_dir_classes
            )
            dir5_pred_list.append(pred_dir5)
            dir5_true_list.append(tgt_dir5)

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
        nc = num_dir_classes
        return {
            "price_mae": float("nan"),
            "price_rmse": float("nan"),
            "direction_5m_accuracy": float("nan"),
            "direction_5m_macro_f1": float("nan"),
            "direction_5m_balanced_accuracy": float("nan"),
            "direction_5m_precision_per_class": [float("nan")] * nc,
            "direction_5m_recall_per_class": [float("nan")] * nc,
            "direction_5m_f1_per_class": [float("nan")] * nc,
            "volatility_10m_rmse": float("nan"),
            "breakout_10m_accuracy": float("nan"),
            "samples": 0,
        }

    price_mae = price_mae_sum / price_n
    price_rmse = math.sqrt(price_mse_sum / price_n)
    vol10_rmse = math.sqrt(vol10_mse_sum / vol10_n) if vol10_n else float("nan")
    brk_acc = brk_correct / brk_total if brk_total else float("nan")

    # Direction: macro-F1 and balanced accuracy from accumulated preds/targets
    all_pred = torch.cat(dir5_pred_list, dim=0).cpu().numpy()
    all_true = torch.cat(dir5_true_list, dim=0).cpu().numpy()
    dir5_acc = float((all_pred == all_true).mean())
    macro_f1, balanced_acc, prec_per_class, rec_per_class, f1_per_class = _macro_f1_and_balanced_accuracy(
        all_pred, all_true, num_classes=num_dir_classes
    )

    return {
        "price_mae": price_mae,
        "price_rmse": price_rmse,
        "direction_5m_accuracy": dir5_acc,
        "direction_5m_macro_f1": macro_f1,
        "direction_5m_balanced_accuracy": balanced_acc,
        "direction_5m_precision_per_class": prec_per_class,
        "direction_5m_recall_per_class": rec_per_class,
        "direction_5m_f1_per_class": f1_per_class,
        "volatility_10m_rmse": vol10_rmse,
        "breakout_10m_accuracy": brk_acc,
        "samples": price_n,
    }


def _run_one_fold(
    sequences: torch.Tensor,
    targets_price: torch.Tensor,
    targets_dir5: torch.Tensor,
    targets_vol10: torch.Tensor,
    targets_breakout: torch.Tensor,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    num_dir_classes: int,
    resume_state: Optional[dict] = None,
    fold_label: str = "",
) -> Tuple[
    Optional[dict],
    Optional[Dict[str, float]],
    Optional[Dict[str, float]],
    List[int],
    List[float],
    List[Dict[str, float]],
    float,
    Optional[float],
]:
    """Train one fold; return (best_state, val_summary, test_summary, history_epochs, history_train_rmse, history_val, direction_confidence_threshold, best_val_metric_value)."""
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
    train_dir5 = targets_dir5[train_idx]
    train_dir5_np = train_dir5.numpy() if hasattr(train_dir5, "numpy") else np.asarray(train_dir5)
    dir5_counts = np.bincount(train_dir5_np.astype(int), minlength=num_dir_classes)
    dir5_total = int(dir5_counts.sum())
    dir5_weights = np.ones(num_dir_classes, dtype="float32")
    for c in range(num_dir_classes):
        if dir5_counts[c] > 0:
            dir5_weights[c] = dir5_total / (float(num_dir_classes) * dir5_counts[c])
    minority_scale = getattr(settings, "BAR_DIR5_MINORITY_WEIGHT_SCALE", 1.0)
    if minority_scale != 1.0:
        dir5_weights[0] *= minority_scale
        dir5_weights[num_dir_classes - 1] *= minority_scale
    dir5_weight_tensor = torch.from_numpy(dir5_weights).to(DEVICE)
    oversample_minority = getattr(settings, "BAR_DIR5_OVERSAMPLE_MINORITY", False)
    if oversample_minority:
        weights_per_class = 1.0 / (dir5_counts.astype(np.float64) + 1e-8)
        sample_weights = weights_per_class[train_dir5_np.astype(int)]
        train_sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights.astype(np.float32)), num_samples=len(sample_weights)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_brk = targets_breakout[train_idx]
    train_brk_np = train_brk.numpy() if hasattr(train_brk, "numpy") else np.asarray(train_brk)
    brk_pos = int((train_brk_np >= 0.5).sum())
    brk_neg = int(train_brk_np.size - brk_pos)
    brk_pos_weight = torch.tensor([brk_neg / max(1, brk_pos)], dtype=torch.float32, device=DEVICE) if brk_pos > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if test_dataset else None
    input_size = sequences.size(2)
    config = NextMinuteModelConfig(input_size=input_size, num_dir_classes=num_dir_classes)
    model = NextMinuteBarLSTM(config).to(DEVICE)
    if resume_state is not None:
        model.load_state_dict(resume_state, strict=True)
    loss_w_price = getattr(settings, "BAR_LOSS_WEIGHT_PRICE", 1.0)
    loss_w_dir5 = getattr(settings, "BAR_LOSS_WEIGHT_DIR5", 2.0)
    loss_w_vol = getattr(settings, "BAR_LOSS_WEIGHT_VOL", 0.5)
    loss_w_brk = getattr(settings, "BAR_LOSS_WEIGHT_BREAKOUT", 2.0)
    train_phase = getattr(settings, "BAR_TRAIN_PHASE", "all")
    dir5_first_epochs = getattr(settings, "BAR_DIR5_FIRST_EPOCHS", 3)
    if train_phase == "direction_first":
        for p in model.lstm.parameters():
            p.requires_grad = False
        for p in model.trunk.parameters():
            p.requires_grad = False
        for p in model.price_head.parameters():
            p.requires_grad = False
        for p in model.vol10_head.parameters():
            p.requires_grad = False
        for p in model.breakout_head.parameters():
            p.requires_grad = False
    dir5_label_smoothing = getattr(settings, "BAR_DIR5_LABEL_SMOOTHING", 0.0)
    dir5_use_focal = getattr(settings, "BAR_DIR5_USE_FOCAL", False)
    dir5_focal_gamma = getattr(settings, "BAR_DIR5_FOCAL_GAMMA", 2.0)
    if dir5_use_focal:
        criterion_dir5 = _FocalLoss(weight=dir5_weight_tensor, gamma=dir5_focal_gamma).to(DEVICE)
    else:
        criterion_dir5 = nn.CrossEntropyLoss(weight=dir5_weight_tensor, label_smoothing=dir5_label_smoothing)
    if train_phase == "direction_first":
        optimizer = optim.Adam([p for p in model.dir5_head.parameters() if p.requires_grad], lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_price = nn.MSELoss()
    criterion_vol10 = nn.MSELoss()
    criterion_brk = nn.BCEWithLogitsLoss(pos_weight=brk_pos_weight) if brk_pos_weight is not None else nn.BCEWithLogitsLoss()
    direction_confidence_threshold = getattr(settings, "BAR_DIR5_CONFIDENCE_THRESHOLD", 0.0)
    best_val_metric = -math.inf
    best_val_metric_value = None
    best_state = None
    epochs_without_improve = 0
    history_epochs = []
    history_train_rmse = []
    history_val = []
    num_batches = len(train_loader)
    total_steps = max(1, EPOCHS * num_batches)
    global_step = 0
    for epoch in range(EPOCHS):
        if train_phase == "direction_first" and epoch == dir5_first_epochs:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        direction_only = train_phase == "direction_first" and epoch < dir5_first_epochs
        for batch_idx, batch in enumerate(train_loader, start=1):
            seq = batch["sequence"].to(DEVICE)
            tgt_price = batch["target_price"].to(DEVICE)
            tgt_dir5 = batch["target_dir5"].to(DEVICE)
            tgt_vol10 = batch["target_vol10"].to(DEVICE)
            tgt_brk = batch["target_breakout"].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seq)
            loss_dir5 = criterion_dir5(outputs["dir5_logits"], tgt_dir5)
            if direction_only:
                loss = loss_w_dir5 * loss_dir5
            else:
                loss_price = criterion_price(outputs["price"], tgt_price)
                loss_vol10 = criterion_vol10(outputs["vol10"], tgt_vol10)
                loss_brk = criterion_brk(outputs["breakout"], tgt_brk)
                loss = loss_w_price * loss_price + loss_w_dir5 * loss_dir5 + loss_w_vol * loss_vol10 + loss_w_brk * loss_brk
            loss.backward()
            if getattr(settings, "BAR_GRADIENT_CLIP_NORM", 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.BAR_GRADIENT_CLIP_NORM)
            optimizer.step()
            epoch_loss += loss.item() * tgt_price.numel()
            n_seen += tgt_price.numel()
            global_step += 1
            # Lightweight progress bar (print ~10x per epoch)
            if num_batches:
                if batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches:
                    pct = global_step / total_steps * 100.0
                    bar = _progress_bar(global_step, total_steps)
                    print(
                        f"Train {bar} {pct:5.1f}%  (epoch {epoch + 1}/{EPOCHS}, batch {batch_idx}/{num_batches})",
                        end="\r",
                        flush=True,
                    )
        if num_batches:
            print()  # newline after epoch progress bar
        train_rmse = math.sqrt(epoch_loss / max(1, n_seen))
        val_metrics = _evaluate(model, val_loader, direction_confidence_threshold, num_dir_classes=num_dir_classes) if val_loader is not None else {"direction_5m_accuracy": float("nan")}
        history_epochs.append(epoch + 1)
        history_train_rmse.append(train_rmse)
        history_val.append(dict(val_metrics))
        if val_loader is None:
            raw_val = -train_rmse  # use negated train RMSE so lower is better
            current_val_compare = -train_rmse
        else:
            raw_val = val_metrics.get(EARLY_STOP_METRIC)
            if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
                current_val_compare = -math.inf
            elif EARLY_STOP_METRIC in EARLY_STOP_LOWER_IS_BETTER:
                current_val_compare = -float(raw_val)
            else:
                current_val_compare = float(raw_val)
        if current_val_compare > best_val_metric:
            best_val_metric = current_val_compare
            best_val_metric_value = raw_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()}, strict=True)
    tune_threshold = getattr(settings, "BAR_DIR5_TUNE_THRESHOLD", False)
    if tune_threshold and val_loader is not None:
        best_f1 = -1.0
        best_th = 0.0
        for th in [0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            m = _evaluate(model, val_loader, direction_confidence_threshold=th, num_dir_classes=num_dir_classes)
            f1 = m.get("direction_5m_macro_f1", -1.0)
            if not math.isnan(f1) and f1 > best_f1:
                best_f1 = f1
                best_th = th
        direction_confidence_threshold = best_th
    val_summary = _evaluate(model, val_loader, direction_confidence_threshold, num_dir_classes=num_dir_classes) if val_loader else None
    test_summary = _evaluate(model, test_loader, direction_confidence_threshold, num_dir_classes=num_dir_classes) if test_loader else None
    return (best_state, val_summary, test_summary, history_epochs, history_train_rmse, history_val, direction_confidence_threshold, best_val_metric_value)


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
    normalize_inputs_expanding = getattr(settings, "BAR_NORMALIZE_INPUTS_EXPANDING", False)
    dir5_threshold = getattr(settings, "BAR_DIR5_THRESHOLD", 0.001)
    dir5_threshold_atr_k = getattr(settings, "BAR_DIR5_THRESHOLD_ATR_K", 0.0)
    dir5_min_band = getattr(settings, "BAR_DIR5_MIN_BAND", 0.0001)
    dir5_two_class = getattr(settings, "BAR_DIR5_TWO_CLASS", False)
    breakout_atr_k = getattr(settings, "BAR_BREAKOUT_ATR_K", 1.0)
    max_abs_ret_1m = getattr(settings, "BAR_MAX_ABS_RET_1M", 0.10)
    max_abs_ret_5m = getattr(settings, "BAR_MAX_ABS_RET_5M", 0.20)
    num_dir_classes = 2 if dir5_two_class else 3

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
            expected_dir5_threshold_atr_k=dir5_threshold_atr_k,
            expected_dir5_min_band=dir5_min_band,
            expected_dir5_two_class=dir5_two_class,
            expected_breakout_atr_k=breakout_atr_k,
            expected_feature_version=FEATURE_VERSION,
            expected_normalize_inputs_expanding=normalize_inputs_expanding,
            expected_max_abs_ret_1m=max_abs_ret_1m,
            expected_max_abs_ret_5m=max_abs_ret_5m,
        )
        if cached is not None:
            (
                sequences,
                targets_price,
                targets_dir5,
                targets_vol10,
                targets_breakout,
                session_id_per_sample,
                sessions,
            ) = cached
            n = sequences.size(0)
            print(f"Using cached sequences from {CACHE_PATH} ({n} samples, built from {len(bars)} bars).")
        else:
            session_id_per_sample = None
            sessions = None
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
            session_id_per_sample,
            sessions,
        ) = _build_sequences(
            bars,
            LOOKBACK,
            normalize_inputs=normalize_inputs,
            normalize_return_target=normalize_return,
            normalize_vol_target=normalize_vol,
            normalize_inputs_expanding=normalize_inputs_expanding,
            dir5_threshold=dir5_threshold,
            dir5_threshold_atr_k=dir5_threshold_atr_k,
            dir5_min_band=dir5_min_band,
            dir5_two_class=dir5_two_class,
            breakout_atr_k=breakout_atr_k,
            atr_window=atr_window,
            max_abs_ret_1m=max_abs_ret_1m,
            max_abs_ret_5m=max_abs_ret_5m,
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
                session_id_per_sample,
                sessions,
                lookback=LOOKBACK,
                symbols=settings.SYMBOLS,
                num_bars=len(bars),
                min_session_date=str(min_sd),
                max_session_date=str(max_sd),
                target_type="return",
                normalize_inputs=normalize_inputs,
                normalize_return=normalize_return,
                normalize_vol=normalize_vol,
                normalize_inputs_expanding=normalize_inputs_expanding,
                dir5_threshold=dir5_threshold,
                dir5_threshold_atr_k=dir5_threshold_atr_k,
                dir5_min_band=dir5_min_band,
                dir5_two_class=dir5_two_class,
                breakout_atr_k=breakout_atr_k,
                feature_version=FEATURE_VERSION,
                max_abs_ret_1m=max_abs_ret_1m,
                max_abs_ret_5m=max_abs_ret_5m,
            )
            print(f"Saved dataset cache to {CACHE_PATH}.")

    walk_forward = getattr(settings, "BAR_WALK_FORWARD", False)
    wf_train_days = getattr(settings, "BAR_WF_TRAIN_DAYS", 365)
    wf_test_days = getattr(settings, "BAR_WF_TEST_DAYS", 90)
    wf_slide_days = getattr(settings, "BAR_WF_SLIDE_DAYS", 90)
    wf_val_ratio = getattr(settings, "BAR_WF_VAL_RATIO", 0.1)

    if walk_forward and session_id_per_sample is not None and sessions is not None:
        folds = _walk_forward_fold_ranges(sessions, wf_train_days, wf_test_days, wf_slide_days)
        if not folds:
            print("Walk-forward: no folds generated (need enough calendar range). Falling back to single split.")
            walk_forward = False
        else:
            print(
                f"Walk-forward: {len(folds)} folds (train={wf_train_days}d, test={wf_test_days}d, slide={wf_slide_days}d)"
            )

    if walk_forward and session_id_per_sample is not None and sessions is not None and folds:
        # --- Walk-forward: train one fold at a time, keep best by test direction accuracy ---
        models_dir = settings.MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        wf_results: List[Dict] = []
        for fold_idx, (train_sids, test_sids, train_start, train_end, test_start, test_end) in enumerate(folds):
            print()
            print("=" * 60)
            print(f"  WF Fold {fold_idx + 1}/{len(folds)}  train {train_start} -> {train_end}  test {test_start} -> {test_end}")
            print("=" * 60)
            train_idx, val_idx, test_idx = _wf_split_indices(
                n, session_id_per_sample, train_sids, test_sids, wf_val_ratio
            )
            if len(train_idx) < BATCH_SIZE or len(test_idx) == 0:
                print(f"Skipping fold {fold_idx + 1}: too few train or test samples.")
                continue
            # Single-fold training (no resume)
            best_state, val_summary, test_summary, history_epochs, history_train_rmse, history_val, dct, _ = _run_one_fold(
                sequences,
                targets_price,
                targets_dir5,
                targets_vol10,
                targets_breakout,
                train_idx,
                val_idx,
                test_idx,
                num_dir_classes,
                resume_state=None,
                fold_label=f"Fold {fold_idx+1}/{len(folds)}",
            )
            fold_ckpt = models_dir / f"wf_fold_{fold_idx}.pt"
            if best_state is not None:
                torch.save(best_state, fold_ckpt)
            wf_results.append({
                "fold_idx": fold_idx,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "val_summary": val_summary,
                "test_summary": test_summary,
                "history_epochs": history_epochs,
                "history_train_rmse": history_train_rmse,
                "history_val": history_val,
                "direction_confidence_threshold": dct,
            })
        if not wf_results:
            print("No walk-forward folds completed. Exiting.")
            return
        best_fold_idx = max(
            range(len(wf_results)),
            key=lambda i: (wf_results[i].get("test_summary") or {}).get("direction_5m_accuracy", -1.0)
            if wf_results[i].get("test_summary") else -1.0,
        )
        print()
        print(f"Best fold by test direction_5m_accuracy: Fold {best_fold_idx + 1} (of {len(wf_results)})")
        fold_ckpt = models_dir / f"wf_fold_{best_fold_idx}.pt"
        if fold_ckpt.is_file():
            state = torch.load(fold_ckpt, map_location=DEVICE, weights_only=True)
            input_size = sequences.size(2)
            config = NextMinuteModelConfig(input_size=input_size, num_dir_classes=num_dir_classes)
            model = NextMinuteBarLSTM(config).to(DEVICE)
            model.load_state_dict(state, strict=True)
            ckpt_path = models_dir / "next_minute_lstm.pt"
            torch.save(model.state_dict(), ckpt_path)
        r = wf_results[best_fold_idx]
        val_summary = r.get("val_summary")
        test_summary = r.get("test_summary")
        direction_confidence_threshold = r.get("direction_confidence_threshold", 0.0)
        metrics = {
            "lookback": LOOKBACK,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "early_stop_metric": EARLY_STOP_METRIC,
            "direction_confidence_threshold": direction_confidence_threshold,
            "val": val_summary,
            "test": test_summary,
            "walk_forward": True,
            "wf_train_days": wf_train_days,
            "wf_test_days": wf_test_days,
            "wf_slide_days": wf_slide_days,
            "wf_folds": [
                {
                    "fold_idx": x["fold_idx"],
                    "train_end": x["train_end"],
                    "test_start": x["test_start"],
                    "test_end": x["test_end"],
                    "test_direction_5m_accuracy": (x.get("test_summary") or {}).get("direction_5m_accuracy"),
                    "test_direction_5m_macro_f1": (x.get("test_summary") or {}).get("direction_5m_macro_f1"),
                    "test_breakout_10m_accuracy": (x.get("test_summary") or {}).get("breakout_10m_accuracy"),
                }
                for x in wf_results
            ],
            "best_fold": best_fold_idx,
        }
        metrics_path = models_dir / "next_minute_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(_nan_to_none(metrics), f, indent=2)
        if r.get("history_epochs") and r.get("history_val"):
            _plot_training_curves(
                r["history_epochs"],
                r["history_train_rmse"],
                r["history_val"],
                models_dir / "training_curves.png",
            )
        _plot_walk_forward_summary(
            [{"fold_idx": x["fold_idx"], "test_start": x["test_start"], "test_end": x["test_end"], "train_end": x["train_end"], "test_summary": x.get("test_summary")} for x in wf_results],
            models_dir / "wf_summary.png",
        )
        print("  Validation metrics (best fold, direction primary):")
        if val_summary:
            print(f"    direction_5m_accuracy:       {val_summary.get('direction_5m_accuracy', float('nan')):.4f}")
            print(f"    direction_5m_macro_f1:       {val_summary.get('direction_5m_macro_f1', float('nan')):.4f}")
        print("  Test metrics (best fold):")
        if test_summary:
            print(f"    direction_5m_accuracy:       {test_summary.get('direction_5m_accuracy', float('nan')):.4f}")
            print(f"    direction_5m_macro_f1:       {test_summary.get('direction_5m_macro_f1', float('nan')):.4f}")
        print(f"  WF summary plot: {models_dir / 'wf_summary.png'}")
        print("=" * 60)
        return
    # --- Single split (non-WF) ---
    split_by_session = getattr(settings, "BAR_VALIDATION_SPLIT_BY_SESSION", True)
    if split_by_session and session_id_per_sample is not None and sessions is not None:
        train_idx, val_idx, test_idx = _session_split_indices(
            n, session_id_per_sample, sessions, VAL_RATIO, TEST_RATIO
        )
        print(
            f"Split by session -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
            f"(VAL_RATIO={VAL_RATIO}, TEST_RATIO={TEST_RATIO})"
        )
    else:
        train_idx, val_idx, test_idx = _time_split_indices(n, VAL_RATIO, TEST_RATIO)
        print(
            f"Split sizes -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
            f"(VAL_RATIO={VAL_RATIO}, TEST_RATIO={TEST_RATIO})"
        )

    models_dir = settings.MODELS_DIR
    ckpt_path = models_dir / "next_minute_lstm.pt"
    resume_state = None
    if getattr(settings, "BAR_RESUME_FROM_CHECKPOINT", False) and ckpt_path.is_file():
        try:
            resume_state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            print(f"Resuming from checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Training from scratch.")
    best_state, val_summary, test_summary, history_epochs, history_train_rmse, history_val, direction_confidence_threshold, best_val_metric_value = _run_one_fold(
        sequences,
        targets_price,
        targets_dir5,
        targets_vol10,
        targets_breakout,
        train_idx,
        val_idx,
        test_idx,
        num_dir_classes,
        resume_state=resume_state,
        fold_label="",
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, ckpt_path)

    metrics = {
        "lookback": LOOKBACK,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "early_stop_metric": EARLY_STOP_METRIC,
        "direction_confidence_threshold": direction_confidence_threshold,
        "val": val_summary,
        "test": test_summary,
    }
    if best_val_metric_value is not None:
        metrics["best_val_metric_value"] = best_val_metric_value
    if getattr(settings, "BAR_DIR5_TUNE_THRESHOLD", False):
        metrics["direction_confidence_threshold_tuned"] = True
    metrics_path = models_dir / "next_minute_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(_nan_to_none(metrics), f, indent=2)

    if history_epochs and history_val:
        curves_path = models_dir / "training_curves.png"
        _plot_training_curves(history_epochs, history_train_rmse, history_val, curves_path)

    # Formatted end-of-training summary
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model checkpoint:  {ckpt_path}")
    print(f"  Metrics JSON:     {metrics_path}")
    if history_epochs and history_val:
        print(f"  Training curves:  {models_dir / 'training_curves.png'}")
    print()
    print("  Validation metrics (direction primary, target ~80%):")
    if val_summary:
        print(f"    direction_5m_accuracy:       {val_summary.get('direction_5m_accuracy', float('nan')):.4f}")
        print(f"    direction_5m_macro_f1:       {val_summary.get('direction_5m_macro_f1', float('nan')):.4f}")
        print(f"    direction_5m_balanced_acc:   {val_summary.get('direction_5m_balanced_accuracy', float('nan')):.4f}")
        print(f"    breakout_10m_accuracy:       {val_summary.get('breakout_10m_accuracy', float('nan')):.4f}")
        print(f"    volatility_10m_rmse:         {val_summary.get('volatility_10m_rmse', float('nan')):.4f}")
        print(f"    price_mae:                   {val_summary.get('price_mae', float('nan')):.4f}")
        print(f"    price_rmse:                  {val_summary.get('price_rmse', float('nan')):.4f}")
        dir_names = ["down", "up"] if num_dir_classes == 2 else ["down", "sideways", "up"]
        for name, prec, rec, f1 in zip(
            dir_names,
            val_summary.get("direction_5m_precision_per_class", []),
            val_summary.get("direction_5m_recall_per_class", []),
            val_summary.get("direction_5m_f1_per_class", []),
        ):
            print(f"      {name}: P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
        print(f"    samples:                     {val_summary.get('samples', 0):,}")
    print()
    print("  Test metrics (direction primary, target ~80%):")
    if test_summary:
        print(f"    direction_5m_accuracy:       {test_summary.get('direction_5m_accuracy', float('nan')):.4f}")
        print(f"    direction_5m_macro_f1:       {test_summary.get('direction_5m_macro_f1', float('nan')):.4f}")
        print(f"    direction_5m_balanced_acc:   {test_summary.get('direction_5m_balanced_accuracy', float('nan')):.4f}")
        print(f"    breakout_10m_accuracy:       {test_summary.get('breakout_10m_accuracy', float('nan')):.4f}")
        print(f"    volatility_10m_rmse:         {test_summary.get('volatility_10m_rmse', float('nan')):.4f}")
        print(f"    price_mae:                   {test_summary.get('price_mae', float('nan')):.4f}")
        print(f"    price_rmse:                  {test_summary.get('price_rmse', float('nan')):.4f}")
        dir_names = ["down", "up"] if num_dir_classes == 2 else ["down", "sideways", "up"]
        for name, prec, rec, f1 in zip(
            dir_names,
            test_summary.get("direction_5m_precision_per_class", []),
            test_summary.get("direction_5m_recall_per_class", []),
            test_summary.get("direction_5m_f1_per_class", []),
        ):
            print(f"      {name}: P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
        print(f"    samples:                     {test_summary.get('samples', 0):,}")
        # Direction accuracy target (~80% for trading focus)
        dir_acc = test_summary.get("direction_5m_accuracy", float("nan"))
        target = 0.80
        if not math.isnan(dir_acc):
            met = "YES" if dir_acc >= target else "no"
            print()
            print(f"  Direction accuracy target: {target:.0%}  ->  test={dir_acc:.2%}  ({met})")
    print("=" * 60)


if __name__ == "__main__":
    main()


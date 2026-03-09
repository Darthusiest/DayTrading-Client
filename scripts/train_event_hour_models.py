"""Train event-driven 1-hour direction models (continuation + reversal).

This script builds an event-triggered dataset from SessionMinuteBar and trains two
separate binary classifiers:
  - continuation: next 60m direction matches event direction (with magnitude band)
  - reversal:     next 60m direction opposes event direction (with magnitude band)

Events supported (configurable via settings / env, see EVENT_*):
  - PDH/PDL sweep (prior day high/low taken)
  - ORB (opening range breakout)
  - ATR expansion
  - impulse candle + volume z-score
  - BOS (break of structure / swing high-low)

Also augments per-bar inputs with a Fibonacci-style scaling based on the
London-session high/low range for that session_date.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
from backend.services.ml.bar_features import build_session_feature_matrix  # noqa: E402
from backend.services.ml.event_hour import EventHourDataset, EventHourLSTM, EventHourModelConfig  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Bump when event input/labeling changes to avoid stale caches.
EVENT_FEATURE_VERSION = 6


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


class _BinaryFocalLoss(nn.Module):
    """Binary focal loss: down-weights easy examples. alpha from pos_weight for imbalance."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = float(pos_weight.item()) if pos_weight is not None else 1.0
        self.gamma = gamma
        self.alpha_pos = self.pos_weight / (1.0 + self.pos_weight)
        self.alpha_neg = 1.0 - self.alpha_pos

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        eps = 1e-7
        p = probs.clamp(eps, 1.0 - eps)
        log_p = torch.log(p)
        log_1mp = torch.log(1.0 - p)
        pt = torch.where(targets >= 0.5, p, 1.0 - p)
        focal_weight = (1.0 - pt) ** self.gamma
        bce = -torch.where(targets >= 0.5, log_p, log_1mp)
        alpha = torch.where(targets >= 0.5, self.alpha_pos, self.alpha_neg)
        out = alpha * focal_weight * bce
        return out.mean() if reduction == "mean" else out


def _parse_hm(s: str) -> Tuple[int, int]:
    parts = s.strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h, m


def _within_hm(bt: datetime, start_hm: Tuple[int, int], end_hm: Tuple[int, int]) -> bool:
    """Inclusive start, exclusive end within same day. Assumes bt is in session timezone."""
    h, m = bt.hour, bt.minute
    t = h * 60 + m
    s = start_hm[0] * 60 + start_hm[1]
    e = end_hm[0] * 60 + end_hm[1]
    if s <= e:
        return s <= t < e
    # Over-midnight window (rare for our use); treat as wrap-around
    return t >= s or t < e


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

    sessions is list of (symbol, session_date_str) in chronological order; session index i = sessions[i].
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
        if train_sids and test_sids:
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


def _plot_wf_summary(
    fold_rows: List[Dict[str, object]],
    save_path: Path,
    metric_key: str = "f1",
) -> None:
    """Plot per-fold test metric for a walk-forward run."""
    if not fold_rows:
        return
    vals = []
    labels = []
    for r in fold_rows:
        ts = r.get("test") or {}
        v = float(ts.get(metric_key, float("nan"))) if isinstance(ts, dict) else float("nan")
        vals.append(0.0 if (isinstance(v, float) and math.isnan(v)) else float(v))
        labels.append(f"{r.get('fold', '')}\\n{str(r.get('test_start', ''))[:7]}")
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.bar(range(len(vals)), vals, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Walk-forward test {metric_key} per fold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int) -> np.ndarray:
    n = closes.size
    tr = np.zeros(n, dtype="float32")
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    atr = np.zeros(n, dtype="float32")
    w = min(max(2, window), n) if n else 1
    for i in range(n):
        start = max(0, i - w + 1)
        atr[i] = float(tr[start : i + 1].mean())
    return atr


def _rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    n = x.size
    out = np.zeros(n, dtype="float32")
    w = min(max(2, window), n) if n else 1
    eps = 1e-8
    for i in range(n):
        start = max(0, i - w + 1)
        win = x[start : i + 1]
        if win.size >= 2 and win.std() > eps:
            out[i] = float((x[i] - win.mean()) / (win.std() + eps))
        else:
            out[i] = 0.0
    return out


def _london_fib_features(
    ordered_bars: Sequence[SessionMinuteBar],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute fib scaling based on London session high/low within the session_date group."""
    london_start = getattr(settings, "EVENT_LONDON_START", "03:00")
    london_end = getattr(settings, "EVENT_LONDON_END", "05:00")
    start_hm = _parse_hm(london_start)
    end_hm = _parse_hm(london_end)

    idx = [i for i, b in enumerate(ordered_bars) if _within_hm(b.bar_time, start_hm, end_hm)]
    if not idx:
        # No London bars available in this dataset/session; return zeros so model remains stable.
        z = np.zeros((len(ordered_bars), 6), dtype="float32")
        return z, {"london_high": float("nan"), "london_low": float("nan")}
    london_high = float(np.max(highs[idx]))
    london_low = float(np.min(lows[idx]))
    rng = max(1e-8, london_high - london_low)

    # Fib scaling: 0 at london_low, 1 at london_high; negative below low; >1 above high.
    fib_scaled = (closes - london_low) / rng

    # Distances (in range-units) to key levels the user mentioned, plus a couple symmetric positives.
    levels = np.array([-4.0, -3.0, -2.5, -2.25, -2.0, -1.5, -1.0, 0.0, 1.0, 2.0, 2.25, 2.5], dtype="float32")
    # nearest distance to any key level
    dist_to_level = np.min(np.abs(fib_scaled.reshape(-1, 1) - levels.reshape(1, -1)), axis=1).astype("float32")

    # Zone flags (continuous 0/1) for the ranges you described
    z_retrace = ((fib_scaled <= -1.0) & (fib_scaled >= -1.5)).astype("float32")
    z_targets = ((fib_scaled <= -2.0) & (fib_scaled >= -2.5)).astype("float32")
    z_deff_rev = ((fib_scaled <= -3.0) & (fib_scaled >= -4.0)).astype("float32")

    # Also include symmetric zones above 1 (optional but helps generalization)
    z_pos_targets = ((fib_scaled >= 2.0) & (fib_scaled <= 2.5)).astype("float32")

    cols = np.stack(
        [
            np.nan_to_num(fib_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype("float32"),
            np.nan_to_num(dist_to_level, nan=0.0, posinf=0.0, neginf=0.0).astype("float32"),
            z_retrace,
            z_targets,
            z_deff_rev,
            z_pos_targets,
        ],
        axis=1,
    ).astype("float32")
    return cols, {"london_high": london_high, "london_low": london_low}


def _load_bars() -> List[SessionMinuteBar]:
    db = SessionLocal()
    try:
        q = (
            db.query(SessionMinuteBar)
            .filter(SessionMinuteBar.symbol.in_(settings.SYMBOLS))
            .order_by(SessionMinuteBar.symbol, SessionMinuteBar.session_date, SessionMinuteBar.bar_time)
        )
        return q.all()
    finally:
        db.close()


def _compute_binary_metrics(probs: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (probs >= threshold).astype(np.int64)
    y = y.astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    total = int(y.size)
    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    pos_rate = float(y.mean()) if total else float("nan")
    # Brier score: mean((p - y)^2); lower is better
    probs_f = probs.astype(np.float64)
    y_f = y.astype(np.float64)
    brier = float(np.mean((probs_f - y_f) ** 2)) if total else float("nan")
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "brier": float(brier),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "pos_rate": float(pos_rate),
        "samples": float(total),
    }


def _predict_probs_and_targets(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (probs, y, avg_loss) for a loader."""
    probs, y, et, loss = _predict_probs_targets_event_type(model, loader)
    return probs, y, loss


def _predict_probs_targets_event_type(
    model: nn.Module, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float]:
    """Return (probs, y, event_type, avg_loss) for a loader. event_type is None if not in batch."""
    model.eval()
    probs_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    et_list: List[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            x = batch["sequence"].to(DEVICE)
            y = batch["target"].to(DEVICE)
            et = batch.get("event_type")
            et = et.to(DEVICE) if et is not None else None
            logits = model(x, event_type=et)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * y.numel()
            n += int(y.numel())
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_list.append(probs)
            y_list.append(y.detach().cpu().numpy())
            if et is not None:
                et_list.append(et.detach().cpu().numpy())
    if n == 0:
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            None,
            float("nan"),
        )
    probs = np.concatenate(probs_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    event_type = np.concatenate(et_list, axis=0) if et_list else None
    return probs, y, event_type, float(loss_sum / max(1, n))


def _compute_val_pnl_proxy(
    probs: np.ndarray, y: np.ndarray, ret_60: np.ndarray, threshold: float = 0.5
) -> float:
    """Proxy PnL: sum over samples with pred>=threshold of sign(pred)*sign(actual)*|ret_60|."""
    if probs.size == 0 or ret_60.size != probs.size:
        return float("nan")
    pred_sign = np.where(probs >= threshold, 1.0, -1.0)
    actual_sign = np.where(y >= 0.5, 1.0, -1.0)
    mask = probs >= threshold
    return float(np.sum(pred_sign[mask] * actual_sign[mask] * np.abs(ret_60[mask])))


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    threshold: float = 0.5,
    forward_return_60m: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    probs, y, avg_loss = _predict_probs_and_targets(model, loader)
    if y.size == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "f1": float("nan"), "samples": 0.0}
    m = _compute_binary_metrics(probs, y, threshold=threshold)
    m["loss"] = float(avg_loss)
    m["threshold"] = float(threshold)
    if forward_return_60m is not None and forward_return_60m.size == probs.size:
        m["val_pnl_proxy"] = _compute_val_pnl_proxy(probs, y, forward_return_60m, threshold)
    return m


def _tune_threshold_for_f1(probs: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Pick threshold that maximizes F1 on provided probs/y."""
    best_th = 0.5
    best = {"f1": -1.0}
    if y.size == 0:
        return best_th, best
    for th in [x / 100.0 for x in range(5, 96, 5)]:  # 0.05..0.95
        m = _compute_binary_metrics(probs, y, threshold=th)
        f1 = m.get("f1", -1.0)
        if not math.isnan(f1) and f1 > best.get("f1", -1.0):
            best_th = th
            best = m
    return float(best_th), best


# Event type id -> metric group for per-event-type reporting (PDH/PDL, ORB, BOS, ATR, IMP)
EVENT_TYPE_TO_METRIC_GROUP = {
    1: "PDH_PDL",
    2: "PDH_PDL",
    3: "ORB",
    4: "ORB",
    5: "ATR",
    6: "ATR",
    7: "IMP",
    8: "IMP",
    9: "BOS",
    10: "BOS",
}
MIN_SAMPLES_PER_EVENT_GROUP = 10


def _compute_binary_metrics_by_event_type(
    probs: np.ndarray,
    y: np.ndarray,
    event_type: Optional[np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1 per event group. Returns {} if event_type is None or group has < MIN_SAMPLES."""
    if event_type is None or event_type.size != y.size:
        return {}
    result: Dict[str, Dict[str, float]] = {}
    for grp_name in ("PDH_PDL", "ORB", "BOS", "ATR", "IMP"):
        mask = np.zeros(event_type.size, dtype=bool)
        for eid, g in EVENT_TYPE_TO_METRIC_GROUP.items():
            if g == grp_name:
                mask |= event_type == eid
        if mask.sum() < MIN_SAMPLES_PER_EVENT_GROUP:
            continue
        p_grp = probs[mask]
        y_grp = y[mask]
        m = _compute_binary_metrics(p_grp, y_grp, threshold=threshold)
        result[grp_name] = {
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "samples": m["samples"],
            "pos_rate": m["pos_rate"],
        }
    return result


def _predict_logits_targets_event_type(
    model: nn.Module, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return (logits, y, event_type) for calibration. event_type is None if not in batch."""
    model.eval()
    logits_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    et_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["sequence"].to(DEVICE)
            y = batch["target"].to(DEVICE)
            et = batch.get("event_type")
            et = et.to(DEVICE) if et is not None else None
            logits = model(x, event_type=et)
            logits_list.append(logits.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())
            if et is not None:
                et_list.append(et.detach().cpu().numpy())
    if not logits_list:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), None
    logits_arr = np.concatenate(logits_list, axis=0)
    y_arr = np.concatenate(y_list, axis=0)
    et_arr = np.concatenate(et_list, axis=0) if et_list else None
    return logits_arr, y_arr, et_arr


def _calibrate_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Fit temperature T to minimize NLL of sigmoid(logits/T) vs y. Grid search (no scipy)."""
    if logits.size == 0:
        return 1.0
    y_ = y.astype(np.float64)
    best_t = 1.0
    best_nll = float("inf")
    for t in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        p = 1.0 / (1.0 + np.exp(-logits.astype(np.float64) / t))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        nll = -np.mean(y_ * np.log(p) + (1.0 - y_) * np.log(1.0 - p))
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return float(best_t)


def _calibrate_platt(logits: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit Platt scaling: P(y=1) = sigmoid(A*logit + B). Returns (A, B)."""
    if logits.size == 0:
        return 1.0, 0.0
    try:
        from scipy.optimize import minimize
    except ImportError:
        return 1.0, 0.0
    logits_f = logits.astype(np.float64).ravel()
    y_f = y.astype(np.float64).ravel()

    def nll(params: np.ndarray) -> float:
        a, b = params[0], params[1]
        p = 1.0 / (1.0 + np.exp(-(a * logits_f + b)))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y_f * np.log(p) + (1.0 - y_f) * np.log(1.0 - p)))

    res = minimize(nll, [1.0, 0.0], method="L-BFGS-B", bounds=[(0.01, 10.0), (-5.0, 5.0)])
    if res.success:
        return float(res.x[0]), float(res.x[1])
    return 1.0, 0.0


def _plot_per_event_performance(
    per_event_test: Dict[str, Dict[str, float]],
    save_path: Path,
    chart_data_path: Path,
) -> None:
    """Plot per-event-type test F1, precision, recall; save chart data JSON."""
    if not per_event_test:
        return
    groups = list(per_event_test.keys())
    f1 = [per_event_test[g].get("f1", 0.0) for g in groups]
    precision = [per_event_test[g].get("precision", 0.0) for g in groups]
    recall = [per_event_test[g].get("recall", 0.0) for g in groups]
    samples = [per_event_test[g].get("samples", 0.0) for g in groups]

    chart_data = {
        "groups": groups,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "samples": samples,
    }
    chart_data_path.parent.mkdir(parents=True, exist_ok=True)
    with chart_data_path.open("w") as f:
        json.dump(chart_data, f, indent=2)

    x = np.arange(len(groups))
    width = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(x - width, f1, width, label="F1", color="#4e79a7")
    ax.bar(x, precision, width, label="Precision", color="#f28e2b")
    ax.bar(x + width, recall, width, label="Recall", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={int(s)})" for g, s in zip(groups, samples)], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test performance by event type")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_curves(history: List[Dict[str, float]], save_path: Path, title: str) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h.get("val_loss", float("nan")) for h in history]
    val_acc = [h.get("val_accuracy", float("nan")) for h in history]
    val_f1 = [h.get("val_f1", float("nan")) for h in history]

    # Save chart data for re-plotting or external use
    chart_data_path = save_path.with_suffix(".chart_data.json")
    chart_data_path.parent.mkdir(parents=True, exist_ok=True)
    with chart_data_path.open("w") as f:
        json.dump(
            {
                "epochs": epochs,
                "train_loss": train_loss,
                "val_loss": [_nan_to_none(v) for v in val_loss],
                "val_accuracy": [_nan_to_none(v) for v in val_acc],
                "val_f1": [_nan_to_none(v) for v in val_f1],
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax[0].plot(epochs, train_loss, "b-o", label="train_loss")
    ax[0].plot(epochs, val_loss, "g-o", label="val_loss")
    ax[0].set_title("Loss")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(epochs, val_acc, "m-o", label="val_acc")
    ax[1].plot(epochs, val_f1, "c-s", label="val_f1")
    ax[1].set_title("Validation")
    ax[1].set_ylim(0, 1)
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _build_event_dataset(
    bars: List[SessionMinuteBar],
    lookback: int,
    horizon_minutes: int = 60,
) -> Dict[str, torch.Tensor | List[Tuple[str, str]] | Dict]:
    """Build event windows and labels for continuation + reversal."""
    # Group by session_date then symbol, so we can compute cross-market (SMT) features.
    by_date: Dict[str, Dict[str, List[SessionMinuteBar]]] = defaultdict(lambda: defaultdict(list))
    for b in bars:
        by_date[b.session_date][b.symbol].append(b)
    session_dates = sorted(by_date.keys())
    # session id is per session_date (shared across symbols)
    sessions_list: List[Tuple[str, str]] = [("ALL", sd) for sd in session_dates]

    # Event config
    atr_window = int(getattr(settings, "BAR_ATR_WINDOW", 14))
    orb_minutes = int(getattr(settings, "EVENT_ORB_MINUTES", 15))
    bos_lookback = int(getattr(settings, "EVENT_BOS_LOOKBACK", 30))
    atr_k = float(getattr(settings, "EVENT_ATR_EXPANSION_K", 1.0))
    range_atr_k = float(getattr(settings, "EVENT_RANGE_ATR_K", 1.5))
    impulse_range_k = float(getattr(settings, "EVENT_IMPULSE_RANGE_ATR_K", 1.5))
    impulse_vol_z = float(getattr(settings, "EVENT_IMPULSE_VOL_Z", 2.0))
    label_band_k = float(getattr(settings, "EVENT_LABEL_BAND_ATR_K", 0.35))
    label_min_band = float(getattr(settings, "EVENT_LABEL_MIN_BAND", 0.0002))
    max_abs_ret_60m = float(getattr(settings, "EVENT_MAX_ABS_RET_60M", 0.20))
    label_skip_minutes = int(getattr(settings, "EVENT_LABEL_SKIP_FIRST_MINUTES", 5))
    min_gap_from_prev = int(getattr(settings, "EVENT_MIN_GAP_FROM_PREV_EVENT", 15))
    label_min_move_atr_k = float(getattr(settings, "EVENT_LABEL_MIN_MOVE_ATR_K", 0.0))
    add_near_miss = bool(getattr(settings, "EVENT_ADD_NEAR_MISS_SAMPLES", False))
    near_miss_max = int(getattr(settings, "EVENT_NEAR_MISS_MAX_PER_SESSION", 20))
    downsample_ratio = float(getattr(settings, "EVENT_DOWNSAMPLE_EASY_NEG_RATIO", 1.0))

    # Continuation filter settings
    cont_require_orb_bos = bool(getattr(settings, "EVENT_CONT_REQUIRE_ORB_AND_BOS", True))
    cont_max_minutes = int(getattr(settings, "EVENT_CONT_MAX_MINUTES_BETWEEN_ORB_BOS", 90))
    cont_require_no_smt = bool(getattr(settings, "EVENT_CONT_REQUIRE_NO_SMT", True))
    cont_event_types_str = str(getattr(settings, "EVENT_CONT_EVENT_TYPES", "ORB_BOS,PDH_PDL"))
    cont_allowed = set(x.strip().upper() for x in cont_event_types_str.split(",") if x.strip())
    # Map event type id -> group: PDH/PDL, ORB/BOS, OTHER
    EVENT_TYPE_TO_GROUP = {
        1: "PDH_PDL", 2: "PDH_PDL", 3: "ORB_BOS", 4: "ORB_BOS",
        5: "OTHER", 6: "OTHER", 7: "OTHER", 8: "OTHER", 9: "ORB_BOS", 10: "ORB_BOS",
    }

    # Triggers enabled
    enable_pdh = bool(getattr(settings, "EVENT_ENABLE_PDH_PDL", True))
    enable_orb = bool(getattr(settings, "EVENT_ENABLE_ORB", True))
    enable_atr = bool(getattr(settings, "EVENT_ENABLE_ATR_EXPANSION", True))
    enable_impulse = bool(getattr(settings, "EVENT_ENABLE_IMPULSE_VOLUME", True))
    enable_bos = bool(getattr(settings, "EVENT_ENABLE_BOS", True))
    enable_smt = bool(getattr(settings, "EVENT_ENABLE_SMT", True))
    smt_lookback = int(getattr(settings, "EVENT_SMT_LOOKBACK", 30))

    # Session start for ORB
    session_start = getattr(settings, "SESSION_START_TIME", "09:30")
    ss_hm = _parse_hm(session_start)
    ss_min = ss_hm[0] * 60 + ss_hm[1]

    # Keep per-symbol previous day range for PDH/PDL
    prev_day_range: Dict[str, Tuple[float, float]] = {}

    seqs: List[np.ndarray] = []
    y_cont: List[float] = []
    y_rev: List[float] = []
    y_forward_return_60m: List[float] = []
    atr_pct_list: List[float] = []
    cont_eligible_list: List[int] = []
    event_dir_list: List[int] = []
    event_type_list: List[int] = []
    session_id_list: List[int] = []
    symbol_id_list: List[int] = []

    EVENT_TYPES = {
        "PDH": 1,
        "PDL": 2,
        "ORB_UP": 3,
        "ORB_DN": 4,
        "ATR_UP": 5,
        "ATR_DN": 6,
        "IMP_UP": 7,
        "IMP_DN": 8,
        "BOS_UP": 9,
        "BOS_DN": 10,
    }

    symbols = list(settings.SYMBOLS)
    sym_to_id = {s: i for i, s in enumerate(symbols)}

    for session_idx, session_date in enumerate(session_dates):
        sym_groups = by_date[session_date]
        # Require both symbols for SMT features; if missing, we still build per-symbol events without SMT
        per_sym = {}
        for symbol, group in sym_groups.items():
            ordered = sorted(group, key=lambda b: b.bar_time)
            all_feat, closes, highs, lows = build_session_feature_matrix(ordered)
            per_sym[symbol] = {
                "ordered": ordered,
                "feat": all_feat,
                "closes": closes,
                "highs": highs,
                "lows": lows,
            }
        if not per_sym:
            continue

        # Build time alignment maps for SMT: intersection of timestamps across available symbols.
        if enable_smt and len(per_sym) >= 2:
            time_sets = []
            for symbol, d in per_sym.items():
                time_sets.append({b.bar_time for b in d["ordered"]})
            common_times = sorted(set.intersection(*time_sets))
        else:
            common_times = []

        # Precompute aligned arrays for each symbol (if SMT enabled and we have common times)
        for symbol, d in per_sym.items():
            ordered = d["ordered"]
            all_feat = d["feat"]
            closes = d["closes"]
            highs = d["highs"]
            lows = d["lows"]
            n = closes.size
            if n < lookback + horizon_minutes + 1:
                prev_day_range[symbol] = (float(highs.max()), float(lows.min()))
                continue

            if enable_smt and common_times:
                idx_map = {b.bar_time: i for i, b in enumerate(ordered)}
                idxs = np.array([idx_map[t] for t in common_times if t in idx_map], dtype=np.int64)
                if idxs.size < lookback + horizon_minutes + 1:
                    # Not enough common minutes; fall back to local indexing without SMT
                    idxs = None
            else:
                idxs = None

            if idxs is not None:
                ordered = [ordered[i] for i in idxs.tolist()]
                all_feat = all_feat[idxs]
                closes = closes[idxs]
                highs = highs[idxs]
                lows = lows[idxs]
                n = closes.size

            # Compute arrays for triggers/labels
            opens = all_feat[:, 0].astype("float32")
            atr = _compute_atr(highs, lows, closes, atr_window)
            ret_1m = np.zeros(n, dtype="float32")
            if n > 1:
                np.divide(
                    closes[1:] - closes[:-1],
                    closes[:-1],
                    out=ret_1m[1:],
                    where=closes[:-1] > 0,
                )
            bar_range = highs - lows
            range_over_atr = bar_range / (atr + 1e-8)
            vol = all_feat[:, 4].astype("float32")
            vol_z = _rolling_zscore(vol, 20)
            fib_cols, _ = _london_fib_features(ordered, highs, lows, closes)

            # SMT cross-market features (if available for this session_date)
            smt_cols = np.zeros((n, 6), dtype="float32")
            if enable_smt and len(per_sym) >= 2:
                # pick one other symbol (for now assume exactly two: MNQ1! and MES1!)
                other = [s for s in per_sym.keys() if s != symbol]
                if other:
                    other_symbol = other[0]
                    od = per_sym[other_symbol]
                    other_ordered = sorted(od["ordered"], key=lambda b: b.bar_time)
                    other_feat, other_closes, other_highs, other_lows = build_session_feature_matrix(other_ordered)
                    if idxs is not None:
                        idx_map2 = {b.bar_time: i for i, b in enumerate(other_ordered)}
                        idxs2 = np.array([idx_map2[b.bar_time] for b in ordered if b.bar_time in idx_map2], dtype=np.int64)
                        if idxs2.size == n:
                            other_feat = other_feat[idxs2]
                            other_closes = other_closes[idxs2]
                            other_highs = other_highs[idxs2]
                            other_lows = other_lows[idxs2]
                    other_atr = _compute_atr(other_highs, other_lows, other_closes, atr_window)
                    other_ret_1m = np.zeros(n, dtype="float32")
                    if n > 1:
                        np.divide(
                            other_closes[1:] - other_closes[:-1],
                            other_closes[:-1],
                            out=other_ret_1m[1:],
                            where=other_closes[:-1] > 0,
                        )

                    spread_ret_1m = ret_1m - other_ret_1m
                    # SMT divergence: HH/LL break mismatch over smt_lookback window
                    smt_hh = np.zeros(n, dtype="float32")
                    smt_ll = np.zeros(n, dtype="float32")
                    for i2 in range(n):
                        start = max(0, i2 - smt_lookback)
                        if i2 <= start:
                            continue
                        sym_hh = closes[i2] > np.max(closes[start:i2])
                        oth_hh = other_closes[i2] > np.max(other_closes[start:i2])
                        sym_ll = closes[i2] < np.min(closes[start:i2])
                        oth_ll = other_closes[i2] < np.min(other_closes[start:i2])
                        smt_hh[i2] = 1.0 if (sym_hh != oth_hh) else 0.0
                        smt_ll[i2] = 1.0 if (sym_ll != oth_ll) else 0.0

                    # Normalize ATR as pct for both
                    atr_pct = np.where(closes > 0, atr / (closes + 1e-8), 0.0).astype("float32")
                    other_atr_pct = np.where(other_closes > 0, other_atr / (other_closes + 1e-8), 0.0).astype("float32")
                    smt_cols = np.stack(
                        [
                            other_ret_1m.astype("float32"),
                            spread_ret_1m.astype("float32"),
                            other_atr_pct.astype("float32"),
                            atr_pct.astype("float32"),
                            smt_hh.astype("float32"),
                            smt_ll.astype("float32"),
                        ],
                        axis=1,
                    ).astype("float32")

            all_feat_aug = np.concatenate([all_feat, fib_cols, smt_cols], axis=1).astype("float32")

            # ORB range (within session start+orb_minutes)
            orb_end_min = ss_min + orb_minutes
            orb_idx = [
                i for i, b in enumerate(ordered) if ss_min <= (b.bar_time.hour * 60 + b.bar_time.minute) < orb_end_min
            ]
            orb_high = float(np.max(highs[orb_idx])) if orb_idx else float("nan")
            orb_low = float(np.min(lows[orb_idx])) if orb_idx else float("nan")

            # Precompute ORB/BOS first occurrences per direction for continuation setup filter
            orb_up_idx_first: Optional[int] = None
            orb_dn_idx_first: Optional[int] = None
            if enable_orb and orb_idx:
                for j in range(max(1, orb_idx[-1] + 1), n):
                    if orb_up_idx_first is None and closes[j] > orb_high and closes[j - 1] <= orb_high:
                        orb_up_idx_first = j
                    if orb_dn_idx_first is None and closes[j] < orb_low and closes[j - 1] >= orb_low:
                        orb_dn_idx_first = j
                    if orb_up_idx_first is not None and orb_dn_idx_first is not None:
                        break

            bos_up_idx_first: Optional[int] = None
            bos_dn_idx_first: Optional[int] = None
            if enable_bos and bos_lookback > 1:
                for j in range(bos_lookback, n):
                    prev_swing_high = float(np.max(highs[j - bos_lookback : j]))
                    prev_swing_low = float(np.min(lows[j - bos_lookback : j]))
                    if bos_up_idx_first is None and closes[j] > prev_swing_high and closes[j - 1] <= prev_swing_high:
                        bos_up_idx_first = j
                    if bos_dn_idx_first is None and closes[j] < prev_swing_low and closes[j - 1] >= prev_swing_low:
                        bos_dn_idx_first = j
                    if bos_up_idx_first is not None and bos_dn_idx_first is not None:
                        break

            fired = set()  # per-session dedupe keys
            pdh_idx_first: Optional[int] = None
            pdl_idx_first: Optional[int] = None
            last_bar_any_event: int = -999
            no_event_bars: List[int] = []

            prev_high, prev_low = prev_day_range.get(symbol, (float("nan"), float("nan")))
            # Label window excludes event spike: ret from close[i+skip] to close[i+skip+horizon]
            i_max = n - horizon_minutes - label_skip_minutes - 1
            for i in range(lookback - 1, i_max + 1):
                # Label band (vol-adjusted); magnitude threshold when EVENT_LABEL_MIN_MOVE_ATR_K > 0
                c0 = float(closes[i]) if closes[i] > 0 else 0.0
                atr_pct_i = float(atr[i]) / c0 if c0 > 0 else 0.0
                band = max(label_min_band, label_band_k * atr_pct_i)
                if label_min_move_atr_k > 0:
                    band = max(band, label_min_move_atr_k * atr_pct_i)

                # Compute 60m return (exclude event spike: start at i+label_skip_minutes)
                c_start = float(closes[i + label_skip_minutes])
                c_end = float(closes[i + label_skip_minutes + horizon_minutes])
                ret_60 = (c_end - c_start) / c_start if c_start > 0 else 0.0
                if abs(ret_60) > max_abs_ret_60m:
                    continue

                # Determine event(s) at i
                events: List[Tuple[int, int]] = []  # (event_type_id, event_dir +1/-1)

                # PDH/PDL sweep
                if enable_pdh and not math.isnan(prev_high) and not math.isnan(prev_low):
                    if "PDH" not in fired and highs[i] >= prev_high and (i == 0 or highs[i - 1] < prev_high):
                        events.append((EVENT_TYPES["PDH"], +1))
                        fired.add("PDH")
                        if pdh_idx_first is None:
                            pdh_idx_first = i
                    if "PDL" not in fired and lows[i] <= prev_low and (i == 0 or lows[i - 1] > prev_low):
                        events.append((EVENT_TYPES["PDL"], -1))
                        fired.add("PDL")
                        if pdl_idx_first is None:
                            pdl_idx_first = i

                # ORB breakout (only after ORB window)
                if enable_orb and orb_idx:
                    tmin = ordered[i].bar_time.hour * 60 + ordered[i].bar_time.minute
                    if tmin >= orb_end_min:
                        if "ORB_UP" not in fired and closes[i] > orb_high and closes[i - 1] <= orb_high:
                            events.append((EVENT_TYPES["ORB_UP"], +1))
                            fired.add("ORB_UP")
                        if "ORB_DN" not in fired and closes[i] < orb_low and closes[i - 1] >= orb_low:
                            events.append((EVENT_TYPES["ORB_DN"], -1))
                            fired.add("ORB_DN")

                # ATR expansion / volatility shock
                if enable_atr:
                    if "ATR_UP" not in fired and ret_1m[i] > atr_k * atr_pct_i:
                        events.append((EVENT_TYPES["ATR_UP"], +1))
                        fired.add("ATR_UP")
                    if "ATR_DN" not in fired and ret_1m[i] < -atr_k * atr_pct_i:
                        events.append((EVENT_TYPES["ATR_DN"], -1))
                        fired.add("ATR_DN")
                    # Range/ATR spike: direction by close-open
                    if range_over_atr[i] > range_atr_k:
                        d = +1 if closes[i] >= opens[i] else -1
                        key = f"RANGE_{d}"
                        if key not in fired:
                            events.append((EVENT_TYPES["ATR_UP"] if d > 0 else EVENT_TYPES["ATR_DN"], d))
                            fired.add(key)

                # Impulse candle + volume
                if enable_impulse and range_over_atr[i] > impulse_range_k and vol_z[i] > impulse_vol_z:
                    d = +1 if closes[i] >= opens[i] else -1
                    key = f"IMP_{d}"
                    if key not in fired:
                        events.append((EVENT_TYPES["IMP_UP"] if d > 0 else EVENT_TYPES["IMP_DN"], d))
                        fired.add(key)

                # Break of structure (swing high/low)
                if enable_bos and i >= bos_lookback:
                    prev_swing_high = float(np.max(highs[i - bos_lookback : i]))
                    prev_swing_low = float(np.min(lows[i - bos_lookback : i]))
                    if "BOS_UP" not in fired and closes[i] > prev_swing_high and closes[i - 1] <= prev_swing_high:
                        events.append((EVENT_TYPES["BOS_UP"], +1))
                        fired.add("BOS_UP")
                    if "BOS_DN" not in fired and closes[i] < prev_swing_low and closes[i - 1] >= prev_swing_low:
                        events.append((EVENT_TYPES["BOS_DN"], -1))
                        fired.add("BOS_DN")

                if not events:
                    no_event_bars.append(i)
                    continue
                # Skip if another event fired within min_gap minutes (avoid overlapping labels)
                if min_gap_from_prev > 0 and (i - last_bar_any_event) < min_gap_from_prev:
                    continue

                # Event-specific features at bar i (bars since ORB/BOS/PDH-PDL, normalized by 60)
                norm_bars = 60.0
                bs_orb_up = min(1.0, (i - orb_up_idx_first) / norm_bars) if orb_up_idx_first is not None and i >= orb_up_idx_first else 0.0
                bs_orb_dn = min(1.0, (i - orb_dn_idx_first) / norm_bars) if orb_dn_idx_first is not None and i >= orb_dn_idx_first else 0.0
                bs_bos_up = min(1.0, (i - bos_up_idx_first) / norm_bars) if bos_up_idx_first is not None and i >= bos_up_idx_first else 0.0
                bs_bos_dn = min(1.0, (i - bos_dn_idx_first) / norm_bars) if bos_dn_idx_first is not None and i >= bos_dn_idx_first else 0.0
                bs_pdh = min(1.0, (i - pdh_idx_first) / norm_bars) if pdh_idx_first is not None and i >= pdh_idx_first else 0.0
                bs_pdl = min(1.0, (i - pdl_idx_first) / norm_bars) if pdl_idx_first is not None and i >= pdl_idx_first else 0.0
                enable_event_type_feat = bool(getattr(settings, "EVENT_ENABLE_EVENT_TYPE_FEATURES", True))

                # For each event at i, create one sample. (Dedup rules above keep this small.)
                base_event_feat = [bs_orb_up, bs_orb_dn, (bs_bos_up + bs_bos_dn) / 2.0, max(bs_pdh, bs_pdl)]

                sign = 0
                if abs(ret_60) > band:
                    sign = +1 if ret_60 > 0 else -1

                for etype, edir in events:
                    # Event-type encoding: normalized etype (0.1-1.0) when enabled; 0 when disabled (keep dim fixed)
                    event_feat_row = list(base_event_feat)
                    event_feat_row.append(float(etype) / 10.0 if enable_event_type_feat else 0.0)
                    event_feat_arr = np.array(event_feat_row, dtype="float32")
                    event_feat_window = np.tile(event_feat_arr, (lookback, 1))

                    window = all_feat_aug[i - lookback + 1 : i + 1]
                    if window.shape[0] != lookback:
                        continue
                    window = np.concatenate([window, event_feat_window], axis=1).astype("float32")

                    cont = 1.0 if (sign != 0 and sign == edir) else 0.0
                    rev = 1.0 if (sign != 0 and sign == -edir) else 0.0
                    # Continuation setup filter: require ORB+BOS same direction occurred within window
                    cont_ok = True
                    if cont_require_orb_bos:
                        if edir > 0:
                            oidx, bidx = orb_up_idx_first, bos_up_idx_first
                        else:
                            oidx, bidx = orb_dn_idx_first, bos_dn_idx_first
                        if oidx is None or bidx is None:
                            cont_ok = False
                        else:
                            if bidx < oidx:
                                cont_ok = False
                            elif (bidx - oidx) > cont_max_minutes:
                                cont_ok = False
                            elif i < bidx:
                                cont_ok = False  # event must occur after BOS
                    if cont_require_no_smt and enable_smt:
                        # smt_cols last two columns are divergence flags
                        if smt_cols[i, 4] > 0.5 or smt_cols[i, 5] > 0.5:
                            cont_ok = False
                    # Event-type filter: only ORB_BOS, PDH_PDL, or ALL
                    cont_event_type_ok = True
                    if cont_allowed and "ALL" not in cont_allowed:
                        grp = EVENT_TYPE_TO_GROUP.get(etype, "OTHER")
                        cont_event_type_ok = grp in cont_allowed
                    cont_eligible_final = cont_ok and cont_event_type_ok
                    # Down-sample easy negatives: |ret_60| < 0.5*band, cont=0, rev=0
                    is_easy_neg = (cont == 0.0 and rev == 0.0 and abs(ret_60) < 0.5 * band)
                    if is_easy_neg and downsample_ratio < 1.0 and np.random.random() >= downsample_ratio:
                        continue
                    seqs.append(window.astype("float32"))
                    # If setup not eligible, drop the continuation label to 0 by construction.
                    # Training will further filter by eligibility when building the continuation dataset.
                    y_cont.append(cont if cont_ok else 0.0)
                    y_rev.append(rev)
                    y_forward_return_60m.append(float(ret_60))
                    atr_pct_list.append(float(atr_pct_i))
                    cont_eligible_list.append(1 if cont_eligible_final else 0)
                    event_dir_list.append(1 if edir > 0 else 0)  # 1=up,0=down
                    event_type_list.append(int(etype))
                    session_id_list.append(int(session_idx))
                    symbol_id_list.append(int(sym_to_id.get(symbol, 0)))
                last_bar_any_event = i

            # Near-miss non-events: bars with similar vol profile but no event, as explicit negatives
            if add_near_miss and near_miss_max > 0 and len(no_event_bars) > 0:
                # Top 30% by range_over_atr and vol_z (i.e. above 70th percentile)
                roa = range_over_atr
                vz = np.nan_to_num(vol_z, nan=0.0, posinf=0.0, neginf=-10.0)
                p70_roa = float(np.nanpercentile(roa, 70)) if roa.size > 0 else 0.0
                p70_vz = float(np.nanpercentile(vz, 70)) if vz.size > 0 else 0.0
                added = 0
                for i in no_event_bars:
                    if added >= near_miss_max:
                        break
                    if i < lookback - 1 or i + label_skip_minutes + horizon_minutes >= n:
                        continue
                    if roa[i] < p70_roa or vz[i] < p70_vz:
                        continue
                    c_start = float(closes[i + label_skip_minutes])
                    c_end = float(closes[i + label_skip_minutes + horizon_minutes])
                    ret_60 = (c_end - c_start) / c_start if c_start > 0 else 0.0
                    if abs(ret_60) > max_abs_ret_60m:
                        continue
                    band_nm = max(label_min_band, label_band_k * (float(atr[i]) / closes[i] if closes[i] > 0 else 0))
                    if abs(ret_60) > band_nm:
                        continue  # only add when |ret_60| <= band (no clear direction)
                    # Neutral event-type features for near-miss
                    event_feat_nm = np.array([0.0, 0.0, 0.0, 0.0, 0.0 if enable_event_type_feat else 0.0], dtype="float32")
                    event_feat_window_nm = np.tile(event_feat_nm, (lookback, 1))
                    window_nm = all_feat_aug[i - lookback + 1 : i + 1]
                    if window_nm.shape[0] != lookback:
                        continue
                    window_nm = np.concatenate([window_nm, event_feat_window_nm], axis=1).astype("float32")
                    atr_pct_nm = float(atr[i]) / closes[i] if closes[i] > 0 else 1e-4
                    seqs.append(window_nm)
                    y_cont.append(0.0)
                    y_rev.append(0.0)
                    y_forward_return_60m.append(float(ret_60))
                    atr_pct_list.append(atr_pct_nm)
                    cont_eligible_list.append(0)
                    event_dir_list.append(0)
                    event_type_list.append(0)
                    session_id_list.append(int(session_idx))
                    symbol_id_list.append(int(sym_to_id.get(symbol, 0)))
                    added += 1

            prev_day_range[symbol] = (float(highs.max()), float(lows.min()))

    if not seqs:
        raise RuntimeError("No event samples were built. Check trigger thresholds or data coverage.")

    sequences = torch.from_numpy(np.stack(seqs).astype("float32"))
    t_cont = torch.from_numpy(np.asarray(y_cont, dtype="float32"))
    t_rev = torch.from_numpy(np.asarray(y_rev, dtype="float32"))
    t_ret60 = torch.from_numpy(np.asarray(y_forward_return_60m, dtype="float32"))
    t_atr_pct = torch.from_numpy(np.asarray(atr_pct_list, dtype="float32"))
    cont_eligible = torch.from_numpy(np.asarray(cont_eligible_list, dtype="int64"))
    event_dir = torch.from_numpy(np.asarray(event_dir_list, dtype="int64"))
    event_type = torch.from_numpy(np.asarray(event_type_list, dtype="int64"))
    session_id = torch.from_numpy(np.asarray(session_id_list, dtype="int64"))
    symbol_id = torch.from_numpy(np.asarray(symbol_id_list, dtype="int64"))

    return {
        "sequences": sequences,
        "targets_cont": t_cont,
        "targets_rev": t_rev,
        "forward_return_60m": t_ret60,
        "atr_pct": t_atr_pct,
        "cont_eligible": cont_eligible,
        "event_dir": event_dir,
        "event_type": event_type,
        "session_id": session_id,
        "symbol_id": symbol_id,
        "sessions": sessions_list,
        "meta": {
            "lookback": lookback,
            "horizon_minutes": horizon_minutes,
            "symbols": list(settings.SYMBOLS),
            "event_feature_version": EVENT_FEATURE_VERSION,
            "enable_smt": bool(enable_smt),
            "smt_lookback": int(smt_lookback),
            "label_band_atr_k": float(label_band_k),
            "label_min_band": float(label_min_band),
            "max_abs_ret_60m": float(max_abs_ret_60m),
            "label_skip_minutes": int(label_skip_minutes),
            "min_gap_from_prev_event": int(min_gap_from_prev),
            "cont_require_orb_bos": bool(cont_require_orb_bos),
            "cont_max_minutes": int(cont_max_minutes),
            "cont_require_no_smt": bool(cont_require_no_smt),
            "event_cont_types": cont_event_types_str,
            "session_phase": True,
            "enable_event_type_features": bool(enable_event_type_feat),
            "label_min_move_atr_k": float(label_min_move_atr_k),
            "add_near_miss": bool(add_near_miss),
            "near_miss_max": int(near_miss_max),
            "downsample_ratio": float(downsample_ratio),
        },
    }


def _session_split(
    session_id: torch.Tensor,
    num_sessions: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    sid = session_id.detach().cpu().numpy()
    n_train_sess = max(1, int(round((1.0 - val_ratio - test_ratio) * num_sessions)))
    n_val_sess = max(0, int(round(val_ratio * num_sessions)))
    n_test_sess = num_sessions - n_train_sess - n_val_sess
    if n_test_sess < 0:
        n_test_sess = 0
        n_val_sess = num_sessions - n_train_sess
    train_sids = set(range(n_train_sess))
    val_sids = set(range(n_train_sess, n_train_sess + n_val_sess))
    test_sids = set(range(n_train_sess + n_val_sess, num_sessions))
    train_idx = [i for i in range(sid.size) if sid[i] in train_sids]
    val_idx = [i for i in range(sid.size) if sid[i] in val_sids]
    test_idx = [i for i in range(sid.size) if sid[i] in test_sids]
    return train_idx, val_idx, test_idx


def _train_one(
    model_name: str,
    sequences: torch.Tensor,
    targets: torch.Tensor,
    event_type: torch.Tensor,
    event_dir: torch.Tensor,
    session_id: torch.Tensor,
    symbol_id: torch.Tensor,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    out_dir: Path,
    forward_return_60m: Optional[torch.Tensor] = None,
    atr_pct: Optional[torch.Tensor] = None,
) -> Tuple[dict, dict]:
    batch_size = int(getattr(settings, "EVENT_BATCH_SIZE", 256))
    lr = float(getattr(settings, "EVENT_LR", 3e-4))
    epochs = int(getattr(settings, "EVENT_EPOCHS", 25))
    patience = int(getattr(settings, "EVENT_EARLY_STOP_PATIENCE", 8))

    # Sample weights for cost-aware loss: 1 + |ret_60|/atr_pct (clamp atr_pct to avoid div by zero)
    sample_weight = None
    if bool(getattr(settings, "EVENT_LOSS_WEIGHT_BY_MAGNITUDE", False)) and forward_return_60m is not None and atr_pct is not None:
        ret = forward_return_60m.detach().cpu().numpy()
        ap = atr_pct.detach().cpu().numpy()
        ap_safe = np.maximum(ap, 1e-6)
        w = 1.0 + np.abs(ret) / ap_safe
        sample_weight = torch.from_numpy(w.astype("float32"))

    def _make_ds(seq, tgt, et, ed, sid, symid, idx):
        sw_sub = sample_weight[idx] if sample_weight is not None else None
        return EventHourDataset(seq, tgt, et, ed, sid, symid, sample_weight=sw_sub)

    train_ds = _make_ds(
        sequences[train_idx], targets[train_idx], event_type[train_idx], event_dir[train_idx],
        session_id[train_idx], symbol_id[train_idx], train_idx,
    )
    val_ds = (
        _make_ds(
            sequences[val_idx], targets[val_idx], event_type[val_idx], event_dir[val_idx],
            session_id[val_idx], symbol_id[val_idx], val_idx,
        )
        if val_idx
        else None
    )
    test_ds = (
        _make_ds(
            sequences[test_idx], targets[test_idx], event_type[test_idx], event_dir[test_idx],
            session_id[test_idx], symbol_id[test_idx], test_idx,
        )
        if test_idx
        else None
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None

    input_size = sequences.size(2)
    config = EventHourModelConfig(input_size=input_size)
    model = EventHourLSTM(config).to(DEVICE)

    # pos_weight for imbalance; scale by EVENT_POS_WEIGHT_SCALE
    y = targets[train_idx].detach().cpu().numpy()
    pos = float((y >= 0.5).sum())
    neg = float(y.size - pos)
    pos_weight_raw = (neg / max(1.0, pos)) if pos > 0 else None
    pos_weight_scale = float(getattr(settings, "EVENT_POS_WEIGHT_SCALE", 1.5))
    pos_weight = (
        torch.tensor([pos_weight_raw * pos_weight_scale], dtype=torch.float32, device=DEVICE)
        if pos_weight_raw is not None and pos_weight_scale > 0
        else None
    )

    use_focal = bool(getattr(settings, "EVENT_USE_FOCAL", True))
    focal_gamma = float(getattr(settings, "EVENT_FOCAL_GAMMA", 2.0))
    if use_focal:
        criterion = _BinaryFocalLoss(pos_weight=pos_weight, gamma=focal_gamma).to(DEVICE)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=float(getattr(settings, "EVENT_WEIGHT_DECAY", 1e-5)))

    # LR scheduler: cosine_warmup, cosine, or none
    lr_scheduler_name = str(getattr(settings, "EVENT_LR_SCHEDULER", "cosine_warmup"))
    warmup_epochs = int(getattr(settings, "EVENT_LR_WARMUP_EPOCHS", 2))
    if lr_scheduler_name == "cosine_warmup" and warmup_epochs > 0:
        def _warmup_cosine(ep: int) -> float:
            if ep < warmup_epochs:
                return (ep + 1) / warmup_epochs
            progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_cosine)
    elif lr_scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = None

    best_state = None
    best_val = -math.inf
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for batch in train_loader:
            x = batch["sequence"].to(DEVICE)
            t = batch["target"].to(DEVICE)
            et = batch.get("event_type")
            et = et.to(DEVICE) if et is not None else None
            sw = batch.get("sample_weight")
            sw = sw.to(DEVICE) if sw is not None else None
            optimizer.zero_grad()
            logits = model(x, event_type=et)
            if sw is not None and use_focal:
                loss_per = criterion(logits, t, reduction="none")
                loss = (loss_per * sw).mean()
            elif sw is not None and not use_focal:
                loss_per = nn.functional.binary_cross_entropy_with_logits(
                    logits, t, reduction="none"
                )
                loss = (loss_per * sw).mean()
            else:
                loss = criterion(logits, t)
            loss.backward()
            clip = float(getattr(settings, "EVENT_GRAD_CLIP_NORM", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            loss_sum += float(loss.item()) * t.numel()
            n_seen += int(t.numel())
        train_loss = loss_sum / max(1, n_seen)

        ret_val = (
            forward_return_60m[val_idx].detach().cpu().numpy()
            if forward_return_60m is not None and val_idx
            else None
        )
        val_metrics = (
            _evaluate(model, val_loader, threshold=0.5, forward_return_60m=ret_val)
            if val_loader is not None
            else {"f1": float("nan"), "accuracy": float("nan"), "loss": float("nan")}
        )
        stop_metric = str(getattr(settings, "EVENT_EARLY_STOP_METRIC", "f1"))
        val_score = float(val_metrics.get(stop_metric, float("nan")))
        min_delta = float(getattr(settings, "EVENT_EARLY_STOP_MIN_DELTA", 0.0))
        if not math.isnan(val_score) and val_score > best_val + min_delta:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics.get("loss", float("nan"))),
                "val_accuracy": float(val_metrics.get("accuracy", float("nan"))),
                "val_f1": float(val_metrics.get("f1", float("nan"))),
            }
        )
        print(
            f"{model_name} Epoch {epoch}/{epochs} | train_loss={train_loss:.5f} "
            f"| val_acc={val_metrics.get('accuracy', float('nan')):.4f} "
            f"| val_f1={val_metrics.get('f1', float('nan')):.4f}"
        )
        if scheduler is not None:
            scheduler.step()
        if no_improve >= patience:
            print(f"{model_name}: early stop after {epoch} epochs (best {best_val:.4f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # Threshold tuning on validation to maximize F1 (for this model)
    best_threshold = 0.5
    ret_val = forward_return_60m[val_idx].detach().cpu().numpy() if forward_return_60m is not None and val_idx else None
    ret_test = forward_return_60m[test_idx].detach().cpu().numpy() if forward_return_60m is not None and test_idx else None
    val_at_0_5 = _evaluate(model, val_loader, threshold=0.5, forward_return_60m=ret_val) if val_loader is not None else None
    test_at_0_5 = _evaluate(model, test_loader, threshold=0.5, forward_return_60m=ret_test) if test_loader is not None else None
    if val_loader is not None:
        probs_v, y_v, _ = _predict_probs_and_targets(model, val_loader)
        best_threshold, _ = _tune_threshold_for_f1(probs_v, y_v)
    val_summary = _evaluate(model, val_loader, threshold=best_threshold, forward_return_60m=ret_val) if val_loader is not None else None
    test_summary = _evaluate(model, test_loader, threshold=best_threshold, forward_return_60m=ret_test) if test_loader is not None else None

    # Per-event-type metrics (val and test)
    per_event_val: Dict[str, Dict[str, float]] = {}
    per_event_test: Dict[str, Dict[str, float]] = {}
    if val_loader is not None:
        probs_v, y_v, et_v, _ = _predict_probs_targets_event_type(model, val_loader)
        if et_v is not None:
            per_event_val = _compute_binary_metrics_by_event_type(
                probs_v, y_v, et_v, threshold=best_threshold
            )
    if test_loader is not None:
        probs_t, y_t, et_t, _ = _predict_probs_targets_event_type(model, test_loader)
        if et_t is not None:
            per_event_test = _compute_binary_metrics_by_event_type(
                probs_t, y_t, et_t, threshold=best_threshold
            )

    # Calibration on validation logits: temperature or Platt scaling
    calibration_method = str(getattr(settings, "EVENT_CALIBRATION_METHOD", "temperature"))
    temperature = 1.0
    platt_a, platt_b = 1.0, 0.0
    if val_loader is not None:
        logits_v, y_cal, _ = _predict_logits_targets_event_type(model, val_loader)
        if logits_v.size > 0 and y_cal.size > 0:
            if calibration_method.lower() == "platt":
                platt_a, platt_b = _calibrate_platt(logits_v, y_cal)
            else:
                temperature = _calibrate_temperature(logits_v, y_cal)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    curves_path = out_dir / f"{model_name}_curves.png"
    _plot_curves(history, curves_path, title=f"{model_name} — training curves")

    # Save calibration JSON for inference
    calibration_path = out_dir / f"{model_name}_calibration.json"
    cal_data = {"method": calibration_method}
    if calibration_method.lower() == "platt":
        cal_data["A"] = float(platt_a)
        cal_data["B"] = float(platt_b)
        cal_data["usage"] = "At inference: probs = 1 / (1 + exp(-(A*logits + B)))"
    else:
        cal_data["temperature"] = float(temperature)
        cal_data["usage"] = "At inference: probs = 1 / (1 + exp(-logits / temperature))"
    with calibration_path.open("w") as f:
        json.dump(cal_data, f, indent=2)

    metrics = {
        "model_name": model_name,
        "config": asdict(config),
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "early_stop_metric": getattr(settings, "EVENT_EARLY_STOP_METRIC", "f1"),
        "threshold_tuned": True,
        "best_threshold": float(best_threshold),
        "temperature": float(temperature),
        "calibration_method": calibration_method,
        "val_at_0_5": val_at_0_5,
        "test_at_0_5": test_at_0_5,
        "val": val_summary,
        "test": test_summary,
        "per_event_type_val": per_event_val,
        "per_event_type_test": per_event_test,
        "history": history,
        "train_pos_rate": float(y.mean()) if y.size else float("nan"),
        "train_pos_weight": float(pos_weight.item()) if pos_weight is not None else None,
    }
    metrics_path = out_dir / f"{model_name}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(_nan_to_none(metrics), f, indent=2)

    # Per-event-type performance chart and chart data
    if per_event_test:
        perf_path = out_dir / f"{model_name}_performance.png"
        chart_data_path = out_dir / f"{model_name}_chart_data.json"
        _plot_per_event_performance(per_event_test, perf_path, chart_data_path)

    return metrics, {
        "ckpt": str(ckpt_path),
        "curves": str(curves_path),
        "metrics": str(metrics_path),
        "calibration": str(calibration_path),
    }


def main() -> None:
    print("Loading SessionMinuteBar data...")
    bars = _load_bars()
    if not bars:
        print("No SessionMinuteBar rows found.")
        return
    print(f"Loaded {len(bars)} minute bars.")

    lookback = int(getattr(settings, "EVENT_LOOKBACK", getattr(settings, "BAR_LOOKBACK", 30)))
    out_dir = settings.MODELS_DIR

    cache_path = out_dir / "event_hour_dataset.pt"
    rebuild = bool(getattr(settings, "EVENT_REBUILD_CACHE", True))
    if cache_path.is_file() and not rebuild:
        data = torch.load(cache_path, map_location="cpu")
        sequences = data["sequences"]
        targets_cont = data["targets_cont"]
        targets_rev = data["targets_rev"]
        forward_return_60m = data.get("forward_return_60m")
        atr_pct = data.get("atr_pct")
        cont_eligible = data.get("cont_eligible")
        event_type = data["event_type"]
        event_dir = data["event_dir"]
        session_id = data["session_id"]
        symbol_id = data.get("symbol_id")
        sessions = data.get("sessions", [])
        meta = data.get("meta", {})
        if meta.get("event_feature_version") != EVENT_FEATURE_VERSION:
            print("Event dataset cache feature_version mismatch; rebuilding.")
            rebuild = True
        if bool(meta.get("enable_smt")) != bool(getattr(settings, "EVENT_ENABLE_SMT", True)):
            print("Event dataset cache SMT setting mismatch; rebuilding.")
            rebuild = True
        # Rebuild if label-band or continuation filter settings changed
        if float(meta.get("label_band_atr_k", -1.0)) != float(getattr(settings, "EVENT_LABEL_BAND_ATR_K", 0.35)):
            print("Event dataset cache label band mismatch; rebuilding.")
            rebuild = True
        if float(meta.get("label_min_band", -1.0)) != float(getattr(settings, "EVENT_LABEL_MIN_BAND", 0.0002)):
            print("Event dataset cache min band mismatch; rebuilding.")
            rebuild = True
        if float(meta.get("max_abs_ret_60m", -1.0)) != float(getattr(settings, "EVENT_MAX_ABS_RET_60M", 0.20)):
            print("Event dataset cache max_abs_ret_60m mismatch; rebuilding.")
            rebuild = True
        if bool(meta.get("cont_require_orb_bos", True)) != bool(getattr(settings, "EVENT_CONT_REQUIRE_ORB_AND_BOS", True)):
            print("Event dataset cache continuation ORB+BOS setting mismatch; rebuilding.")
            rebuild = True
        if int(meta.get("cont_max_minutes", -1)) != int(getattr(settings, "EVENT_CONT_MAX_MINUTES_BETWEEN_ORB_BOS", 90)):
            print("Event dataset cache continuation window mismatch; rebuilding.")
            rebuild = True
        if bool(meta.get("cont_require_no_smt", True)) != bool(getattr(settings, "EVENT_CONT_REQUIRE_NO_SMT", True)):
            print("Event dataset cache continuation SMT filter mismatch; rebuilding.")
            rebuild = True
        if meta.get("event_cont_types", "") != str(getattr(settings, "EVENT_CONT_EVENT_TYPES", "ORB_BOS,PDH_PDL")):
            print("Event dataset cache event_cont_types mismatch; rebuilding.")
            rebuild = True
        if int(meta.get("label_skip_minutes", -1)) != int(getattr(settings, "EVENT_LABEL_SKIP_FIRST_MINUTES", 5)):
            print("Event dataset cache label_skip_minutes mismatch; rebuilding.")
            rebuild = True
        if int(meta.get("min_gap_from_prev_event", -1)) != int(getattr(settings, "EVENT_MIN_GAP_FROM_PREV_EVENT", 15)):
            print("Event dataset cache min_gap_from_prev_event mismatch; rebuilding.")
            rebuild = True
        if bool(meta.get("enable_event_type_features", True)) != bool(getattr(settings, "EVENT_ENABLE_EVENT_TYPE_FEATURES", True)):
            print("Event dataset cache event_type_features mismatch; rebuilding.")
            rebuild = True
        if int(meta.get("horizon_minutes", -1)) != int(getattr(settings, "EVENT_LABEL_HORIZON_MINUTES", 60)):
            print("Event dataset cache horizon_minutes mismatch; rebuilding.")
            rebuild = True
        if float(meta.get("label_min_move_atr_k", -1.0)) != float(getattr(settings, "EVENT_LABEL_MIN_MOVE_ATR_K", 0.0)):
            print("Event dataset cache label_min_move_atr_k mismatch; rebuilding.")
            rebuild = True
        if bool(meta.get("add_near_miss", False)) != bool(getattr(settings, "EVENT_ADD_NEAR_MISS_SAMPLES", False)):
            print("Event dataset cache add_near_miss mismatch; rebuilding.")
            rebuild = True
        if int(meta.get("near_miss_max", -1)) != int(getattr(settings, "EVENT_NEAR_MISS_MAX_PER_SESSION", 20)):
            print("Event dataset cache near_miss_max mismatch; rebuilding.")
            rebuild = True
        if float(meta.get("downsample_ratio", -1.0)) != float(getattr(settings, "EVENT_DOWNSAMPLE_EASY_NEG_RATIO", 1.0)):
            print("Event dataset cache downsample_ratio mismatch; rebuilding.")
            rebuild = True
        if not meta.get("session_phase", False):
            print("Event dataset cache missing session_phase; rebuilding.")
            rebuild = True
        if forward_return_60m is None:
            print("Event dataset cache missing forward_return_60m; rebuilding.")
            rebuild = True
        if atr_pct is None:
            atr_pct = torch.ones(sequences.size(0), dtype=torch.float32) * 1e-4  # fallback for old caches
        print(f"Using cached event dataset: {cache_path} (N={sequences.size(0)})")
    if not cache_path.is_file() or rebuild:
        print(f"Building event dataset (lookback={lookback})...")
        horizon = int(getattr(settings, "EVENT_LABEL_HORIZON_MINUTES", 60))
        built = _build_event_dataset(bars, lookback=lookback, horizon_minutes=horizon)
        sequences = built["sequences"]
        targets_cont = built["targets_cont"]
        targets_rev = built["targets_rev"]
        forward_return_60m = built["forward_return_60m"]
        atr_pct = built.get("atr_pct")
        if atr_pct is None:
            atr_pct = torch.ones(sequences.size(0), dtype=torch.float32) * 1e-4
        cont_eligible = built.get("cont_eligible")
        event_type = built["event_type"]
        event_dir = built["event_dir"]
        session_id = built["session_id"]
        symbol_id = built.get("symbol_id")
        sessions = built["sessions"]
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sequences": sequences,
                "targets_cont": targets_cont,
                "targets_rev": targets_rev,
                "forward_return_60m": forward_return_60m,
                "atr_pct": atr_pct,
                "cont_eligible": cont_eligible,
                "event_type": event_type,
                "event_dir": event_dir,
                "session_id": session_id,
                "symbol_id": symbol_id,
                "sessions": sessions,
                "meta": built.get("meta", {}),
            },
            cache_path,
        )
        print(f"Saved event dataset cache: {cache_path} (N={sequences.size(0)})")

    if symbol_id is None:
        symbol_id = torch.zeros_like(session_id)
    if cont_eligible is None:
        cont_eligible = torch.ones_like(session_id)
    n_sessions = len(sessions) if isinstance(sessions, list) else int(session_id.max().item() + 1)
    val_ratio = float(getattr(settings, "EVENT_VAL_RATIO", 0.1))
    test_ratio = float(getattr(settings, "EVENT_TEST_RATIO", 0.1))
    walk_forward = bool(getattr(settings, "EVENT_WALK_FORWARD", False))
    if walk_forward and isinstance(sessions, list) and sessions:
        wf_train_days = int(getattr(settings, "EVENT_WF_TRAIN_DAYS", 365))
        wf_test_days = int(getattr(settings, "EVENT_WF_TEST_DAYS", 90))
        wf_slide_days = int(getattr(settings, "EVENT_WF_SLIDE_DAYS", 90))
        wf_val_ratio = float(getattr(settings, "EVENT_WF_VAL_RATIO", 0.1))
        folds = _walk_forward_fold_ranges(sessions, wf_train_days, wf_test_days, wf_slide_days)
        if not folds:
            print("EVENT_WALK_FORWARD enabled but no folds could be generated; falling back to single split.")
            walk_forward = False
        else:
            print(
                f"Walk-forward: {len(folds)} folds (train={wf_train_days}d, test={wf_test_days}d, slide={wf_slide_days}d)"
            )
    else:
        folds = []
        wf_val_ratio = 0.1

    if walk_forward and folds:
        wf_dir = out_dir / "event_wf"
        wf_dir.mkdir(parents=True, exist_ok=True)
        fold_rows: List[Dict[str, object]] = []
        for fold_idx, (train_sids, test_sids, train_start, train_end, test_start, test_end) in enumerate(folds):
            n = int(sequences.size(0))
            train_idx, val_idx, test_idx = _wf_split_indices(
                n, session_id, train_sids, test_sids, wf_val_ratio
            )
            if len(train_idx) < 1000 or len(test_idx) < 100:
                print(f"WF fold {fold_idx + 1}: skipping (train={len(train_idx)}, test={len(test_idx)})")
                continue
            fold_out = wf_dir / f"fold_{fold_idx}"
            print()
            print("=" * 60)
            print(f"WF Fold {fold_idx + 1}/{len(folds)}  train {train_start}->{train_end}  test {test_start}->{test_end}")
            print("=" * 60)
            cont_metrics, cont_paths = _train_one(
                f"event_hour_continuation_fold_{fold_idx}",
                sequences,
                targets_cont,
                event_type,
                event_dir,
                session_id,
                symbol_id,
                [i for i in train_idx if int(cont_eligible[i].item()) == 1],
                [i for i in val_idx if int(cont_eligible[i].item()) == 1],
                [i for i in test_idx if int(cont_eligible[i].item()) == 1],
                fold_out,
                forward_return_60m=forward_return_60m,
                atr_pct=atr_pct,
            )
            rev_metrics, rev_paths = _train_one(
                f"event_hour_reversal_fold_{fold_idx}",
                sequences,
                targets_rev,
                event_type,
                event_dir,
                session_id,
                symbol_id,
                train_idx,
                val_idx,
                test_idx,
                fold_out,
                forward_return_60m=forward_return_60m,
                atr_pct=atr_pct,
            )
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "cont": cont_metrics,
                    "rev": rev_metrics,
                }
            )

        if not fold_rows:
            print("Walk-forward: no folds completed. Exiting.")
            return

        selection_metric = str(getattr(settings, "EVENT_EARLY_STOP_METRIC", "f1"))

        def _best_fold(model_key: str, metric_key: str) -> int:
            best_i = 0
            best_v = -1.0
            for i, r in enumerate(fold_rows):
                m = r.get(model_key, {})
                if isinstance(m, dict):
                    vs = m.get("val", {}) if isinstance(m.get("val", {}), dict) else {}
                    v = vs.get(metric_key, -1.0)
                    try:
                        v = float(v)
                    except Exception:
                        v = -1.0
                    if not math.isnan(v) and v > best_v:
                        best_v = v
                        best_i = i
            return best_i

        best_cont_i = _best_fold("cont", selection_metric)
        best_rev_i = _best_fold("rev", selection_metric)

        # Save final best models to canonical paths in data/models/
        for key, best_i, final_name in [
            ("cont", best_cont_i, "event_hour_continuation"),
            ("rev", best_rev_i, "event_hour_reversal"),
        ]:
            m = fold_rows[best_i].get(key, {})
            if not isinstance(m, dict):
                continue
            # Load fold checkpoint and re-save to final name
            fold = int(fold_rows[best_i]["fold"])
            fold_ckpt = wf_dir / f"fold_{fold}" / f"{final_name}_fold_{fold}.pt"
            if fold_ckpt.is_file():
                state = torch.load(fold_ckpt, map_location="cpu")
                torch.save(state, out_dir / f"{final_name}.pt")
            # Write final metrics as best fold + wf summary
            final_metrics = {
                "walk_forward": True,
                "selection_split": "val",
                "selection_metric": selection_metric,
                "best_fold": int(fold_rows[best_i]["fold"]),
                "best_fold_window": {
                    "train_start": fold_rows[best_i]["train_start"],
                    "train_end": fold_rows[best_i]["train_end"],
                    "test_start": fold_rows[best_i]["test_start"],
                    "test_end": fold_rows[best_i]["test_end"],
                },
                "best_fold_metrics": m,
                "folds": [
                    {
                        "fold": r["fold"],
                        "train_end": r["train_end"],
                        "test_start": r["test_start"],
                        "test_end": r["test_end"],
                        "val_metric": (r.get(key, {}).get("val", {}) or {}).get(selection_metric) if isinstance(r.get(key, {}), dict) else None,
                        "test_f1": (r.get(key, {}).get("test", {}) or {}).get("f1") if isinstance(r.get(key, {}), dict) else None,
                        "test_accuracy": (r.get(key, {}).get("test", {}) or {}).get("accuracy") if isinstance(r.get(key, {}), dict) else None,
                        "test_pos_rate": (r.get(key, {}).get("test", {}) or {}).get("pos_rate") if isinstance(r.get(key, {}), dict) else None,
                    }
                    for r in fold_rows
                ],
            }
            with (out_dir / f"{final_name}_metrics.json").open("w") as f:
                json.dump(_nan_to_none(final_metrics), f, indent=2)
            # Curves (best fold history)
            hist = m.get("history", [])
            if isinstance(hist, list) and hist:
                _plot_curves(hist, out_dir / f"{final_name}_curves.png", title=f"{final_name} — best fold curves")

        # WF summary plots
        cont_rows = [
            {"fold": r["fold"], "test_start": r["test_start"], "test": (r.get("cont", {}) or {}).get("test", {})}
            for r in fold_rows
            if isinstance(r.get("cont", {}), dict)
        ]
        rev_rows = [
            {"fold": r["fold"], "test_start": r["test_start"], "test": (r.get("rev", {}) or {}).get("test", {})}
            for r in fold_rows
            if isinstance(r.get("rev", {}), dict)
        ]
        _plot_wf_summary(cont_rows, out_dir / "event_hour_continuation_wf_summary.png", metric_key="f1")
        _plot_wf_summary(rev_rows, out_dir / "event_hour_reversal_wf_summary.png", metric_key="f1")

        with (out_dir / "event_hour_wf_summary.json").open("w") as f:
            json.dump(_nan_to_none({"folds": fold_rows, "best_cont": best_cont_i, "best_rev": best_rev_i}), f, indent=2)
        print("Walk-forward complete. Wrote best models + wf summaries to data/models/.")
        return

    # Single split path
    train_idx, val_idx, test_idx = _session_split(session_id, n_sessions, val_ratio=val_ratio, test_ratio=test_ratio)
    print(f"Split by session -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Continuation setup-specific filter: restrict continuation training to eligible subset
    cont_train_idx = [i for i in train_idx if int(cont_eligible[i].item()) == 1]
    cont_val_idx = [i for i in val_idx if int(cont_eligible[i].item()) == 1]
    cont_test_idx = [i for i in test_idx if int(cont_eligible[i].item()) == 1]
    print(
        f"Continuation eligible -> train={len(cont_train_idx)}, val={len(cont_val_idx)}, test={len(cont_test_idx)}"
    )

    print()
    print("=" * 60)
    print("Training continuation model...")
    print("=" * 60)
    cont_metrics, cont_paths = _train_one(
        "event_hour_continuation",
        sequences,
        targets_cont,
        event_type,
        event_dir,
        session_id,
        symbol_id,
        cont_train_idx,
        cont_val_idx,
        cont_test_idx,
        out_dir,
        forward_return_60m=forward_return_60m,
        atr_pct=atr_pct,
    )
    print()
    print("=" * 60)
    print("Training reversal model...")
    print("=" * 60)
    rev_metrics, rev_paths = _train_one(
        "event_hour_reversal",
        sequences,
        targets_rev,
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

    summary = {
        "event_dataset_cache": str(cache_path),
        "continuation": cont_paths,
        "reversal": rev_paths,
    }
    summary_path = out_dir / "event_hour_summary.json"
    with summary_path.open("w") as f:
        json.dump(_nan_to_none(summary), f, indent=2)
    print()
    print("Saved event-hour summary:", summary_path)


if __name__ == "__main__":
    main()


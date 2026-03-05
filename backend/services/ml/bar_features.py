"""Bar feature engineering utilities for 1m SessionMinuteBar data.

This module centralizes all derived features built from OHLCV and time for
minute bars so that both sequence models (LSTM) and tabular models can share
the same feature definitions.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from backend.config.settings import settings
from backend.database.models import SessionMinuteBar


def _build_base_ohlcv(bars: Sequence[SessionMinuteBar]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return base OHLCV matrix [T,5] and separate close/high/low vectors (float32)."""
    n_bars = len(bars)
    feat = np.zeros((n_bars, 5), dtype="float32")
    highs = np.zeros(n_bars, dtype="float32")
    lows = np.zeros(n_bars, dtype="float32")
    closes = np.zeros(n_bars, dtype="float32")

    for i, g in enumerate(bars):
        o = float(g.open_price)
        h = float(g.high_price)
        l = float(g.low_price)
        c = float(g.close_price)
        v = float(g.volume or 0.0)
        feat[i, 0] = o
        feat[i, 1] = h
        feat[i, 2] = l
        feat[i, 3] = c
        feat[i, 4] = v
        highs[i] = h
        lows[i] = l
        closes[i] = c

    return feat, closes, highs, lows


def _price_and_volatility_features(closes: np.ndarray, vols: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Price-based returns, log-returns, rolling volatility, VWAP, RSI, MACD, momentum."""
    n_bars = closes.shape[0]

    returns = np.zeros(n_bars, dtype="float32")
    log_returns = np.zeros(n_bars, dtype="float32")
    if n_bars > 1:
        prev_close = closes[:-1].copy()
        curr_close = closes[1:].copy()
        # Require both prices to be strictly positive to avoid invalid ratios/logs
        valid = (prev_close > 0) & (curr_close > 0)
        ret = np.zeros_like(curr_close)
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_ratio = np.zeros_like(curr_close, dtype="float32")
            safe_ratio[valid] = curr_close[valid] / prev_close[valid]
            ret[valid] = safe_ratio[valid] - 1.0
            returns[1:] = ret

            log_ret = np.zeros_like(curr_close, dtype="float32")
            log_ret[valid] = np.log(safe_ratio[valid])
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
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap[nonzero] = cum_pv[nonzero] / cum_v[nonzero]
    if n_bars and not nonzero[0]:
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

    # Replace any NaN/inf that might still slip through with safe zeros
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
    rolling_vol = np.nan_to_num(rolling_vol, nan=0.0, posinf=0.0, neginf=0.0)
    vwap = np.nan_to_num(vwap, nan=0.0, posinf=0.0, neginf=0.0)
    rsi = np.nan_to_num(rsi, nan=50.0, posinf=100.0, neginf=0.0)
    macd = np.nan_to_num(macd, nan=0.0, posinf=0.0, neginf=0.0)
    momentum = np.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)

    return returns, log_returns, rolling_vol, vwap, rsi, macd, momentum


def _time_features(bars: Sequence[SessionMinuteBar]) -> Tuple[np.ndarray, ...]:
    """Time-of-day and session-relative features from bar_time."""
    n_bars = len(bars)
    hours = np.zeros(n_bars, dtype="float32")
    minutes = np.zeros(n_bars, dtype="float32")
    day_of_week = np.zeros(n_bars, dtype="float32")
    minutes_since_open = np.zeros(n_bars, dtype="float32")
    is_ny_open = np.zeros(n_bars, dtype="float32")
    is_power_hour = np.zeros(n_bars, dtype="float32")

    def _parse_hm(s: str) -> tuple[int, int]:
        parts = s.strip().split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        return h, m

    start_h, start_m = _parse_hm(settings.SESSION_START_TIME)
    end_h, end_m = _parse_hm(settings.SESSION_END_TIME)
    start_total = start_h * 60 + start_m
    end_total = end_h * 60 + end_m

    for i, g in enumerate(bars):
        bt = g.bar_time  # naive datetime in session TZ
        h = bt.hour
        m = bt.minute
        hours[i] = float(h)
        minutes[i] = float(m)
        day_of_week[i] = float(bt.weekday())

        total_min = h * 60 + m
        ms_open = max(0, total_min - start_total)
        minutes_since_open[i] = float(ms_open)

        # Example flags: first 60 minutes of session as \"NY open\", last 60 as \"power hour\"
        is_ny_open[i] = 1.0 if 0 <= ms_open <= 60 else 0.0
        mins_to_close = end_total - total_min
        is_power_hour[i] = 1.0 if 0 <= mins_to_close <= 60 else 0.0

    return hours, minutes, day_of_week, minutes_since_open, is_ny_open, is_power_hour


def build_session_feature_matrix(
    bars: Sequence[SessionMinuteBar],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build full feature matrix for a single (symbol, session_date) sequence.

    Args:
        bars: Iterable of SessionMinuteBar for one symbol+session, ANY order.

    Returns:
        features: [T, F] float32 (OHLCV + derived + time features)
        closes:   [T]    float32
        highs:    [T]    float32
        lows:     [T]    float32
    """
    if not bars:
        raise ValueError("build_session_feature_matrix() received empty bars sequence")

    # Ensure deterministic order by bar_time
    ordered = sorted(bars, key=lambda b: b.bar_time)
    base, closes, highs, lows = _build_base_ohlcv(ordered)
    vols = base[:, 4].copy()

    (
        returns,
        log_returns,
        rolling_vol,
        vwap,
        rsi,
        macd,
        momentum,
    ) = _price_and_volatility_features(closes, vols)

    hours, minutes, dow, minutes_since_open, is_ny_open, is_power_hour = _time_features(
        ordered
    )

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
            dow,
            minutes_since_open,
            is_ny_open,
            is_power_hour,
        ],
        axis=1,
    ).astype("float32")

    features = np.concatenate([base, derived_cols], axis=1)
    # Final safety: ensure no NaN/inf reaches the model
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    closes = np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0)
    highs = np.nan_to_num(highs, nan=0.0, posinf=0.0, neginf=0.0)
    lows = np.nan_to_num(lows, nan=0.0, posinf=0.0, neginf=0.0)

    return features, closes.astype("float32"), highs.astype("float32"), lows.astype("float32")


# NOTE: Cross-market features (MNQ vs MES) can be added with helper functions here
# in the future, e.g. by aligning two symbol sequences on timestamp and augmenting
# the returned feature matrix with MES-based returns and spreads.


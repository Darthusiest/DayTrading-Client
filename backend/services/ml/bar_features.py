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

    # Multi-horizon returns (1m already in returns; add 5m and 10m for direction context)
    return_5m = np.zeros(n_bars, dtype="float32")
    for i in range(5, n_bars):
        if closes[i - 5] > 0:
            return_5m[i] = (closes[i] - closes[i - 5]) / closes[i - 5]
    return_10m = np.zeros(n_bars, dtype="float32")
    for i in range(10, n_bars):
        if closes[i - 10] > 0:
            return_10m[i] = (closes[i] - closes[i - 10]) / closes[i - 10]
    return_15m = np.zeros(n_bars, dtype="float32")
    for i in range(15, n_bars):
        if closes[i - 15] > 0:
            return_15m[i] = (closes[i] - closes[i - 15]) / closes[i - 15]

    # VWAP distance: (close - vwap) / vwap (normalized; positive = above VWAP)
    vwap_safe = np.where(vwap > 0, vwap, 1.0)
    vwap_distance = (closes - vwap) / vwap_safe

    # Volume delta: 1-bar relative change (vol[i] - vol[i-1]) / (vol[i-1] + eps)
    volume_delta = np.zeros(n_bars, dtype="float32")
    if n_bars > 1:
        prev_vol = vols[:-1]
        curr_vol = vols[1:]
        denom = np.where(prev_vol > 0, prev_vol, 1.0)
        volume_delta[1:] = (curr_vol - prev_vol) / denom

    # Replace any NaN/inf that might still slip through with safe zeros
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
    rolling_vol = np.nan_to_num(rolling_vol, nan=0.0, posinf=0.0, neginf=0.0)
    vwap = np.nan_to_num(vwap, nan=0.0, posinf=0.0, neginf=0.0)
    rsi = np.nan_to_num(rsi, nan=50.0, posinf=100.0, neginf=0.0)
    macd = np.nan_to_num(macd, nan=0.0, posinf=0.0, neginf=0.0)
    momentum = np.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)
    return_5m = np.nan_to_num(return_5m, nan=0.0, posinf=0.0, neginf=0.0)
    return_10m = np.nan_to_num(return_10m, nan=0.0, posinf=0.0, neginf=0.0)
    return_15m = np.nan_to_num(return_15m, nan=0.0, posinf=0.0, neginf=0.0)
    vwap_distance = np.nan_to_num(vwap_distance, nan=0.0, posinf=0.0, neginf=0.0)
    volume_delta = np.nan_to_num(volume_delta, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        returns,
        log_returns,
        rolling_vol,
        vwap,
        rsi,
        macd,
        momentum,
        return_5m,
        return_10m,
        return_15m,
        vwap_distance,
        volume_delta,
    )


def _time_features(bars: Sequence[SessionMinuteBar]) -> Tuple[np.ndarray, ...]:
    """Time-of-day and session-relative features from bar_time."""
    n_bars = len(bars)
    hours = np.zeros(n_bars, dtype="float32")
    minutes = np.zeros(n_bars, dtype="float32")
    day_of_week = np.zeros(n_bars, dtype="float32")
    minutes_since_open = np.zeros(n_bars, dtype="float32")
    is_ny_open = np.zeros(n_bars, dtype="float32")
    is_power_hour = np.zeros(n_bars, dtype="float32")
    session_phase = np.zeros(n_bars, dtype="float32")

    def _parse_hm(s: str) -> tuple[int, int]:
        parts = s.strip().split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        return h, m

    start_h, start_m = _parse_hm(settings.SESSION_START_TIME)
    end_h, end_m = _parse_hm(settings.SESSION_END_TIME)
    start_total = start_h * 60 + start_m
    end_total = end_h * 60 + end_m
    session_duration = max(1, end_total - start_total)

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

        # Session phase: 0=open, 0.5=mid, 1=close
        session_phase[i] = float(np.clip(ms_open / session_duration, 0.0, 1.0))

        # Example flags: first 60 minutes of session as \"NY open\", last 60 as \"power hour\"
        is_ny_open[i] = 1.0 if 0 <= ms_open <= 60 else 0.0
        mins_to_close = end_total - total_min
        is_power_hour[i] = 1.0 if 0 <= mins_to_close <= 60 else 0.0

    return hours, minutes, day_of_week, minutes_since_open, is_ny_open, is_power_hour, session_phase


def _regime_and_microstructure(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    vols: np.ndarray,
    rolling_vol: np.ndarray,
    atr: np.ndarray,
    vol_window: int = 20,
    gap_pct_threshold: float = 1e-6,
) -> Tuple[np.ndarray, ...]:
    """Volatility regime, trend strength, wick/body ratios, range/ATR, volume z-score, gap flag."""
    n_bars = len(closes)
    eps = 1e-8

    # Volatility regime: tertiles of rolling_vol (0, 0.5, 1) or single float by percentile
    vol_regime = np.zeros(n_bars, dtype="float32")
    if n_bars >= 3 and np.any(rolling_vol > 0):
        p33 = np.nanpercentile(rolling_vol[rolling_vol > 0], 33.33)
        p66 = np.nanpercentile(rolling_vol[rolling_vol > 0], 66.66)
        vol_regime[:] = 0.5
        vol_regime[rolling_vol <= p33] = 0.0
        vol_regime[rolling_vol >= p66] = 1.0

    # Trend strength: |close - SMA(20)| / (ATR + eps)
    sma_period = min(20, max(1, n_bars))
    sma = np.zeros(n_bars, dtype="float32")
    for i in range(n_bars):
        start = max(0, i - sma_period + 1)
        sma[i] = float(closes[start : i + 1].mean())
    trend_strength = np.abs(closes - sma) / (atr + eps)

    # Wick/body ratios: upper_wick, lower_wick, body over range
    bar_range = highs - lows
    bar_range_safe = np.where(bar_range > 0, bar_range, 1.0)
    body = np.abs(closes - opens)
    upper_wick = highs - np.maximum(opens, closes)
    lower_wick = np.minimum(opens, closes) - lows
    upper_wick_ratio = np.clip(upper_wick / bar_range_safe, 0.0, 1.0)
    lower_wick_ratio = np.clip(lower_wick / bar_range_safe, 0.0, 1.0)
    body_ratio = np.clip(body / bar_range_safe, 0.0, 1.0)

    # Range / ATR
    range_over_atr = bar_range / (atr + eps)

    # Volume z-score (rolling window)
    vol_win = min(vol_window, n_bars)
    volume_zscore = np.zeros(n_bars, dtype="float32")
    for i in range(n_bars):
        start = max(0, i - vol_win + 1)
        w = vols[start : i + 1]
        if w.size >= 2 and w.std() > 0:
            volume_zscore[i] = float((vols[i] - w.mean()) / (w.std() + eps))
        else:
            volume_zscore[i] = 0.0

    # Gap flag: 1 if |open - prev_close|/prev_close > threshold
    gap_flag = np.zeros(n_bars, dtype="float32")
    if n_bars > 1:
        prev_close = closes[:-1]
        curr_open = opens[1:]
        gap_pct = np.abs(curr_open - prev_close) / (prev_close + eps)
        gap_flag[1:] = (gap_pct > gap_pct_threshold).astype("float32")

    return (
        vol_regime,
        trend_strength,
        upper_wick_ratio,
        lower_wick_ratio,
        body_ratio,
        range_over_atr,
        volume_zscore,
        gap_flag,
    )


def _atr_and_close_zscore(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr_window: int = 14,
    zscore_window: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """ATR (average true range) and rolling z-score of close for volatility/level context."""
    n_bars = closes.shape[0]
    atr = np.zeros(n_bars, dtype="float32")
    tr = np.zeros(n_bars, dtype="float32")
    for j in range(1, n_bars):
        hl = highs[j] - lows[j]
        hc = abs(highs[j] - closes[j - 1])
        lc = abs(lows[j] - closes[j - 1])
        tr[j] = max(hl, hc, lc)
    w = min(atr_window, n_bars)
    for j in range(n_bars):
        start = max(0, j - w + 1)
        atr[j] = float(tr[start : j + 1].mean())

    close_zscore = np.zeros(n_bars, dtype="float32")
    win = min(zscore_window, n_bars)
    for j in range(n_bars):
        start = max(0, j - win + 1)
        window = closes[start : j + 1]
        mu = window.mean()
        sigma = window.std()
        if sigma and sigma > 1e-8:
            close_zscore[j] = (closes[j] - mu) / sigma
        else:
            close_zscore[j] = 0.0
    return atr, close_zscore


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
        return_5m,
        return_10m,
        return_15m,
        vwap_distance,
        volume_delta,
    ) = _price_and_volatility_features(closes, vols)

    atr_window = getattr(settings, "BAR_ATR_WINDOW", 14)
    atr, close_zscore = _atr_and_close_zscore(closes, highs, lows, atr_window=atr_window, zscore_window=20)

    opens = base[:, 0].copy()
    (
        vol_regime,
        trend_strength,
        upper_wick_ratio,
        lower_wick_ratio,
        body_ratio,
        range_over_atr,
        volume_zscore,
        gap_flag,
    ) = _regime_and_microstructure(opens, closes, highs, lows, vols, rolling_vol, atr)
    vol_regime = np.nan_to_num(vol_regime, nan=0.5, posinf=1.0, neginf=0.0)
    trend_strength = np.nan_to_num(trend_strength, nan=0.0, posinf=0.0, neginf=0.0)
    upper_wick_ratio = np.nan_to_num(upper_wick_ratio, nan=0.0, posinf=1.0, neginf=0.0)
    lower_wick_ratio = np.nan_to_num(lower_wick_ratio, nan=0.0, posinf=1.0, neginf=0.0)
    body_ratio = np.nan_to_num(body_ratio, nan=0.0, posinf=1.0, neginf=0.0)
    range_over_atr = np.nan_to_num(range_over_atr, nan=0.0, posinf=0.0, neginf=0.0)
    volume_zscore = np.nan_to_num(volume_zscore, nan=0.0, posinf=0.0, neginf=0.0)
    gap_flag = np.nan_to_num(gap_flag, nan=0.0, posinf=1.0, neginf=0.0)

    hours, minutes, dow, minutes_since_open, is_ny_open, is_power_hour, session_phase = _time_features(
        ordered
    )

    # Stack all features: OHLCV + returns (1m/5m/10m/15m), rolling_vol, VWAP distance, volume delta,
    # ATR, close_zscore, regime + microstructure (vol_regime, trend_strength, wick/body, range/ATR, vol_zscore, gap), time
    derived_cols = np.stack(
        [
            returns,        # 1m return
            log_returns,
            return_5m,
            return_10m,
            return_15m,
            rolling_vol,
            vwap_distance,
            volume_delta,
            vwap,
            rsi,
            macd,
            momentum,
            atr,
            close_zscore,
            vol_regime,
            trend_strength,
            upper_wick_ratio,
            lower_wick_ratio,
            body_ratio,
            range_over_atr,
            volume_zscore,
            gap_flag,
            hours,
            minutes,
            dow,
            minutes_since_open,
            is_ny_open,
            is_power_hour,
            session_phase,
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


"""Trading strategy domain models and simple backtest utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Iterable, List, Optional

import numpy as np

# Default annualization: assume ~5 trades per session day, 252 session days/year
DEFAULT_TRADES_PER_YEAR = 252 * 5


class SignalDirection(str, Enum):
    """Direction suggested by a model signal."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    """Model output normalized for strategy/backtest usage."""

    timestamp: datetime
    symbol: str
    source: str
    direction: SignalDirection
    confidence: float
    horizon_minutes: int
    metadata: Dict[str, float | str | int] = field(default_factory=dict)


@dataclass
class RiskConfig:
    """Risk controls for sizing and execution constraints."""

    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    max_concurrent_positions: int = 1
    fee_per_trade: float = 0.10
    slippage_bps: float = 1.0
    min_confidence: float = 0.0
    # Optional position sizing (P2)
    sizing_method: str = "fixed"  # "fixed" | "vol_target" | "kelly" | "confidence_scaled"
    kelly_fraction: float = 0.5  # cap for half-Kelly when sizing_method == "kelly"
    max_risk_fraction: float = 0.25  # cap notional as fraction of capital (e.g. 0.25 = 25%)
    min_vol_for_target: float = 1e-6  # floor vol when vol_target to avoid huge notionals
    # Optional cost model (P3): spread + market-impact style slippage
    spread_bps: float = 0.0  # half-spread cost (applied once per round-trip)
    impact_coef: float = 0.0  # slippage_bps += impact_coef * sqrt(notional / max(adv, 1))
    adv: float = 0.0  # average daily volume/notional for impact; 0 => use notional-only form
    vol_slippage_coef: float = 0.0  # (P7) slip_bps += vol_slippage_coef * vol_at_signal when vol_at_signal provided


@dataclass
class Trade:
    """Single executed trade record."""

    symbol: str
    source: str
    entry_time: datetime
    exit_time: datetime
    direction: SignalDirection
    confidence: float
    size: float
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    return_pct: float


@dataclass
class Position:
    """Open position state (included for future live execution)."""

    symbol: str
    direction: SignalDirection
    size: float
    entry_price: float
    entry_time: datetime
    source: str


def _notional_for_sizing(
    capital: float,
    risk: RiskConfig,
    confidence: float,
    vol_at_signal: Optional[float],
    win_rate: Optional[float],
    avg_win: Optional[float],
    avg_loss: Optional[float],
) -> float:
    """Compute notional from sizing method. Returns 0 if invalid/fallback."""
    base = capital * float(risk.risk_per_trade)
    method = (risk.sizing_method or "fixed").strip().lower()
    max_frac = max(1e-6, float(getattr(risk, "max_risk_fraction", 0.25)))

    if method == "fixed":
        return max(0.0, base)

    if method == "vol_target":
        if vol_at_signal is None or vol_at_signal <= 0:
            return max(0.0, base)
        min_vol = max(1e-12, getattr(risk, "min_vol_for_target", 1e-6))
        vol = max(vol_at_signal, min_vol)
        notional = base / vol
        cap = capital * max_frac
        return max(0.0, min(notional, cap))

    if method == "kelly":
        if (
            win_rate is None
            or avg_win is None
            or avg_loss is None
            or avg_loss <= 0
        ):
            return max(0.0, base)
        p, q = float(win_rate), 1.0 - float(win_rate)
        b = float(avg_win) / float(avg_loss)
        kelly_f = (p * b - q) / b if b > 0 else 0.0
        kelly_frac = float(getattr(risk, "kelly_fraction", 0.5))
        f = max(0.0, min(kelly_frac * kelly_f, max_frac))
        return max(0.0, capital * f)

    if method == "confidence_scaled":
        min_conf = float(risk.min_confidence)
        if min_conf >= 1.0:
            return max(0.0, base)
        scale = (confidence - min_conf) / (1.0 - min_conf)
        scale = max(0.0, min(1.0, scale))
        notional = base * scale
        cap = capital * max_frac
        return max(0.0, min(notional, cap))

    return max(0.0, base)


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = peaks - equity
    return float(drawdowns.max())


def _risk_adjusted_stats(
    pnl: np.ndarray,
    equity: np.ndarray,
    initial_capital: float,
    trades_per_year: float,
) -> Dict[str, float]:
    """Compute Sharpe, Sortino, Calmar, profit factor from per-trade PnL."""
    n = pnl.size
    if n == 0:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
        }
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())
    if std_pnl <= 0:
        sharpe = 0.0
    else:
        sharpe = (mean_pnl / std_pnl) * (trades_per_year ** 0.5)

    negatives = pnl[pnl < 0]
    if negatives.size == 0:
        sortino = 99.0 if mean_pnl > 0 else 0.0
    else:
        down_dev = float(negatives.std())
        if down_dev <= 0:
            sortino = 99.0 if mean_pnl > 0 else 0.0
        else:
            sortino = (mean_pnl / down_dev) * (trades_per_year ** 0.5)

    max_dd = _max_drawdown(equity)
    if max_dd <= 0:
        calmar = 0.0
    else:
        total_return = (equity[-1] - initial_capital) / max(initial_capital, 1e-12)
        annualized_return = total_return * (trades_per_year / max(n, 1))
        calmar = annualized_return / max_dd

    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(np.abs(pnl[pnl < 0].sum()))
    if gross_loss <= 0:
        profit_factor = 99.0 if gross_profit > 0 else 0.0
    else:
        profit_factor = min(99.0, gross_profit / gross_loss)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "profit_factor": profit_factor,
    }


def run_signal_backtest(
    signals: Iterable[Signal],
    realized_returns: np.ndarray,
    risk: RiskConfig,
    trades_per_year: Optional[float] = None,
    session_days_per_year: float = 252.0,
    vol_at_signal: Optional[np.ndarray] = None,
    win_rate: Optional[float] = None,
    avg_win: Optional[float] = None,
    avg_loss: Optional[float] = None,
) -> dict[str, object]:
    """
    Backtest signal stream using realized returns in decimal form.

    The `realized_returns` array must be aligned with `signals`, where each value
    is the forward return over the signal horizon (for example, +0.0025 for +0.25%).

    Optional `trades_per_year` is used to annualize Sharpe/Sortino/Calmar; if None,
    defaults to session_days_per_year * 5 (roughly 5 trades per day).

    Optional `vol_at_signal` (aligned with signals) is used for vol_target sizing.
    Optional `win_rate`, `avg_win`, `avg_loss` are used for kelly sizing.
    """
    tpy = trades_per_year if trades_per_year is not None else DEFAULT_TRADES_PER_YEAR

    sig_list: List[Signal] = list(signals)
    n_sig = len(sig_list)
    if n_sig != int(realized_returns.size):
        raise ValueError("signals and realized_returns must have the same length")
    if vol_at_signal is not None and vol_at_signal.size != n_sig:
        raise ValueError("vol_at_signal must have the same length as signals")

    trades: List[Trade] = []
    equity_points: List[float] = []
    capital = float(risk.initial_capital)
    open_positions = 0

    for idx, sig in enumerate(sig_list):
        if sig.direction == SignalDirection.FLAT:
            continue
        if sig.confidence < float(risk.min_confidence):
            continue
        if open_positions >= int(max(1, risk.max_concurrent_positions)):
            continue

        vol_val: Optional[float] = float(vol_at_signal[idx]) if vol_at_signal is not None else None
        notional = _notional_for_sizing(
            capital,
            risk,
            float(sig.confidence),
            vol_val,
            win_rate,
            avg_win,
            avg_loss,
        )
        if notional <= 0:
            continue

        forward_return = float(realized_returns[idx])
        directional_return = forward_return if sig.direction == SignalDirection.LONG else -forward_return

        gross_pnl = notional * directional_return
        fee_cost = float(risk.fee_per_trade)
        spread_cost = notional * (float(getattr(risk, "spread_bps", 0.0)) / 10_000.0)
        slip_bps = float(risk.slippage_bps)
        if vol_val is not None and getattr(risk, "vol_slippage_coef", 0.0) > 0 and vol_val > 0:
            slip_bps += float(risk.vol_slippage_coef) * vol_val
        if getattr(risk, "impact_coef", 0.0) > 0:
            adv = max(1.0, float(getattr(risk, "adv", 0.0)))
            if adv <= 0:
                slip_bps += float(risk.impact_coef) * (notional ** 0.5) / (1e6 ** 0.5)
            else:
                slip_bps += float(risk.impact_coef) * (notional / adv) ** 0.5
        slip_cost = notional * (slip_bps / 10_000.0)
        total_cost = fee_cost + spread_cost + slip_cost
        net_pnl = gross_pnl - total_cost
        capital += net_pnl
        open_positions = 0

        trade = Trade(
            symbol=sig.symbol,
            source=sig.source,
            entry_time=sig.timestamp,
            exit_time=sig.timestamp,
            direction=sig.direction,
            confidence=float(sig.confidence),
            size=notional,
            gross_pnl=gross_pnl,
            fees=fee_cost,
            slippage=spread_cost + slip_cost,
            net_pnl=net_pnl,
            return_pct=directional_return,
        )
        trades.append(trade)
        equity_points.append(capital)

    pnl = np.asarray([t.net_pnl for t in trades], dtype=np.float32)
    wins = np.asarray([1.0 if t.net_pnl > 0 else 0.0 for t in trades], dtype=np.float32)
    equity = np.asarray(equity_points, dtype=np.float32)
    max_dd = _max_drawdown(equity)

    stats: Dict[str, object] = {
        "trade_count": int(len(trades)),
        "hit_rate": float(wins.mean()) if wins.size else 0.0,
        "avg_pnl": float(pnl.mean()) if pnl.size else 0.0,
        "total_pnl": float(pnl.sum()) if pnl.size else 0.0,
        "max_drawdown": max_dd,
        "ending_capital": float(capital),
    }
    risk_stats = _risk_adjusted_stats(pnl, equity, float(risk.initial_capital), tpy)
    stats.update(risk_stats)

    return {
        "trades": trades,
        "stats": stats,
        "equity_curve": equity.tolist(),
    }

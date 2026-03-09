"""Trading strategy domain models and simple backtest utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Iterable, List

import numpy as np


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


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = peaks - equity
    return float(drawdowns.max())


def run_signal_backtest(
    signals: Iterable[Signal],
    realized_returns: np.ndarray,
    risk: RiskConfig,
) -> dict[str, object]:
    """
    Backtest signal stream using realized returns in decimal form.

    The `realized_returns` array must be aligned with `signals`, where each value
    is the forward return over the signal horizon (for example, +0.0025 for +0.25%).
    """

    sig_list: List[Signal] = list(signals)
    if len(sig_list) != int(realized_returns.size):
        raise ValueError("signals and realized_returns must have the same length")

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

        forward_return = float(realized_returns[idx])
        directional_return = forward_return if sig.direction == SignalDirection.LONG else -forward_return

        notional = max(0.0, capital * float(risk.risk_per_trade))
        gross_pnl = notional * directional_return
        fee_cost = float(risk.fee_per_trade)
        slip_cost = notional * (float(risk.slippage_bps) / 10_000.0)
        net_pnl = gross_pnl - fee_cost - slip_cost
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
            slippage=slip_cost,
            net_pnl=net_pnl,
            return_pct=directional_return,
        )
        trades.append(trade)
        equity_points.append(capital)

    pnl = np.asarray([t.net_pnl for t in trades], dtype=np.float32)
    wins = np.asarray([1.0 if t.net_pnl > 0 else 0.0 for t in trades], dtype=np.float32)
    equity = np.asarray(equity_points, dtype=np.float32)

    return {
        "trades": trades,
        "stats": {
            "trade_count": int(len(trades)),
            "hit_rate": float(wins.mean()) if wins.size else 0.0,
            "avg_pnl": float(pnl.mean()) if pnl.size else 0.0,
            "total_pnl": float(pnl.sum()) if pnl.size else 0.0,
            "max_drawdown": _max_drawdown(equity),
            "ending_capital": float(capital),
        },
        "equity_curve": equity.tolist(),
    }

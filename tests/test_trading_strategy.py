"""Tests for trading strategy domain backtest metrics."""

from datetime import datetime

import numpy as np

from backend.services.trading.strategy import RiskConfig, Signal, SignalDirection, run_signal_backtest


def test_run_signal_backtest_outputs_expected_stats():
    signals = [
        Signal(
            timestamp=datetime(2025, 1, 1, 9, 30),
            symbol="MNQ1!",
            source="test",
            direction=SignalDirection.LONG,
            confidence=0.8,
            horizon_minutes=60,
        ),
        Signal(
            timestamp=datetime(2025, 1, 1, 10, 30),
            symbol="MNQ1!",
            source="test",
            direction=SignalDirection.SHORT,
            confidence=0.9,
            horizon_minutes=60,
        ),
    ]
    returns = np.asarray([0.01, -0.02], dtype=np.float32)
    result = run_signal_backtest(signals, returns, RiskConfig(initial_capital=10000, risk_per_trade=0.1))
    stats = result["stats"]
    assert stats["trade_count"] == 2
    assert stats["hit_rate"] == 1.0
    assert stats["ending_capital"] > 10000.0
    assert len(result["trades"]) == 2
    assert len(result["equity_curve"]) == 2


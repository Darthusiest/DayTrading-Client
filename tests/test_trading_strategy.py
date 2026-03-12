"""Tests for trading strategy domain backtest metrics."""

from datetime import datetime

import numpy as np

from backend.services.trading.strategy import (
    RiskConfig,
    Signal,
    SignalDirection,
    run_signal_backtest,
)


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
    # Risk-adjusted metrics (P1)
    assert "sharpe" in stats and "sortino" in stats and "calmar" in stats and "profit_factor" in stats
    assert stats["profit_factor"] > 0  # one win, one loss -> gross profit / gross loss > 0


def test_vol_target_sizing_uses_vol_at_signal():
    """With vol_target sizing, notional differs from fixed when vol varies."""
    signals = [
        Signal(datetime(2025, 1, 1, 9, 30), "MNQ1!", "test", SignalDirection.LONG, 0.8, 60),
        Signal(datetime(2025, 1, 1, 10, 30), "MNQ1!", "test", SignalDirection.LONG, 0.8, 60),
    ]
    returns = np.asarray([0.01, 0.01], dtype=np.float32)
    vol_low_high = np.asarray([0.005, 0.02], dtype=np.float32)  # 0.5% vs 2% vol
    risk_fixed = RiskConfig(initial_capital=10_000, risk_per_trade=0.01, sizing_method="fixed")
    risk_vol = RiskConfig(initial_capital=10_000, risk_per_trade=0.01, sizing_method="vol_target")
    res_fixed = run_signal_backtest(signals, returns, risk_fixed)
    res_vol = run_signal_backtest(signals, returns, risk_vol, vol_at_signal=vol_low_high)
    sizes_fixed = [t.size for t in res_fixed["trades"]]
    sizes_vol = [t.size for t in res_vol["trades"]]
    assert sizes_fixed[0] == sizes_fixed[1]
    assert sizes_vol[0] != sizes_vol[1]
    assert sizes_vol[0] > sizes_vol[1]


def test_spread_and_impact_increase_costs():
    """With spread_bps and impact_coef, total costs increase and net_pnl decreases."""
    signals = [
        Signal(datetime(2025, 1, 1, 9, 30), "MNQ1!", "test", SignalDirection.LONG, 0.8, 60),
        Signal(datetime(2025, 1, 1, 10, 30), "MNQ1!", "test", SignalDirection.LONG, 0.8, 60),
    ]
    returns = np.asarray([0.01, 0.01], dtype=np.float32)
    risk_baseline = RiskConfig(initial_capital=10_000, risk_per_trade=0.1, fee_per_trade=0, slippage_bps=0)
    risk_costly = RiskConfig(
        initial_capital=10_000,
        risk_per_trade=0.1,
        fee_per_trade=0,
        slippage_bps=0,
        spread_bps=5.0,
        impact_coef=0.1,
        adv=1e6,
    )
    res_baseline = run_signal_backtest(signals, returns, risk_baseline)
    res_costly = run_signal_backtest(signals, returns, risk_costly)
    total_cost_baseline = sum(t.fees + t.slippage for t in res_baseline["trades"])
    total_cost_costly = sum(t.fees + t.slippage for t in res_costly["trades"])
    assert total_cost_costly > total_cost_baseline
    assert res_costly["stats"]["total_pnl"] < res_baseline["stats"]["total_pnl"]


"""Backtest statistics: risk-adjusted metrics, bootstrap p-values, regime breakdown."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def sharpe(pnl: np.ndarray, trades_per_year: float) -> float:
    """Annualized Sharpe ratio from per-trade PnL."""
    if pnl.size == 0:
        return 0.0
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())
    if std_pnl <= 0:
        return 0.0
    return (mean_pnl / std_pnl) * (trades_per_year ** 0.5)


def sortino(pnl: np.ndarray, trades_per_year: float) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    if pnl.size == 0:
        return 0.0
    mean_pnl = float(pnl.mean())
    negatives = pnl[pnl < 0]
    if negatives.size == 0:
        return 99.0 if mean_pnl > 0 else 0.0
    down_dev = float(negatives.std())
    if down_dev <= 0:
        return 99.0 if mean_pnl > 0 else 0.0
    return (mean_pnl / down_dev) * (trades_per_year ** 0.5)


def profit_factor(pnl: np.ndarray) -> float:
    """Gross profit / gross loss; 99 if no losses, 0 if no wins."""
    if pnl.size == 0:
        return 0.0
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(np.abs(pnl[pnl < 0].sum()))
    if gross_loss <= 0:
        return 99.0 if gross_profit > 0 else 0.0
    return min(99.0, gross_profit / gross_loss)


def bootstrap_pnl_pvalue(
    pnl: np.ndarray,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """
    Permutation p-value: fraction of bootstrap samples with total PnL >= observed.
    For negative observed PnL, p-value = fraction with total PnL <= observed.
    """
    if pnl.size == 0:
        return 0.5
    rng = np.random.default_rng(seed)
    observed = float(pnl.sum())
    boot_totals: List[float] = []
    for _ in range(n_bootstrap):
        shuffled = rng.permutation(pnl)
        boot_totals.append(float(shuffled.sum()))
    boot_totals_arr = np.asarray(boot_totals)
    if observed >= 0:
        pvalue = float((boot_totals_arr >= observed).mean())
    else:
        pvalue = float((boot_totals_arr <= observed).mean())
    return pvalue


def bootstrap_sharpe_pvalue(
    pnl: np.ndarray,
    trades_per_year: float,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """P-value that observed Sharpe is above the permutation distribution."""
    if pnl.size == 0:
        return 0.5
    rng = np.random.default_rng(seed)
    observed_sharpe = sharpe(pnl, trades_per_year)
    boot_sharpes: List[float] = []
    for _ in range(n_bootstrap):
        shuffled = rng.permutation(pnl)
        boot_sharpes.append(sharpe(shuffled, trades_per_year))
    boot_arr = np.asarray(boot_sharpes)
    pvalue = float((boot_arr >= observed_sharpe).mean())
    return pvalue


def drawdown_distribution(
    pnl: np.ndarray,
    initial_capital: float = 1.0,
    n_shuffle: int = 500,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Shuffle order of trades, recompute equity and max drawdown each time.
    Returns array of max drawdowns (length n_shuffle).
    """
    if pnl.size == 0:
        return np.array([0.0] * n_shuffle)
    rng = np.random.default_rng(seed)
    drawdowns: List[float] = []
    for _ in range(n_shuffle):
        perm = rng.permutation(pnl)
        equity = np.concatenate([[initial_capital], np.cumsum(perm) + initial_capital])
        peaks = np.maximum.accumulate(equity)
        dd = float((peaks - equity).max())
        drawdowns.append(dd)
    return np.asarray(drawdowns)


def stats_by_regime(
    pnl: np.ndarray,
    win: np.ndarray,
    regime_label: np.ndarray,
    trades_per_year: float = 1260.0,
) -> Dict[str, Dict[str, float]]:
    """
    Per-regime stats. regime_label should be int (e.g. 0=low, 1=mid, 2=high vol).
    Returns dict mapping regime key (e.g. "0", "1", "2") to {trade_count, hit_rate, total_pnl, sharpe, ...}.
    """
    out: Dict[str, Dict[str, float]] = {}
    unique = np.unique(regime_label)
    for r in unique:
        mask = regime_label == r
        if not np.any(mask):
            continue
        p = pnl[mask]
        w = win[mask]
        key = str(int(r))
        n = int(p.size)
        tpy_scale = trades_per_year * (n / max(pnl.size, 1)) if pnl.size else trades_per_year
        out[key] = {
            "trade_count": float(n),
            "hit_rate": float(w.mean()) if w.size else 0.0,
            "total_pnl": float(p.sum()),
            "avg_pnl": float(p.mean()) if p.size else 0.0,
            "sharpe": sharpe(p, tpy_scale),
            "sortino": sortino(p, tpy_scale),
            "profit_factor": profit_factor(p),
        }
    return out

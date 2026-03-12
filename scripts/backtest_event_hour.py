"""
Backtest event-hour continuation/reversal models on cached event dataset.

Look-ahead discipline: (1) Event features use only bars up to the event bar.
(2) Labels are forward returns after the event (e.g. 60m). (3) Backtest uses
realized forward return as outcome (entry at signal time, exit at signal + horizon).
No feature uses data from after signal time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from backend.config.settings import settings
from backend.services.ml.event_hour import EventHourLSTM, EventHourModelConfig
from backend.services.ml.model_registry import load_json_artifact, load_torch_model
from backend.services.trading.backtest_stats import (
    bootstrap_pnl_pvalue,
    drawdown_distribution,
    stats_by_regime,
)
from backend.services.trading.strategy import RiskConfig, Signal, SignalDirection, run_signal_backtest

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVENT_GROUP = {
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


def _load_model(model_name: str, input_size: int) -> EventHourLSTM:
    model = load_torch_model(
        model_name,
        model_builder=lambda: EventHourLSTM(EventHourModelConfig(input_size=input_size)),
        version_key=f"input={input_size}",
        strict=True,
    ).to(DEVICE)
    model.eval()
    return model


def _predict_probs(
    model: EventHourLSTM,
    sequences: torch.Tensor,
    event_types: torch.Tensor,
    calibration: dict,
    batch_size: int,
) -> np.ndarray:
    ds = TensorDataset(sequences, event_types)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out: List[np.ndarray] = []
    method = str(calibration.get("method", "temperature"))
    with torch.no_grad():
        for seq, et in loader:
            seq = seq.to(DEVICE)
            et = et.to(DEVICE)
            logits = model(seq, event_type=et)
            if method.lower() == "platt":
                a = float(calibration.get("A", 1.0))
                b = float(calibration.get("B", 0.0))
                scaled = a * logits + b
            else:
                t = max(1e-6, float(calibration.get("temperature", 1.0)))
                scaled = logits / t
            probs = torch.sigmoid(scaled).detach().cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0) if out else np.asarray([], dtype=np.float32)


def run_backtest(
    prefix: str = "event_hour_backtest",
    batch_size: int = 512,
    risk: RiskConfig | None = None,
    min_confidence: float = 0.0,
    min_action_margin: float = 0.0,
    min_continuation_prob: float = 0.0,
    min_reversal_prob: float = 0.0,
    trades_per_year: float | None = None,
    sizing_method: str = "fixed",
    kelly_fraction: float = 0.5,
    signal_mode: str = "default",
    ev_threshold: float = 0.0,
    validation_stats_path: str | None = None,
) -> dict[str, object]:
    cache_path = settings.MODELS_DIR / "event_hour_dataset.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing dataset cache: {cache_path}. Run train_event_hour_models.py first.")

    data = torch.load(cache_path, map_location="cpu")
    sequences = data["sequences"].float()
    event_type = data["event_type"].long()
    event_dir = data.get("event_dir")
    targets_cont = data["targets_cont"].float().numpy()
    targets_rev = data["targets_rev"].float().numpy()
    forward_returns = data.get("forward_return_60m")
    if forward_returns is not None:
        realized_returns = forward_returns.float().numpy()
    else:
        # Fallback for legacy caches that only contain binary labels.
        realized_returns = np.where(targets_cont >= 0.5, 0.0025, -0.0025).astype(np.float32)
    if event_dir is None:
        # Legacy fallback: odd ids are up events, even ids are down events.
        event_dir_np = np.asarray([(int(et.item()) % 2 == 1) for et in event_type], dtype=np.int64)
    else:
        event_dir_np = event_dir.long().numpy()

    atr_pct = data.get("atr_pct")
    if atr_pct is not None:
        atr_pct = atr_pct.float().numpy()
    else:
        atr_pct = None

    input_size = int(sequences.size(2))
    cont_model = _load_model("event_hour_continuation", input_size=input_size)
    rev_model = _load_model("event_hour_reversal", input_size=input_size)

    cont_metrics = load_json_artifact(settings.MODELS_DIR / "event_hour_continuation_metrics.json", default={})
    rev_metrics = load_json_artifact(settings.MODELS_DIR / "event_hour_reversal_metrics.json", default={})
    cont_cal = load_json_artifact(settings.MODELS_DIR / "event_hour_continuation_calibration.json", default={})
    rev_cal = load_json_artifact(settings.MODELS_DIR / "event_hour_reversal_calibration.json", default={})

    cont_threshold = float(cont_metrics.get("best_threshold", 0.5))
    rev_threshold = float(rev_metrics.get("best_threshold", 0.5))
    cont_cal = cont_cal or {"method": "temperature", "temperature": 1.0}
    rev_cal = rev_cal or {"method": "temperature", "temperature": 1.0}

    cont_probs = _predict_probs(cont_model, sequences, event_type, cont_cal, batch_size)
    rev_probs = _predict_probs(rev_model, sequences, event_type, rev_cal, batch_size)

    ev_stats: dict | None = None
    if signal_mode == "ev" and validation_stats_path:
        p = Path(validation_stats_path)
        if not p.is_absolute():
            p = settings.MODELS_DIR / validation_stats_path
        if p.exists():
            with p.open() as f:
                ev_stats = json.load(f)
            ev_stats = {k: float(v) for k, v in ev_stats.items() if k in ("p_win", "avg_win", "avg_loss")}
        if not ev_stats or set(ev_stats.keys()) != {"p_win", "avg_win", "avg_loss"}:
            ev_stats = None

    action = np.full(cont_probs.shape[0], "none", dtype=object)
    reason_counts: Counter[str] = Counter()
    for i in range(action.size):
        if event_dir_np[i] not in (0, 1):
            reason_counts["invalid_event_dir"] += 1
            continue
        if signal_mode == "combined":
            direction_score = float(cont_probs[i] - rev_probs[i])
            conviction = abs(direction_score)
            confidence = conviction
            if confidence < min_confidence:
                reason_counts["below_confidence"] += 1
                continue
            if conviction < min_action_margin:
                reason_counts["below_margin"] += 1
                continue
            action[i] = "continuation" if direction_score >= 0 else "reversal"
            continue
        confidence = float(max(cont_probs[i], rev_probs[i]))
        if confidence < min_confidence:
            reason_counts["below_confidence"] += 1
            continue
        c_hit = (cont_probs[i] >= cont_threshold) and (cont_probs[i] >= min_continuation_prob)
        r_hit = (rev_probs[i] >= rev_threshold) and (rev_probs[i] >= min_reversal_prob)
        if not c_hit and not r_hit:
            reason_counts["below_model_threshold"] += 1
            continue
        margin = float(abs(cont_probs[i] - rev_probs[i]))
        if margin < min_action_margin:
            reason_counts["below_margin"] += 1
            continue
        if c_hit and (not r_hit or cont_probs[i] >= rev_probs[i]):
            action[i] = "continuation"
        elif r_hit:
            action[i] = "reversal"
        else:
            reason_counts["unresolved_action"] += 1
        if signal_mode == "ev" and ev_stats and action[i] != "none":
            p_win = ev_stats["p_win"]
            avg_win = ev_stats["avg_win"]
            avg_loss = ev_stats["avg_loss"]
            ev = p_win * avg_win - (1.0 - p_win) * avg_loss
            if ev <= ev_threshold:
                action[i] = "none"
                reason_counts["ev_below_threshold"] = reason_counts.get("ev_below_threshold", 0) + 1

    ts0 = datetime(2020, 1, 1)
    signals: List[Signal] = []
    for i, a in enumerate(action):
        if a == "none":
            continue
        # event_dir uses 1=up, 0=down; continuation follows event direction, reversal opposes it.
        if a == "continuation":
            direction = SignalDirection.LONG if event_dir_np[i] == 1 else SignalDirection.SHORT
        else:
            direction = SignalDirection.SHORT if event_dir_np[i] == 1 else SignalDirection.LONG
        action_margin = float(abs(cont_probs[i] - rev_probs[i]))
        confidence = float(abs(cont_probs[i] - rev_probs[i])) if signal_mode == "combined" else float(max(cont_probs[i], rev_probs[i]))
        signals.append(
            Signal(
                timestamp=ts0 + timedelta(minutes=i),
                symbol="MNQ1!",
                source="event_hour_model",
                direction=direction,
                confidence=confidence,
                horizon_minutes=60,
                metadata={
                    "event_type": int(event_type[i].item()),
                    "event_dir": int(event_dir_np[i]),
                    "continuation_prob": float(cont_probs[i]),
                    "reversal_prob": float(rev_probs[i]),
                    "action": str(a),
                    "action_margin": action_margin,
                },
            )
        )

    traded_mask = action != "none"
    traded_returns = realized_returns[traded_mask]
    risk_config = risk or RiskConfig()
    risk_config.sizing_method = sizing_method
    risk_config.kelly_fraction = kelly_fraction
    vol_at_signal_arr = None
    if atr_pct is not None and traded_mask.any():
        vol_at_signal_arr = atr_pct[traded_mask]
    bt = run_signal_backtest(
        signals,
        traded_returns,
        risk=risk_config,
        trades_per_year=trades_per_year,
        vol_at_signal=vol_at_signal_arr,
    )
    trade_objs = bt["trades"]
    stats = bt["stats"]
    equity = np.asarray(bt["equity_curve"], dtype=np.float32)
    trades = int(stats["trade_count"])
    trade_pnl = np.asarray([t.net_pnl for t in trade_objs], dtype=np.float32)
    win = np.asarray([1.0 if t.net_pnl > 0 else 0.0 for t in trade_objs], dtype=np.float32)

    per_group: Dict[str, Dict[str, float]] = {}
    et_np = event_type.numpy()
    for g in ("PDH_PDL", "ORB", "BOS", "ATR", "IMP"):
        mask = np.array([EVENT_GROUP.get(int(et), "OTHER") == g for et in et_np]) & traded_mask
        n = int(mask.sum())
        if n == 0:
            continue
        group_indices = np.where(mask)[0]
        group_pnl = []
        group_win = []
        for gi in group_indices:
            if action[gi] == "none":
                continue
            if action[gi] == "continuation":
                ok = bool(targets_cont[gi] >= 0.5)
            else:
                ok = bool(targets_rev[gi] >= 0.5)
            group_win.append(1.0 if ok else 0.0)
            group_pnl.append(1.0 if ok else -1.0)
        gpnl = np.asarray(group_pnl, dtype=np.float32)
        gwin = np.asarray(group_win, dtype=np.float32)
        per_group[g] = {
            "trades": float(gpnl.size),
            "hit_rate": float(gwin.mean()),
            "avg_pnl": float(gpnl.mean()),
            "total_pnl": float(gpnl.sum()),
        }

    # Robustness: bootstrap p-value, drawdown distribution, regime stats (P4)
    bootstrap_pvalue: float = 0.5
    drawdown_95pct: float = 0.0
    regime_stats: Dict[str, Dict[str, float]] = {}
    if trades >= 2:
        bootstrap_pvalue = bootstrap_pnl_pvalue(trade_pnl, n_bootstrap=500, seed=42)
        dd_dist = drawdown_distribution(
            trade_pnl,
            initial_capital=float(risk_config.initial_capital),
            n_shuffle=500,
            seed=42,
        )
        drawdown_95pct = float(np.percentile(dd_dist, 95))
        if atr_pct is not None and traded_mask.any():
            ap = atr_pct[traded_mask]
            p33, p66 = np.percentile(ap, 33), np.percentile(ap, 66)
            regime_label = np.zeros(ap.size, dtype=np.int32)
            regime_label[ap <= p33] = 0
            regime_label[(ap > p33) & (ap < p66)] = 1
            regime_label[ap >= p66] = 2
            regime_stats = stats_by_regime(trade_pnl, win, regime_label, trades_per_year=1260.0)
            regime_stats = {"low_vol": regime_stats.get("0", {}), "mid_vol": regime_stats.get("1", {}), "high_vol": regime_stats.get("2", {})}

    summary = {
        "dataset_path": str(cache_path),
        "samples": int(action.size),
        "trades": trades,
        "trade_rate": float(trades / max(1, int(action.size))),
        "signals_filtered": int((action == "none").sum()),
        "filter_reason_counts": dict(reason_counts),
        "min_confidence": float(min_confidence),
        "min_action_margin": float(min_action_margin),
        "min_continuation_prob": float(min_continuation_prob),
        "min_reversal_prob": float(min_reversal_prob),
        "continuation_threshold": cont_threshold,
        "reversal_threshold": rev_threshold,
        "continuation_calibration": cont_cal,
        "reversal_calibration": rev_cal,
        "hit_rate": float(stats["hit_rate"]),
        "avg_pnl": float(stats["avg_pnl"]),
        "total_pnl": float(stats["total_pnl"]),
        "max_drawdown": float(stats["max_drawdown"]),
        "ending_capital": float(stats["ending_capital"]),
        "sharpe": float(stats.get("sharpe", 0.0)),
        "sortino": float(stats.get("sortino", 0.0)),
        "calmar": float(stats.get("calmar", 0.0)),
        "profit_factor": float(stats.get("profit_factor", 0.0)),
        "continuation_trades": int((action == "continuation").sum()),
        "reversal_trades": int((action == "reversal").sum()),
        "by_event_group": per_group,
        "bootstrap_pnl_pvalue": bootstrap_pvalue,
        "drawdown_95pct": drawdown_95pct,
        "stats_by_regime": regime_stats,
        "signal_mode": signal_mode,
    }
    if signal_mode == "ev":
        summary["ev_threshold"] = ev_threshold

    # OOS by period: last 25% of trades as "recent" out-of-sample
    if trades >= 4:
        last_q = trades // 4
        last_q_pnl = trade_pnl[-last_q:]
        last_q_win = win[-last_q:]
        summary["last_quarter_trades"] = last_q
        summary["last_quarter_pnl"] = float(last_q_pnl.sum())
        summary["last_quarter_hit_rate"] = float(last_q_win.mean())
    else:
        summary["last_quarter_trades"] = 0
        summary["last_quarter_pnl"] = 0.0
        summary["last_quarter_hit_rate"] = 0.0

    out_json = settings.MODELS_DIR / f"{prefix}_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    out_trades = settings.MODELS_DIR / f"{prefix}_trades.json"
    with out_trades.open("w") as f:
        json.dump(
            [
                {
                    "symbol": t.symbol,
                    "source": t.source,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "direction": t.direction.value,
                    "confidence": t.confidence,
                    "size": t.size,
                    "gross_pnl": t.gross_pnl,
                    "fees": t.fees,
                    "slippage": t.slippage,
                    "net_pnl": t.net_pnl,
                    "return_pct": t.return_pct,
                }
                for t in trade_objs
            ],
            f,
            indent=2,
        )

    # Compute drawdown and chart data
    drawdown = np.zeros_like(equity)
    if trades:
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
    chart_data = {
        "trade_count": int(trades),
        "equity_curve": equity.tolist(),
        "drawdown": drawdown.tolist(),
        "trade_pnl": trade_pnl.tolist(),
        "action_counts": {"continuation": int((action == "continuation").sum()), "reversal": int((action == "reversal").sum())},
        "by_event_group": {g: {k: float(v) if isinstance(v, (int, float)) else v for k, v in d.items()} for g, d in per_group.items()},
        "bootstrap_pnl_pvalue": bootstrap_pvalue,
        "drawdown_95pct": drawdown_95pct,
        "stats_by_regime": regime_stats,
    }
    chart_data_path = settings.MODELS_DIR / f"{prefix}_chart_data.json"
    with chart_data_path.open("w") as f:
        json.dump(chart_data, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax0, ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    if trades:
        ax0.plot(np.arange(1, trades + 1), equity, color="steelblue", linewidth=1.5)
    ax0.set_title("Equity curve")
    ax0.set_xlabel("Trade #")
    ax0.set_ylabel("Cumulative PnL (unit)")
    ax0.grid(True, alpha=0.3)

    if trades:
        ax1.fill_between(np.arange(1, trades + 1), 0, -drawdown, color="coral", alpha=0.6)
        ax1.plot(np.arange(1, trades + 1), -drawdown, color="darkred", linewidth=1)
    ax1.set_title("Drawdown")
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("Drawdown (unit)")
    ax1.grid(True, alpha=0.3)

    labels = ["continuation", "reversal"]
    counts = [chart_data["action_counts"]["continuation"], chart_data["action_counts"]["reversal"]]
    ax2.bar(labels, counts, color=["#4e79a7", "#f28e2b"], edgecolor="black", linewidth=0.5)
    ax2.set_title("Trade counts by action")
    ax2.set_ylabel("Trades")
    ax2.grid(True, axis="y", alpha=0.3)

    if per_group:
        grp_names = list(per_group.keys())
        grp_trades = [int(per_group[g]["trades"]) for g in grp_names]
        grp_pnl = [per_group[g]["total_pnl"] for g in grp_names]
        grp_hr = [per_group[g]["hit_rate"] * 100 for g in grp_names]
        x = np.arange(len(grp_names))
        bars = ax3.bar(x - 0.2, grp_pnl, 0.35, label="Total PnL", color="#4e79a7")
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x + 0.2, grp_hr, 0.35, label="Hit rate %", color="#f28e2b", alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(grp_names)
        ax3.set_ylabel("Total PnL (unit)")
        ax3_twin.set_ylabel("Hit rate %")
        ax3.set_title("PnL and hit rate by event group")
        ax3.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax3.grid(True, axis="y", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No event groups", ha="center", va="center", transform=ax3.transAxes)

    plt.tight_layout()
    out_png = settings.MODELS_DIR / f"{prefix}_curves.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "summary": summary,
        "summary_path": str(out_json),
        "trades_path": str(out_trades),
        "curves_path": str(out_png),
        "chart_data_path": str(chart_data_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest event-hour models on cached dataset.")
    parser.add_argument("--prefix", type=str, default="event_hour_backtest", help="Output filename prefix in data/models")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="Risk fraction of capital per trade")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum signal confidence required")
    parser.add_argument("--min-action-margin", type=float, default=0.0, help="Require abs(continuation_prob - reversal_prob) >= this")
    parser.add_argument("--min-continuation-prob", type=float, default=0.0, help="Extra continuation probability floor")
    parser.add_argument("--min-reversal-prob", type=float, default=0.0, help="Extra reversal probability floor")
    parser.add_argument("--fee-per-trade", type=float, default=0.10, help="Flat fee cost per trade")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage in basis points")
    parser.add_argument("--spread-bps", type=float, default=0.0, help="Half-spread cost in bps per round-trip")
    parser.add_argument("--impact-coef", type=float, default=0.0, help="Market-impact slippage coefficient (sqrt(notional/adv))")
    parser.add_argument("--adv", type=float, default=0.0, help="Average daily volume/notional for impact (0 => notional-only)")
    parser.add_argument("--vol-slippage-coef", type=float, default=0.0, help="Slippage scaling by vol_at_signal (e.g. ATR%): slip_bps += coef * vol")
    parser.add_argument("--trades-per-year", type=float, default=None, help="Trades per year for annualization (default: 252*5)")
    parser.add_argument("--sizing-method", type=str, default="fixed", choices=["fixed", "vol_target", "kelly", "confidence_scaled"], help="Position sizing method")
    parser.add_argument("--kelly-fraction", type=float, default=0.5, help="Cap for fractional Kelly when sizing_method=kelly")
    parser.add_argument("--signal-mode", type=str, default="default", choices=["default", "ev", "combined"], help="Signal mode: default, ev (expected value filter), or combined (cont-rev score)")
    parser.add_argument("--ev-threshold", type=float, default=0.0, help="Min expected value when signal_mode=ev")
    parser.add_argument("--validation-stats-path", type=str, default="", help="JSON path with p_win, avg_win, avg_loss for EV filter (relative to data/models or absolute)")
    args = parser.parse_args()
    result = run_backtest(
        prefix=args.prefix,
        batch_size=args.batch_size,
        risk=RiskConfig(
            risk_per_trade=args.risk_per_trade,
            min_confidence=args.min_confidence,
            fee_per_trade=args.fee_per_trade,
            slippage_bps=args.slippage_bps,
            spread_bps=args.spread_bps,
            impact_coef=args.impact_coef,
            adv=args.adv,
            vol_slippage_coef=args.vol_slippage_coef,
        ),
        min_confidence=args.min_confidence,
        min_action_margin=args.min_action_margin,
        min_continuation_prob=args.min_continuation_prob,
        min_reversal_prob=args.min_reversal_prob,
        trades_per_year=args.trades_per_year,
        sizing_method=args.sizing_method,
        kelly_fraction=args.kelly_fraction,
        signal_mode=args.signal_mode,
        ev_threshold=args.ev_threshold,
        validation_stats_path=args.validation_stats_path or None,
    )
    summary = result["summary"]
    print("Saved backtest summary:", result["summary_path"])
    print("Saved backtest trades:", result["trades_path"])
    print("Saved backtest curves:", result["curves_path"])
    print("Saved chart data:", result["chart_data_path"])
    print("Trades:", summary["trades"], "Hit-rate:", f"{summary['hit_rate']:.4f}", "Total PnL:", f"{summary['total_pnl']:.2f}")
    print("Sharpe:", f"{summary.get('sharpe', 0):.4f}", "Sortino:", f"{summary.get('sortino', 0):.4f}", "Calmar:", f"{summary.get('calmar', 0):.4f}", "Profit factor:", f"{summary.get('profit_factor', 0):.4f}")


if __name__ == "__main__":
    main()

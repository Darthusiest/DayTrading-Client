"""Backtest event-hour continuation/reversal models on cached event dataset."""

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
    temperature: float,
    batch_size: int,
) -> np.ndarray:
    ds = TensorDataset(sequences, event_types)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out: List[np.ndarray] = []
    t = max(1e-6, float(temperature))
    with torch.no_grad():
        for seq, et in loader:
            seq = seq.to(DEVICE)
            et = et.to(DEVICE)
            logits = model(seq, event_type=et) / t
            probs = torch.sigmoid(logits).detach().cpu().numpy()
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

    input_size = int(sequences.size(2))
    cont_model = _load_model("event_hour_continuation", input_size=input_size)
    rev_model = _load_model("event_hour_reversal", input_size=input_size)

    cont_metrics = load_json_artifact(settings.MODELS_DIR / "event_hour_continuation_metrics.json", default={})
    rev_metrics = load_json_artifact(settings.MODELS_DIR / "event_hour_reversal_metrics.json", default={})
    cont_cal = load_json_artifact(settings.MODELS_DIR / "event_hour_continuation_calibration.json", default={})
    rev_cal = load_json_artifact(settings.MODELS_DIR / "event_hour_reversal_calibration.json", default={})

    cont_threshold = float(cont_metrics.get("best_threshold", 0.5))
    rev_threshold = float(rev_metrics.get("best_threshold", 0.5))
    cont_temp = float(cont_cal.get("temperature", 1.0))
    rev_temp = float(rev_cal.get("temperature", 1.0))

    cont_probs = _predict_probs(cont_model, sequences, event_type, cont_temp, batch_size)
    rev_probs = _predict_probs(rev_model, sequences, event_type, rev_temp, batch_size)

    action = np.full(cont_probs.shape[0], "none", dtype=object)
    reason_counts: Counter[str] = Counter()
    for i in range(action.size):
        if event_dir_np[i] not in (0, 1):
            reason_counts["invalid_event_dir"] += 1
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
        confidence = float(max(cont_probs[i], rev_probs[i]))
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
    bt = run_signal_backtest(signals, traded_returns, risk=risk or RiskConfig())
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
        "continuation_temperature": cont_temp,
        "reversal_temperature": rev_temp,
        "hit_rate": float(stats["hit_rate"]),
        "avg_pnl": float(stats["avg_pnl"]),
        "total_pnl": float(stats["total_pnl"]),
        "max_drawdown": float(stats["max_drawdown"]),
        "ending_capital": float(stats["ending_capital"]),
        "continuation_trades": int((action == "continuation").sum()),
        "reversal_trades": int((action == "reversal").sum()),
        "by_event_group": per_group,
    }

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

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    if trades:
        ax[0].plot(np.arange(1, trades + 1), equity, color="steelblue", linewidth=1.5)
    ax[0].set_title("Event-hour strategy equity curve")
    ax[0].set_xlabel("Trade #")
    ax[0].set_ylabel("Cumulative PnL (unit)")
    ax[0].grid(True, alpha=0.3)

    labels = ["continuation", "reversal"]
    counts = [int((action == "continuation").sum()), int((action == "reversal").sum())]
    ax[1].bar(labels, counts, color=["#4e79a7", "#f28e2b"], edgecolor="black", linewidth=0.5)
    ax[1].set_title("Trade counts by action")
    ax[1].set_ylabel("Trades")
    ax[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_png = settings.MODELS_DIR / f"{prefix}_curves.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "summary": summary,
        "summary_path": str(out_json),
        "trades_path": str(out_trades),
        "curves_path": str(out_png),
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
    args = parser.parse_args()
    result = run_backtest(
        prefix=args.prefix,
        batch_size=args.batch_size,
        risk=RiskConfig(
            risk_per_trade=args.risk_per_trade,
            min_confidence=args.min_confidence,
            fee_per_trade=args.fee_per_trade,
            slippage_bps=args.slippage_bps,
        ),
        min_confidence=args.min_confidence,
        min_action_margin=args.min_action_margin,
        min_continuation_prob=args.min_continuation_prob,
        min_reversal_prob=args.min_reversal_prob,
    )
    summary = result["summary"]
    print("Saved backtest summary:", result["summary_path"])
    print("Saved backtest trades:", result["trades_path"])
    print("Saved backtest curves:", result["curves_path"])
    print("Trades:", summary["trades"], "Hit-rate:", f"{summary['hit_rate']:.4f}", "Total PnL:", f"{summary['total_pnl']:.2f}")


if __name__ == "__main__":
    main()

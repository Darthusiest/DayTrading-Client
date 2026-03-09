#!/usr/bin/env python3
"""Plot reliability curves for event-hour continuation and reversal models.

Reliability: expected (mean predicted prob) vs observed (actual positive rate)
per probability bin. Well-calibrated models have points on the diagonal.

Usage:
  python scripts/plot_event_calibration.py
  python scripts/plot_event_calibration.py --n-bins 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from backend.config.settings import settings
from backend.services.ml.event_hour import EventHourDataset, EventHourLSTM, EventHourModelConfig


def _load_model(name: str, input_size: int) -> tuple:
    ckpt = settings.MODELS_DIR / f"{name}.pt"
    if not ckpt.is_file():
        return None, None
    config = EventHourModelConfig(input_size=input_size)
    model = EventHourLSTM(config)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model.eval()
    return model, config


def _reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple:
    """Return (bin_centers, mean_pred, frac_pos, counts) per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean_pred = np.zeros(n_bins)
    frac_pos = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if mask.sum() > 0:
            mean_pred[i] = probs[mask].mean()
            frac_pos[i] = y[mask].mean()
            counts[i] = int(mask.sum())
    return bin_centers, mean_pred, frac_pos, counts


def main():
    parser = argparse.ArgumentParser(description="Plot event-hour calibration reliability curves")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    cache_path = settings.MODELS_DIR / "event_hour_dataset.pt"
    if not cache_path.is_file():
        print(f"Missing {cache_path}. Run train_event_hour_models.py first.")
        return 1

    data = torch.load(cache_path, map_location="cpu")
    sequences = data["sequences"].float()
    targets_cont = data["targets_cont"].float().numpy()
    targets_rev = data["targets_rev"].float().numpy()
    event_type = data["event_type"]
    event_dir = data.get("event_dir")
    session_id = data.get("session_id")
    symbol_id = data.get("symbol_id")

    if symbol_id is None:
        symbol_id = torch.zeros_like(session_id)
    if session_id is None:
        session_id = torch.arange(sequences.size(0), dtype=torch.long)

    input_size = int(sequences.size(2))
    cont_model, _ = _load_model("event_hour_continuation", input_size)
    rev_model, _ = _load_model("event_hour_reversal", input_size)

    if cont_model is None or rev_model is None:
        print("Missing model checkpoints. Train first.")
        return 1

    cont_cal = {}
    rev_cal = {}
    try:
        import json
        with (settings.MODELS_DIR / "event_hour_continuation_calibration.json").open() as f:
            cont_cal = json.load(f)
        with (settings.MODELS_DIR / "event_hour_reversal_calibration.json").open() as f:
            rev_cal = json.load(f)
    except Exception:
        pass

    cont_cal = cont_cal or {"method": "temperature", "temperature": 1.0}
    rev_cal = rev_cal or {"method": "temperature", "temperature": 1.0}

    cont_ds = EventHourDataset(sequences, torch.from_numpy(targets_cont), event_type, event_dir, session_id, symbol_id)
    rev_ds = EventHourDataset(sequences, torch.from_numpy(targets_rev), event_type, event_dir, session_id, symbol_id)
    cont_loader = DataLoader(cont_ds, batch_size=512, shuffle=False)
    rev_loader = DataLoader(rev_ds, batch_size=512, shuffle=False)

    def _get_probs(model, loader, cal):
        probs_list, y_list = [], []
        method = str(cal.get("method", "temperature"))
        with torch.no_grad():
            for batch in loader:
                x = batch["sequence"]
                t = batch["target"].numpy()
                et = batch.get("event_type")
                logits = model(x, event_type=et)
                if method.lower() == "platt":
                    a, b = float(cal.get("A", 1.0)), float(cal.get("B", 0.0))
                    scaled = a * logits + b
                else:
                    t_sc = max(1e-6, float(cal.get("temperature", 1.0)))
                    scaled = logits / t_sc
                p = torch.sigmoid(scaled).numpy()
                probs_list.append(p)
                y_list.append(t)
        return np.concatenate(probs_list), np.concatenate(y_list)

    cont_probs, cont_y = _get_probs(cont_model, cont_loader, cont_cal)
    rev_probs, rev_y = _get_probs(rev_model, rev_loader, rev_cal)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, probs, y, title in [
        (axes[0], cont_probs, cont_y, "Continuation"),
        (axes[1], rev_probs, rev_y, "Reversal"),
    ]:
        bc, mean_pred, frac_pos, counts = _reliability_curve(probs, y, n_bins=args.n_bins)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        valid = counts > 0
        ax.plot(mean_pred[valid], frac_pos[valid], "o-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{title} reliability")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = settings.MODELS_DIR / "event_hour_reliability.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # Save chart data for re-plotting
    chart_data = {
        "continuation": {
            "mean_pred": mean_pred[valid].tolist() if isinstance(mean_pred, np.ndarray) else [],
            "frac_pos": frac_pos[valid].tolist() if isinstance(frac_pos, np.ndarray) else [],
            "counts": counts[valid].tolist() if isinstance(counts, np.ndarray) else [],
        },
        "reversal": {"mean_pred": [], "frac_pos": [], "counts": []},
    }
    bc_c, mp_c, fp_c, ct_c = _reliability_curve(cont_probs, cont_y, n_bins=args.n_bins)
    v_c = ct_c > 0
    chart_data["continuation"] = {
        "bin_centers": bc_c[v_c].tolist(),
        "mean_pred": mp_c[v_c].tolist(),
        "frac_pos": fp_c[v_c].tolist(),
        "counts": ct_c[v_c].tolist(),
    }
    bc_r, mp_r, fp_r, ct_r = _reliability_curve(rev_probs, rev_y, n_bins=args.n_bins)
    v_r = ct_r > 0
    chart_data["reversal"] = {
        "bin_centers": bc_r[v_r].tolist(),
        "mean_pred": mp_r[v_r].tolist(),
        "frac_pos": fp_r[v_r].tolist(),
        "counts": ct_r[v_r].tolist(),
    }
    chart_path = settings.MODELS_DIR / "event_hour_reliability_chart_data.json"
    with chart_path.open("w") as f:
        json.dump(chart_data, f, indent=2)

    print(f"Saved {out_path}")
    print(f"Saved chart data {chart_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

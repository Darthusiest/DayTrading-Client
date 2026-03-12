"""
Ablate event-hour feature groups and report validation F1 delta vs baseline.

Feature indices must match the event dataset build in train_event_hour_models.py:
- Bar features from build_session_feature_matrix: 0..36 (base 5 + derived 32).
  Within that: vol_regime at 20, session_phase at 31 (see bar_features.py).
- London fib: 37.
- SMT: 38-43.
- Event-type row: 44-48 (last column 48 = event type encoding).

Run from project root: python -m scripts.ablate_event_hour_features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from backend.config.settings import settings
from backend.services.ml.event_hour import EventHourLSTM, EventHourModelConfig
from backend.services.ml.model_registry import load_json_artifact, load_torch_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Feature index groups (update if event feature build changes)
FEATURE_GROUPS = {
    "london": [37],
    "vol_regime": [20],
    "session_phase": [31],
    "event_type_encoding": [48],
    "smt": list(range(38, 44)),
}


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_bin = (y_pred >= 0.5).astype(np.float32)
    tp = ((y_true >= 0.5) & (y_bin >= 0.5)).sum()
    fp = ((y_true < 0.5) & (y_bin >= 0.5)).sum()
    fn = ((y_true >= 0.5) & (y_bin < 0.5)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec <= 0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def run_ablation(
    model_name: str = "event_hour_reversal",
    n_bootstrap: int = 0,
) -> None:
    cache_path = settings.MODELS_DIR / "event_hour_dataset.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing dataset cache: {cache_path}. Run train_event_hour_models.py first.")
    data = torch.load(cache_path, map_location="cpu")
    sequences = data["sequences"].float()
    event_type = data["event_type"].long()
    targets = data["targets_rev"].float()
    input_size = int(sequences.size(2))

    model = load_torch_model(
        model_name,
        model_builder=lambda: EventHourLSTM(EventHourModelConfig(input_size=input_size)),
        version_key=f"input={input_size}",
        strict=True,
    ).to(DEVICE)
    model.eval()

    cal_path = settings.MODELS_DIR / f"{model_name}_calibration.json"
    cal = load_json_artifact(cal_path, default={}) if cal_path.exists() else {}
    if not cal:
        cal = {"method": "temperature", "temperature": 1.0}

    def eval_sequences(seq: torch.Tensor) -> float:
        ds = TensorDataset(seq, event_type)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        probs_list = []
        with torch.no_grad():
            for batch in loader:
                x, et = batch[0].to(DEVICE), batch[1].to(DEVICE)
                logits = model(x, event_type=et)
                t = max(1e-6, float(cal.get("temperature", 1.0)))
                probs = torch.sigmoid(logits / t).detach().cpu().numpy()
                probs_list.append(probs)
        probs = np.concatenate(probs_list, axis=0)
        y_true = targets.numpy()
        return _f1(y_true, probs)

    baseline_f1 = eval_sequences(sequences)
    print(f"Baseline validation F1 ({model_name}): {baseline_f1:.4f}\n")

    for group_name, indices in FEATURE_GROUPS.items():
        seq_abl = sequences.clone()
        for idx in indices:
            if idx < seq_abl.size(2):
                seq_abl[:, :, idx] = 0.0
        f1_abl = eval_sequences(seq_abl)
        delta = f1_abl - baseline_f1
        print(f"  {group_name}: F1={f1_abl:.4f}  delta={delta:+.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablate event-hour feature groups and report F1 delta.")
    parser.add_argument("--model", type=str, default="event_hour_reversal", choices=["event_hour_continuation", "event_hour_reversal"])
    parser.add_argument("--n-bootstrap", type=int, default=0, help="Not used yet; for future CI of delta")
    args = parser.parse_args()
    run_ablation(model_name=args.model, n_bootstrap=args.n_bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

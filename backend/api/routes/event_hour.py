"""Event-hour inference API endpoints."""

from __future__ import annotations

from typing import Any, Optional

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config.settings import settings
from backend.services.ml.event_hour import EventHourLSTM, EventHourModelConfig
from backend.services.ml.model_registry import load_json_artifact, load_torch_model

router = APIRouter(prefix="/event-hour", tags=["event-hour"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EventHourPredictRequest(BaseModel):
    sequence: list[list[float]] = Field(..., description="Lookback x features sequence.")
    event_type: Optional[int] = Field(default=None, ge=0, le=10)


def _load_model(model_name: str, input_size: int) -> EventHourLSTM:
    try:
        model = load_torch_model(
            model_name,
            model_builder=lambda: EventHourLSTM(EventHourModelConfig(input_size=input_size)),
            version_key=f"input={input_size}",
            strict=True,
        ).to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"{exc}. Train event-hour models first.",
        )


def _load_calibration(model_name: str) -> dict:
    cal_path = settings.MODELS_DIR / f"{model_name}_calibration.json"
    try:
        data = load_json_artifact(cal_path, default={})
        if not data:
            return {"method": "temperature", "temperature": 1.0}
        return data
    except Exception:
        return {"method": "temperature", "temperature": 1.0}


def _load_threshold(model_name: str) -> float:
    mpath = settings.MODELS_DIR / f"{model_name}_metrics.json"
    threshold = 0.5
    try:
        data = load_json_artifact(mpath, default={})
        threshold = float(data.get("best_threshold", 0.5))
    except Exception:
        threshold = 0.5
    return threshold


def _predict_one(model_name: str, sequence: torch.Tensor, event_type: Optional[torch.Tensor]) -> dict[str, float]:
    model = _load_model(model_name, input_size=int(sequence.size(-1)))
    cal = _load_calibration(model_name)
    method = str(cal.get("method", "temperature"))
    with torch.no_grad():
        logits = model(sequence, event_type=event_type)
        if method.lower() == "platt":
            a, b = float(cal.get("A", 1.0)), float(cal.get("B", 0.0))
            scaled = a * logits + b
        else:
            t = max(1e-6, float(cal.get("temperature", 1.0)))
            scaled = logits / t
        prob = torch.sigmoid(scaled).item()
    return {
        "logit": float(logits.item()),
        "calibration": cal,
        "probability": float(prob),
        "threshold": float(_load_threshold(model_name)),
    }


def choose_event_hour_signal(
    continuation_prob: float,
    continuation_threshold: float,
    reversal_prob: float,
    reversal_threshold: float,
) -> str:
    """Choose continuation/reversal/neutral signal based on calibrated thresholds."""
    cont_hit = continuation_prob >= continuation_threshold
    rev_hit = reversal_prob >= reversal_threshold
    if cont_hit and (not rev_hit or continuation_prob >= reversal_prob):
        return "continuation"
    if rev_hit:
        return "reversal"
    return "neutral"


@router.post("/predict")
def predict_event_hour(req: EventHourPredictRequest) -> dict[str, Any]:
    if not req.sequence or not req.sequence[0]:
        raise HTTPException(status_code=400, detail="sequence must be non-empty 2D list")
    x = torch.tensor(req.sequence, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    et = None
    if req.event_type is not None:
        et = torch.tensor([req.event_type], dtype=torch.long, device=DEVICE)

    cont = _predict_one("event_hour_continuation", x, et)
    rev = _predict_one("event_hour_reversal", x, et)

    signal = choose_event_hour_signal(
        continuation_prob=cont["probability"],
        continuation_threshold=cont["threshold"],
        reversal_prob=rev["probability"],
        reversal_threshold=rev["threshold"],
    )

    return {
        "signal": signal,
        "continuation": cont,
        "reversal": rev,
    }


@router.get("/predict-from-cache/{sample_index}")
def predict_from_cache(sample_index: int) -> dict[str, Any]:
    cache_path = settings.MODELS_DIR / "event_hour_dataset.pt"
    if not cache_path.is_file():
        raise HTTPException(status_code=404, detail=f"Missing dataset cache: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")
    seq = data.get("sequences")
    event_type = data.get("event_type")
    targets_cont = data.get("targets_cont")
    targets_rev = data.get("targets_rev")
    if seq is None:
        raise HTTPException(status_code=500, detail="Invalid event_hour_dataset.pt: missing sequences")
    n = int(seq.size(0))
    if sample_index < 0 or sample_index >= n:
        raise HTTPException(status_code=400, detail=f"sample_index out of range [0, {n - 1}]")

    x = seq[sample_index : sample_index + 1].to(DEVICE)
    et = event_type[sample_index : sample_index + 1].to(DEVICE) if event_type is not None else None
    cont = _predict_one("event_hour_continuation", x, et)
    rev = _predict_one("event_hour_reversal", x, et)
    return {
        "sample_index": sample_index,
        "event_type": int(event_type[sample_index].item()) if event_type is not None else None,
        "target_continuation": float(targets_cont[sample_index].item()) if targets_cont is not None else None,
        "target_reversal": float(targets_rev[sample_index].item()) if targets_rev is not None else None,
        "continuation": cont,
        "reversal": rev,
    }

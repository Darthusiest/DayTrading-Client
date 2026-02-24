"""Evaluation and learning metrics API endpoints."""
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from backend.config.settings import settings
from backend.database.db import get_db
from backend.database.models import LearningMetric, TrainingSample
from backend.services.evaluation.learning_tracker import LearningTracker
from backend.services.evaluation.metrics import MetricsCalculator
from backend.services.ml.inference.predictor import Predictor

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/learning-status")
def get_learning_status(db: Session = Depends(get_db)):
    """Get current learning performance metrics."""
    calculator = MetricsCalculator(db)
    metrics = calculator.calculate_learning_status()
    return metrics


@router.get("/learning-curve")
def get_learning_curve(
    metric_type: str = "val_loss",
    days_back: int = 30,
    db: Session = Depends(get_db)
):
    """Get learning curve data for visualization."""
    tracker = LearningTracker(db)
    curve_data = tracker.get_learning_curve(
        metric_type=metric_type,
        days_back=days_back
    )
    return {"metric_type": metric_type, "data": curve_data}


@router.get("/best-model")
def get_best_model_info(db: Session = Depends(get_db)):
    """Get information about the best model."""
    tracker = LearningTracker(db)
    model_info = tracker.get_best_model_info()
    return model_info


@router.get("/learning-plot", response_class=Response)
def get_learning_plot(db: Session = Depends(get_db)):
    """
    Return a PNG plot of training/validation loss and direction accuracy over epochs
    for the most recent training run (from LearningMetric).
    """
    # Latest run = most recent contiguous val_loss by timestamp (until epoch 0)
    val_loss_rows = (
        db.query(LearningMetric)
        .filter(LearningMetric.metric_type == "val_loss")
        .order_by(LearningMetric.timestamp.desc())
        .all()
    )
    if not val_loss_rows:
        raise HTTPException(
            status_code=404,
            detail="No learning metrics found. Train a model first (POST /api/v1/train)."
        )
    seen_epochs = set()
    run_epochs = []
    model_version = val_loss_rows[0].model_version
    for row in val_loss_rows:
        if row.model_version != model_version:
            break
        if row.epoch in seen_epochs:
            continue
        seen_epochs.add(row.epoch)
        run_epochs.append(row.epoch)
        if row.epoch == 0:
            break
    if not run_epochs:
        raise HTTPException(status_code=404, detail="No complete run found in learning metrics.")
    run_epochs = sorted(run_epochs)

    # Load all four series for this run
    types = ("train_loss", "val_loss", "train_direction_accuracy", "val_direction_accuracy")
    rows = (
        db.query(LearningMetric)
        .filter(
            LearningMetric.model_version == model_version,
            LearningMetric.epoch.in_(run_epochs),
            LearningMetric.metric_type.in_(types),
        )
        .all()
    )
    by_epoch = {e: {} for e in run_epochs}
    for r in rows:
        by_epoch[r.epoch][r.metric_type] = r.metric_value

    # Only plot epochs that have at least val_loss
    run_epochs_ok = [e for e in run_epochs if by_epoch[e].get("val_loss") is not None]
    if not run_epochs_ok:
        raise HTTPException(status_code=404, detail="No valid loss metrics found for plotting.")
    train_loss = [by_epoch[e].get("train_loss") or 0.0 for e in run_epochs_ok]
    val_loss = [by_epoch[e].get("val_loss") or 0.0 for e in run_epochs_ok]
    train_acc = [by_epoch[e].get("train_direction_accuracy") or 0.0 for e in run_epochs_ok]
    val_acc = [by_epoch[e].get("val_direction_accuracy") or 0.0 for e in run_epochs_ok]
    epochs_display = [e + 1 for e in run_epochs_ok]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(epochs_display, train_loss, label="Train loss", color="C0")
    ax1.plot(epochs_display, val_loss, label="Validation loss", color="C1")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and validation loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_display, train_acc, label="Train accuracy", color="C0")
    ax2.plot(epochs_display, val_acc, label="Validation accuracy", color="C1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")
    ax2.set_title("Training and validation accuracy (direction)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@router.get("/test-accuracy")
def get_test_accuracy(db: Session = Depends(get_db)):
    """
    Run the best model on the chronological test split of training data and return
    accuracy metrics (MAE, RMSE, direction accuracy). Uses the same split logic as
    training (VALIDATION_SPLIT, TEST_SPLIT).
    """
    samples = (
        db.query(TrainingSample)
        .filter(TrainingSample.actual_price.isnot(None))
        .order_by(TrainingSample.session_date, TrainingSample.id)
        .all()
    )
    if len(samples) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 3 samples with actual_price (have {len(samples)}). Add data and run process-training-data."
        )

    test_ratio = getattr(settings, "TEST_SPLIT", 0.1)
    val_ratio = settings.VALIDATION_SPLIT
    train_ratio = 1.0 - val_ratio - test_ratio
    n = len(samples)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise HTTPException(
            status_code=400,
            detail="No test set after split. Add more samples or reduce VALIDATION_SPLIT/TEST_SPLIT."
        )
    test_samples = samples[n_train + n_val :]

    predictor = Predictor(device="cpu")
    if not predictor.load_model(db=db):
        raise HTTPException(
            status_code=503,
            detail="Could not load best model. Train a model first (POST /api/v1/train)."
        )

    actuals = []
    predicteds = []
    direction_correct = 0
    direction_total = 0
    skipped = 0

    for sample in test_samples:
        path = sample.processed_image_path or (sample.snapshot.image_path if sample.snapshot else None)
        if not path or not Path(path).is_file():
            skipped += 1
            continue
        features = sample.features or {}
        result = predictor.predict(Path(path), expected_price=None, features=features)
        if "error" in result or result.get("predicted_price") is None:
            skipped += 1
            continue
        pred_price = result["predicted_price"]
        actual_price = float(sample.actual_price)
        actuals.append(actual_price)
        predicteds.append(pred_price)
        # Direction accuracy: predicted direction (up/down/sideways) vs actual label
        before_price = actual_price - (sample.price_change_absolute or 0)
        if before_price <= 0:
            before_price = actual_price * 0.999
        pct_change = (pred_price - before_price) / before_price
        threshold = 0.01
        if pct_change > threshold:
            pred_direction = "up"
        elif pct_change < -threshold:
            pred_direction = "down"
        else:
            pred_direction = "sideways"
        actual_direction = (sample.direction or "").lower()
        if actual_direction in ("up", "down", "sideways"):
            direction_correct += 1 if pred_direction == actual_direction else 0
            direction_total += 1

    if not actuals:
        raise HTTPException(
            status_code=502,
            detail=f"No test predictions (all {len(test_samples)} skipped: missing image or prediction failed)."
        )

    n_eval = len(actuals)
    mae = sum(abs(a - p) for a, p in zip(actuals, predicteds)) / n_eval
    rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actuals, predicteds)) / n_eval)
    direction_accuracy = (direction_correct / direction_total) if direction_total else None

    return {
        "test_samples_total": len(test_samples),
        "test_samples_evaluated": n_eval,
        "skipped": skipped,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "direction_accuracy": round(direction_accuracy, 4) if direction_accuracy is not None else None,
        "direction_correct": direction_correct,
        "direction_total": direction_total,
    }

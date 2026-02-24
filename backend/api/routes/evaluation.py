"""Evaluation and learning metrics API endpoints."""
import math
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.config.settings import settings
from backend.database.db import get_db
from backend.database.models import TrainingSample
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

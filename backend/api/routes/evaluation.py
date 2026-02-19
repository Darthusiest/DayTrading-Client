"""Evaluation and learning metrics API endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.services.evaluation.metrics import MetricsCalculator
from backend.services.evaluation.learning_tracker import LearningTracker

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

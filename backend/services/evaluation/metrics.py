"""Evaluation metrics for model learning."""
import logging
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from backend.database.models import Prediction, LearningMetric, ModelCheckpoint
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate learning and performance metrics."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_learning_status(self) -> Dict[str, Any]:
        """Calculate overall learning status and metrics."""
        metrics = {}
        
        # Get latest model checkpoint
        latest_checkpoint = self.db.query(ModelCheckpoint).order_by(
            ModelCheckpoint.created_at.desc()
        ).first()
        
        if latest_checkpoint:
            metrics["model_version"] = latest_checkpoint.version
            metrics["latest_epoch"] = latest_checkpoint.epoch
            metrics["train_loss"] = latest_checkpoint.train_loss
            metrics["val_loss"] = latest_checkpoint.val_loss
            metrics["train_accuracy"] = latest_checkpoint.train_accuracy
            metrics["val_accuracy"] = latest_checkpoint.val_accuracy
        
        # Calculate prediction accuracy from recent predictions
        recent_predictions = self.db.query(Prediction).filter(
            Prediction.actual_price.isnot(None)
        ).order_by(Prediction.created_at.desc()).limit(100).all()
        
        if recent_predictions:
            metrics["prediction_accuracy"] = self._calculate_prediction_accuracy(
                recent_predictions
            )
            metrics["calibration_score"] = self._calculate_calibration(
                recent_predictions
            )
            metrics["direction_accuracy"] = self._calculate_direction_accuracy(
                recent_predictions
            )
            metrics["num_evaluated_predictions"] = len(recent_predictions)
        
        # Learning progress over time
        metrics["learning_progress"] = self._calculate_learning_progress()
        
        return metrics
    
    def _calculate_prediction_accuracy(
        self,
        predictions: List[Prediction]
    ) -> Dict[str, float]:
        """Calculate prediction accuracy metrics."""
        errors = []
        for pred in predictions:
            if pred.actual_price and pred.model_predicted_price:
                error = abs(pred.model_predicted_price - pred.actual_price)
                error_pct = (error / pred.actual_price * 100) if pred.actual_price > 0 else 0
                errors.append(error_pct)
        
        if not errors:
            return {"mae": 0.0, "mae_pct": 0.0, "rmse": 0.0}
        
        errors = np.array(errors)
        return {
            "mae": float(np.mean(errors)),
            "mae_pct": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2)))
        }
    
    def _calculate_calibration(
        self,
        predictions: List[Prediction]
    ) -> Dict[str, float]:
        """Calculate calibration score (how well probabilities match reality)."""
        # Group predictions by probability bins
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_counts = {i: {"predicted": 0, "actual": 0} for i in range(len(bins) - 1)}
        
        for pred in predictions:
            if pred.probability_hit is not None and pred.was_hit is not None:
                prob = pred.probability_hit
                bin_idx = min(int(prob * 5), 4)  # 0-4 bins
                bin_counts[bin_idx]["predicted"] += 1
                if pred.was_hit:
                    bin_counts[bin_idx]["actual"] += 1
        
        # Calculate calibration error
        calibration_errors = []
        for bin_idx, counts in bin_counts.items():
            if counts["predicted"] > 0:
                expected_rate = bins[bin_idx] + (bins[bin_idx + 1] - bins[bin_idx]) / 2
                actual_rate = counts["actual"] / counts["predicted"]
                calibration_errors.append(abs(expected_rate - actual_rate))
        
        return {
            "calibration_error": float(np.mean(calibration_errors)) if calibration_errors else 0.0,
            "bin_counts": {str(k): v for k, v in bin_counts.items()}
        }
    
    def _calculate_direction_accuracy(
        self,
        predictions: List[Prediction]
    ) -> float:
        """Calculate direction prediction accuracy."""
        correct = 0
        total = 0
        
        for pred in predictions:
            if pred.actual_price and pred.model_predicted_price:
                pred_direction = "up" if pred.model_predicted_price > pred.actual_price else "down"
                # Simplified: compare with previous price (would need historical data)
                # For now, use a placeholder
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_learning_progress(self) -> Dict[str, Any]:
        """Calculate learning progress over epochs."""
        metrics = self.db.query(LearningMetric).filter(
            LearningMetric.metric_type == "val_loss"
        ).order_by(LearningMetric.epoch).all()
        
        if not metrics:
            return {"improvement": 0.0, "trend": "unknown"}
        
        losses = [m.metric_value for m in metrics]
        
        # Calculate improvement (first vs last)
        if len(losses) > 1:
            improvement = ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] > 0 else 0.0
        else:
            improvement = 0.0
        
        # Determine trend
        if len(losses) >= 3:
            recent_trend = np.mean(losses[-3:]) < np.mean(losses[-6:-3]) if len(losses) >= 6 else False
            trend = "improving" if recent_trend else "plateauing"
        else:
            trend = "insufficient_data"
        
        return {
            "improvement_pct": float(improvement),
            "trend": trend,
            "num_epochs": len(losses),
            "latest_loss": float(losses[-1]) if losses else None
        }

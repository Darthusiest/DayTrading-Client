"""Track and monitor learning progress."""
import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from backend.database.models import LearningMetric, ModelCheckpoint
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LearningTracker:
    """Track learning progress and performance."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_learning_curve(
        self,
        model_version: str = None,
        metric_type: str = "val_loss",
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get learning curve data for visualization."""
        query = self.db.query(LearningMetric).filter(
            LearningMetric.metric_type == metric_type
        )
        
        if model_version:
            query = query.filter(LearningMetric.model_version == model_version)
        
        # Filter by date if specified
        if days_back:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            query = query.filter(LearningMetric.timestamp >= cutoff_date)
        
        metrics = query.order_by(LearningMetric.epoch).all()
        
        return [
            {
                "epoch": m.epoch,
                "value": m.metric_value,
                "timestamp": m.timestamp.isoformat()
            }
            for m in metrics
        ]
    
    def get_best_model_info(self) -> Dict[str, Any]:
        """Get information about the best model."""
        best_checkpoint = self.db.query(ModelCheckpoint).filter(
            ModelCheckpoint.is_best == True
        ).order_by(ModelCheckpoint.created_at.desc()).first()
        
        if not best_checkpoint:
            return {"error": "No best model found"}
        
        return {
            "model_name": best_checkpoint.model_name,
            "version": best_checkpoint.version,
            "epoch": best_checkpoint.epoch,
            "train_loss": best_checkpoint.train_loss,
            "val_loss": best_checkpoint.val_loss,
            "train_accuracy": best_checkpoint.train_accuracy,
            "val_accuracy": best_checkpoint.val_accuracy,
            "checkpoint_path": best_checkpoint.checkpoint_path,
            "created_at": best_checkpoint.created_at.isoformat()
        }
    
    def track_prediction_outcome(
        self,
        prediction_id: int,
        actual_price: float,
        was_hit: bool
    ) -> bool:
        """Track the outcome of a prediction."""
        try:
            from backend.database.models import Prediction
            prediction = self.db.query(Prediction).filter(
                Prediction.id == prediction_id
            ).first()
            
            if prediction:
                prediction.actual_price = actual_price
                prediction.was_hit = was_hit
                self.db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error tracking prediction outcome: {e}")
            self.db.rollback()
            return False

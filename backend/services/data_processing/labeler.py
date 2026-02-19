"""Labeling system for training data."""
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from backend.database.models import Snapshot, PriceData, TrainingSample
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class Labeler:
    """Create labels for training data from before/after snapshots."""
    
    def create_labels(
        self,
        db: Session,
        before_snapshot_id: int,
        after_snapshot_id: int,
        expected_price: Optional[float] = None
    ) -> Optional[TrainingSample]:
        """
        Create training labels from before/after snapshot pair.
        
        Args:
            db: Database session
            before_snapshot_id: ID of before snapshot
            after_snapshot_id: ID of after snapshot
            expected_price: Optional expected price (for user predictions)
        
        Returns:
            TrainingSample with labels or None if failed
        """
        try:
            # Fetch snapshots
            before_snapshot = db.query(Snapshot).filter(
                Snapshot.id == before_snapshot_id
            ).first()
            after_snapshot = db.query(Snapshot).filter(
                Snapshot.id == after_snapshot_id
            ).first()
            
            if not before_snapshot or not after_snapshot:
                logger.error("Snapshots not found")
                return None
            
            # Fetch price data
            before_price_data = db.query(PriceData).filter(
                PriceData.snapshot_id == before_snapshot_id
            ).first()
            after_price_data = db.query(PriceData).filter(
                PriceData.snapshot_id == after_snapshot_id
            ).first()
            
            if not before_price_data or not after_price_data:
                logger.warning("Price data not available for labeling")
                return None
            
            # Calculate price movement
            before_price = before_price_data.close_price
            after_price = after_price_data.close_price
            
            price_change_absolute = after_price - before_price
            price_change_percentage = (price_change_absolute / before_price * 100) if before_price > 0 else 0.0
            
            # Determine direction
            threshold = 0.01  # 0.01% threshold for sideways
            if abs(price_change_percentage) < threshold:
                direction = "sideways"
            elif price_change_percentage > 0:
                direction = "up"
            else:
                direction = "down"
            
            # Check if expected price was hit
            target_hit = None
            if expected_price is not None:
                # Check if price reached expected price during the session
                # Use high/low prices to determine if target was hit
                session_high = max(
                    before_price_data.high_price,
                    after_price_data.high_price
                )
                session_low = min(
                    before_price_data.low_price,
                    after_price_data.low_price
                )
                
                if expected_price >= session_low and expected_price <= session_high:
                    target_hit = True
                else:
                    target_hit = False
            
            # Create training sample
            training_sample = TrainingSample(
                snapshot_id=before_snapshot_id,
                symbol=before_snapshot.symbol,
                session_date=before_snapshot.session_date,
                price_change_absolute=price_change_absolute,
                price_change_percentage=price_change_percentage,
                direction=direction,
                target_hit=target_hit,
                expected_price=expected_price,
                actual_price=after_price
            )
            
            db.add(training_sample)
            db.commit()
            db.refresh(training_sample)
            
            logger.info(
                f"Created training sample {training_sample.id} for {before_snapshot.symbol} "
                f"with {direction} movement ({price_change_percentage:.2f}%)"
            )
            
            return training_sample
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            db.rollback()
            return None
    
    def calculate_learning_metrics(
        self,
        predicted_price: float,
        actual_price: float,
        expected_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate learning metrics for a prediction.
        
        Args:
            predicted_price: Model's predicted price
            actual_price: Actual price that occurred
            expected_price: User's expected price (optional)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Prediction accuracy (how close predicted to actual)
        price_error = abs(predicted_price - actual_price)
        price_error_pct = (price_error / actual_price * 100) if actual_price > 0 else 0.0
        metrics["prediction_error"] = price_error
        metrics["prediction_error_pct"] = price_error_pct
        
        # Direction accuracy
        predicted_direction = "up" if predicted_price > actual_price else "down"
        actual_direction = "up" if actual_price > 0 else "down"  # Simplified
        metrics["direction_correct"] = 1.0 if predicted_direction == actual_direction else 0.0
        
        # Expected price hit (if provided)
        if expected_price is not None:
            metrics["expected_price_hit"] = 1.0 if abs(actual_price - expected_price) < (actual_price * 0.001) else 0.0
        
        return metrics

"""Labeling system for training data."""
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from backend.database.models import Snapshot, PriceData, TrainingSample, SessionMinuteBar
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class Labeler:
    """Create labels for training data from before/after snapshots."""

    def _prices_from_session_minute_bars(
        self, db: Session, session_date: str, symbol: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Get (before_price, after_price, session_high, session_low) from SessionMinuteBar.
        Before = first bar close, after = last bar close. Returns None if no bars.
        """
        bars = (
            db.query(SessionMinuteBar)
            .filter(
                SessionMinuteBar.session_date == session_date,
                SessionMinuteBar.symbol == symbol,
            )
            .order_by(SessionMinuteBar.bar_time)
            .all()
        )
        if not bars or len(bars) < 2:
            return None
        before_price = float(bars[0].close_price)
        after_price = float(bars[-1].close_price)
        session_high = max(float(b.high_price) for b in bars)
        session_low = min(float(b.low_price) for b in bars)
        return (before_price, after_price, session_high, session_low)

    def create_labels(
        self,
        db: Session,
        before_snapshot_id: int,
        after_snapshot_id: int,
        expected_price: Optional[float] = None
    ) -> Optional[TrainingSample]:
        """
        Create training labels from before/after snapshot pair.
        Uses PriceData when available; otherwise uses SessionMinuteBar (Databento):
        first bar close = before price, last bar close = after price.
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
            
            before_price: float
            after_price: float
            session_high: float
            session_low: float

            if before_price_data and after_price_data:
                before_price = before_price_data.close_price
                after_price = after_price_data.close_price
                session_high = max(
                    before_price_data.high_price,
                    after_price_data.high_price,
                )
                session_low = min(
                    before_price_data.low_price,
                    after_price_data.low_price,
                )
            else:
                # Fallback: use SessionMinuteBar (Databento minute bars)
                session_date = before_snapshot.session_date
                symbol = before_snapshot.symbol
                result = self._prices_from_session_minute_bars(db, session_date, symbol)
                if result is None:
                    logger.warning(
                        "No SessionMinuteBar data for %s %s (and no PriceData); cannot label",
                        symbol,
                        session_date,
                    )
                    return None
                before_price, after_price, session_high, session_low = result
                logger.debug(
                    "Using SessionMinuteBar for before/after labels: %s %s",
                    symbol,
                    session_date,
                )

            price_change_absolute = after_price - before_price
            price_change_percentage = (price_change_absolute / before_price * 100) if before_price > 0 else 0.0

            threshold = 0.01
            if abs(price_change_percentage) < threshold:
                direction = "sideways"
            elif price_change_percentage > 0:
                direction = "up"
            else:
                direction = "down"

            target_hit = None
            if expected_price is not None:
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

"""Update predictions with actual outcomes after session ends (actual_price, was_hit)."""
import logging
from sqlalchemy.orm import Session

from backend.database.models import Prediction, Snapshot, PriceData, SessionMinuteBar

logger = logging.getLogger(__name__)


def update_predictions_with_outcomes(db: Session, session_date: str) -> int:
    """
    Fill actual_price and was_hit for predictions that match the given session.
    Uses after-snapshot close price and session high/low (from SessionMinuteBar or PriceData).

    Returns:
        Number of predictions updated.
    """
    updated = 0
    try:
        after_snapshots = (
            db.query(Snapshot)
            .filter(
                Snapshot.session_date == session_date,
                Snapshot.snapshot_type == "after",
            )
            .all()
        )
        for snap in after_snapshots:
            after_price = db.query(PriceData).filter(PriceData.snapshot_id == snap.id).first()
            if not after_price:
                continue
            actual_price = float(after_price.close_price)
            session_high = float(after_price.high_price)
            session_low = float(after_price.low_price)

            bars = (
                db.query(SessionMinuteBar)
                .filter(
                    SessionMinuteBar.session_date == session_date,
                    SessionMinuteBar.symbol == snap.symbol,
                )
                .all()
            )
            if bars:
                session_high = max(session_high, max(float(b.high_price) for b in bars))
                session_low = min(session_low, min(float(b.low_price) for b in bars))

            for pred in db.query(Prediction).filter(
                Prediction.symbol == snap.symbol,
                Prediction.actual_price.is_(None),
            ).all():
                pred_date = pred.timestamp.date() if hasattr(pred.timestamp, "date") else pred.timestamp
                if str(pred_date) != session_date:
                    continue
                pred.actual_price = actual_price
                target = pred.user_expected_price if pred.user_expected_price is not None else pred.model_predicted_price
                pred.was_hit = (session_low <= target <= session_high) if target is not None else None
                updated += 1
        if updated:
            db.commit()
    except Exception as e:
        logger.exception("Outcome feedback update failed: %s", e)
        db.rollback()
        raise
    if updated:
        logger.info("Updated %s predictions with outcomes for session %s", updated, session_date)
    return updated

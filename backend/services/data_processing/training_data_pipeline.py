"""Pipeline to create training samples from before/after snapshot pairs and session candle pairs."""
import logging
from pathlib import Path
from datetime import datetime

from sqlalchemy.orm import Session

from backend.database.models import Snapshot, TrainingSample, SessionMinuteBar
from backend.config.settings import settings
from backend.services.data_processing.image_preprocessor import ImagePreprocessor
from backend.services.data_processing.feature_extractor import FeatureExtractor
from backend.services.data_processing.labeler import Labeler

logger = logging.getLogger(__name__)

# Per-interval first candle-close time (hour, minute) in session 6:30–8:00
SESSION_FIRST_BAR_TIME = {1: (6, 31), 5: (6, 35), 15: (6, 45), 60: (7, 0)}
SESSION_END_HOUR, SESSION_END_MINUTE = 8, 0


def process_training_data_from_snapshots(db: Session) -> dict:
    """
    Process before/after snapshot pairs into TrainingSamples (preprocess images,
    extract features, create labels). Skips pairs that already have a training sample
    or where the before snapshot has no valid image file.

    Returns:
        Dict with keys: created (int), skipped (int), errors (list).
    """
    preprocessor = ImagePreprocessor()
    feature_extractor = FeatureExtractor()
    labeler = Labeler()
    created = 0
    skipped = 0
    errors = []

    try:
        before_snapshots = db.query(Snapshot).filter(
            Snapshot.snapshot_type == "before"
        ).all()
    except Exception as e:
        logger.exception("Failed to query before snapshots: %s", e)
        return {"created": 0, "skipped": 0, "errors": [str(e)]}

    for before_snapshot in before_snapshots:
        try:
            after_snapshot = db.query(Snapshot).filter(
                Snapshot.symbol == before_snapshot.symbol,
                Snapshot.session_date == before_snapshot.session_date,
                Snapshot.snapshot_type == "after",
            ).first()

            if not after_snapshot:
                skipped += 1
                continue

            existing = db.query(TrainingSample).filter(
                TrainingSample.snapshot_id == before_snapshot.id
            ).first()
            if existing:
                skipped += 1
                continue

            before_image_path = Path(before_snapshot.image_path)
            if not before_image_path.is_file():
                logger.debug(
                    "Skipping %s %s: no image file at %s",
                    before_snapshot.symbol,
                    before_snapshot.session_date,
                    before_snapshot.image_path,
                )
                skipped += 1
                continue

            processed_image = preprocessor.preprocess(before_image_path)
            if processed_image is None:
                errors.append(
                    f"Preprocess failed: {before_snapshot.symbol} {before_snapshot.session_date}"
                )
                continue

            processed_filename = (
                f"processed_{before_snapshot.symbol}_{before_snapshot.session_date}_before.png"
            )
            processed_path = settings.PROCESSED_DATA_DIR / processed_filename
            preprocessor.save_processed_image(processed_image, processed_path)

            features = feature_extractor.extract_features(
                before_image_path,
                before_snapshot.timestamp,
                before_snapshot.symbol,
            )
            session_bar_features = feature_extractor.extract_session_bar_features(
                before_snapshot.session_date,
                before_snapshot.symbol,
                db,
            )
            features.update(session_bar_features)
            pattern_features = feature_extractor.extract_chart_patterns(processed_image)
            features.update(pattern_features)

            training_sample = labeler.create_labels(
                db,
                before_snapshot.id,
                after_snapshot.id,
            )
            if training_sample:
                training_sample.processed_image_path = str(processed_path)
                training_sample.features = features
                db.commit()
                created += 1
                logger.info(
                    "Created training sample for %s on %s",
                    before_snapshot.symbol,
                    before_snapshot.session_date,
                )
        except Exception as e:
            logger.exception(
                "Error processing %s %s: %s",
                before_snapshot.symbol,
                before_snapshot.session_date,
                e,
            )
            db.rollback()
            errors.append(f"{before_snapshot.symbol} {before_snapshot.session_date}: {e}")

    return {"created": created, "skipped": skipped, "errors": errors}


def process_training_data_from_session_candles(db: Session) -> dict:
    """
    Create training samples from session_candle snapshots: per-interval "before" (first bar) vs "after" (8:00).
    Uses SessionMinuteBar for labels (before_price, after_price, session_high, session_low).
    Idempotent: skips (session_date, symbol, interval) that already have a training sample from that before snapshot.

    Returns:
        Dict with keys: created (int), skipped (int), errors (list).
    """
    preprocessor = ImagePreprocessor()
    feature_extractor = FeatureExtractor()
    created = 0
    skipped = 0
    errors = []

    for interval_minutes in (1, 5, 15, 60):
        h, m = SESSION_FIRST_BAR_TIME[interval_minutes]
        for before_snapshot in db.query(Snapshot).filter(
            Snapshot.snapshot_type == "session_candle",
            Snapshot.interval_minutes == interval_minutes,
            Snapshot.bar_time.isnot(None),
        ).all():
            try:
                # bar_time is stored naive; first bar has time (h, m)
                if before_snapshot.bar_time.hour != h or before_snapshot.bar_time.minute != m:
                    continue
                session_date = before_snapshot.session_date
                end_dt = datetime.combine(
                    datetime.strptime(session_date, "%Y-%m-%d").date(),
                    datetime.strptime(f"{SESSION_END_HOUR:02d}:{SESSION_END_MINUTE:02d}", "%H:%M").time(),
                )
                after_snapshot = db.query(Snapshot).filter(
                    Snapshot.symbol == before_snapshot.symbol,
                    Snapshot.session_date == session_date,
                    Snapshot.snapshot_type == "session_candle",
                    Snapshot.interval_minutes == interval_minutes,
                    Snapshot.bar_time == end_dt,
                ).first()
                if not after_snapshot:
                    skipped += 1
                    continue
                existing = db.query(TrainingSample).filter(
                    TrainingSample.snapshot_id == before_snapshot.id,
                ).first()
                if existing:
                    skipped += 1
                    continue
                before_image_path = Path(before_snapshot.image_path)
                if not before_image_path.is_file():
                    skipped += 1
                    continue
                bars = (
                    db.query(SessionMinuteBar)
                    .filter(
                        SessionMinuteBar.session_date == session_date,
                        SessionMinuteBar.symbol == before_snapshot.symbol,
                    )
                    .order_by(SessionMinuteBar.bar_time)
                    .all()
                )
                if not bars:
                    errors.append(f"No SessionMinuteBar for {before_snapshot.symbol} {session_date}")
                    continue
                before_bar = next((b for b in bars if b.bar_time == before_snapshot.bar_time), None)
                after_bar = next((b for b in bars if b.bar_time == end_dt), None)
                if not before_bar or not after_bar:
                    errors.append(
                        f"Missing bar for {before_snapshot.symbol} {session_date} interval={interval_minutes}"
                    )
                    continue
                before_price = float(before_bar.close_price)
                after_price = float(after_bar.close_price)
                session_high = max(float(b.high_price) for b in bars)
                session_low = min(float(b.low_price) for b in bars)
                price_change_absolute = after_price - before_price
                price_change_percentage = (price_change_absolute / before_price * 100) if before_price > 0 else 0.0
                threshold = 0.01
                if abs(price_change_percentage) < threshold:
                    direction = "sideways"
                elif price_change_percentage > 0:
                    direction = "up"
                else:
                    direction = "down"
                processed_image = preprocessor.preprocess(before_image_path)
                if processed_image is None:
                    errors.append(f"Preprocess failed: {before_snapshot.symbol} {session_date} {interval_minutes}m")
                    continue
                processed_filename = (
                    f"processed_{before_snapshot.symbol}_{session_date}_{interval_minutes}m_before.png"
                )
                processed_path = settings.PROCESSED_DATA_DIR / processed_filename
                preprocessor.save_processed_image(processed_image, processed_path)
                features = feature_extractor.extract_features(
                    before_image_path,
                    before_snapshot.timestamp,
                    before_snapshot.symbol,
                )
                session_bar_features = feature_extractor.extract_session_bar_features(
                    session_date,
                    before_snapshot.symbol,
                    db,
                )
                features.update(session_bar_features)
                features["interval_minutes"] = interval_minutes
                features["bar_time"] = before_snapshot.bar_time.isoformat() if before_snapshot.bar_time else None
                pattern_features = feature_extractor.extract_chart_patterns(processed_image)
                features.update(pattern_features)
                training_sample = TrainingSample(
                    snapshot_id=before_snapshot.id,
                    symbol=before_snapshot.symbol,
                    session_date=session_date,
                    interval_minutes=interval_minutes,
                    processed_image_path=str(processed_path),
                    features=features,
                    price_change_absolute=price_change_absolute,
                    price_change_percentage=price_change_percentage,
                    direction=direction,
                    target_hit=None,
                    expected_price=None,
                    actual_price=after_price,
                )
                db.add(training_sample)
                db.commit()
                created += 1
                logger.info(
                    "Created session_candle training sample %s %s %sm %s",
                    before_snapshot.symbol,
                    session_date,
                    interval_minutes,
                    direction,
                )
            except Exception as e:
                logger.exception(
                    "Error processing session_candle %s %s %sm: %s",
                    getattr(before_snapshot, "symbol", "?"),
                    getattr(before_snapshot, "session_date", "?"),
                    interval_minutes,
                    e,
                )
                db.rollback()
                errors.append(f"session_candle {interval_minutes}m: {e}")

    return {"created": created, "skipped": skipped, "errors": errors}

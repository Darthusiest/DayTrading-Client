"""Pipeline to create training samples from before/after snapshot pairs."""
import logging
from pathlib import Path
from sqlalchemy.orm import Session

from backend.database.models import Snapshot, TrainingSample
from backend.config.settings import settings
from backend.services.data_processing.image_preprocessor import ImagePreprocessor
from backend.services.data_processing.feature_extractor import FeatureExtractor
from backend.services.data_processing.labeler import Labeler

logger = logging.getLogger(__name__)


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

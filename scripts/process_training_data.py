"""Script to process training data from snapshots."""
import sys
from pathlib import Path
from sqlalchemy.orm import Session

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.db import SessionLocal
from backend.database.models import Snapshot, TrainingSample
from backend.services.data_processing.image_preprocessor import ImagePreprocessor
from backend.services.data_processing.feature_extractor import FeatureExtractor
from backend.services.data_processing.labeler import Labeler
from backend.config.settings import settings

def process_training_data():
    """Process snapshots into training samples."""
    db: Session = SessionLocal()
    
    preprocessor = ImagePreprocessor()
    feature_extractor = FeatureExtractor()
    labeler = Labeler()
    
    try:
        # Get all before snapshots
        before_snapshots = db.query(Snapshot).filter(
            Snapshot.snapshot_type == "before"
        ).all()
        
        print(f"Found {len(before_snapshots)} before snapshots")
        
        for before_snapshot in before_snapshots:
            # Find corresponding after snapshot
            after_snapshot = db.query(Snapshot).filter(
                Snapshot.symbol == before_snapshot.symbol,
                Snapshot.session_date == before_snapshot.session_date,
                Snapshot.snapshot_type == "after"
            ).first()
            
            if not after_snapshot:
                print(f"No after snapshot found for {before_snapshot.symbol} on {before_snapshot.session_date}")
                continue
            
            # Check if training sample already exists
            existing = db.query(TrainingSample).filter(
                TrainingSample.snapshot_id == before_snapshot.id
            ).first()
            
            if existing:
                print(f"Training sample already exists for {before_snapshot.symbol} on {before_snapshot.session_date}")
                continue
            
            # Preprocess images
            before_image_path = Path(before_snapshot.image_path)
            processed_image = preprocessor.preprocess(before_image_path)
            
            if processed_image is None:
                print(f"Failed to preprocess image: {before_image_path}")
                continue
            
            # Save processed image
            processed_filename = f"processed_{before_snapshot.symbol}_{before_snapshot.session_date}_{before_snapshot.snapshot_type}.png"
            processed_path = settings.PROCESSED_DATA_DIR / processed_filename
            preprocessor.save_processed_image(processed_image, processed_path)
            
            # Extract features
            features = feature_extractor.extract_features(
                before_image_path,
                before_snapshot.timestamp,
                before_snapshot.symbol
            )
            
            # Create labels
            training_sample = labeler.create_labels(
                db,
                before_snapshot.id,
                after_snapshot.id
            )
            
            if training_sample:
                training_sample.processed_image_path = str(processed_path)
                training_sample.features = features
                db.commit()
                print(f"Created training sample for {before_snapshot.symbol} on {before_snapshot.session_date}")
    
    except Exception as e:
        print(f"Error processing training data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    process_training_data()

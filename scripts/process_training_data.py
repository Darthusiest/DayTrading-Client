"""Script to process training data from snapshots (uses shared pipeline)."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.db import SessionLocal
from backend.services.data_processing.training_data_pipeline import process_training_data_from_snapshots


def main():
    """Process snapshots into training samples."""
    db = SessionLocal()
    try:
        result = process_training_data_from_snapshots(db)
        print(f"Created: {result['created']}, Skipped: {result['skipped']}")
        if result["errors"]:
            for err in result["errors"]:
                print(f"Error: {err}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

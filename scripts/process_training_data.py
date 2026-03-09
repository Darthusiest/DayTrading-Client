"""Script to process training data from snapshots (uses shared pipeline)."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.db import SessionLocal
from backend.services.pipeline.orchestrator import build_datasets


def main():
    """Process snapshots into training samples."""
    db = SessionLocal()
    try:
        result = build_datasets(db, mode="snapshots")
        print(f"Created: {result['created']}, Skipped: {result['skipped']}")
        if result["errors"]:
            for err in result["errors"]:
                print(f"Error: {err}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

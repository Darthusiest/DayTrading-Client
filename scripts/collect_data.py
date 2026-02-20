"""Script to collect before/after snapshots (uses shared collector)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.db import SessionLocal
from backend.services.data_collection.collector import run_collection


def main():
    db = SessionLocal()
    try:
        result = run_collection(db, capture_screenshots=True)
        print(
            f"Collected: {result['collected']}, Failed: {result['failed']}, "
            f"Type: {result['snapshot_type']}"
        )
        for err in result.get("errors", []):
            print(f"  Error: {err}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

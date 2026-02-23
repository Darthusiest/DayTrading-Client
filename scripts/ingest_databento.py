"""CLI to ingest Databento OHLCV-1m batch files into session_minute_bars."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database.db import init_db, SessionLocal
from backend.services.data_collection.databento_ingestion import run_ingestion


def main():
    parser = argparse.ArgumentParser(description="Ingest Databento .dbn.zst files into session_minute_bars")
    parser.add_argument("--dry-run", action="store_true", help="Decode and count bars only; do not write to DB")
    parser.add_argument("--path", type=Path, default=None, help="Single file or directory to process (default: DATABENTO_RAW_DIR)")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        result = run_ingestion(db, path_override=args.path, dry_run=args.dry_run)
        print(f"Files processed: {result['files_processed']}, Bars inserted: {result['bars_inserted']}")
        if result.get("errors"):
            for err in result["errors"]:
                print(f"Error: {err}")
            sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

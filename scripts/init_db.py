"""Initialize database tables."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `backend` imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import settings  # noqa: E402
from backend.database.db import init_db  # noqa: E402

if __name__ == "__main__":
    print(f"Initializing database: {settings.DATABASE_URL}")
    init_db()
    print("Database initialized successfully!")

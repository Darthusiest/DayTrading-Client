"""Initialize database tables."""
from backend.database.db import init_db
from backend.config.settings import settings

if __name__ == "__main__":
    print(f"Initializing database: {settings.DATABASE_URL}")
    init_db()
    print("Database initialized successfully!")

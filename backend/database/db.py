"""Database connection and session management."""
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.config.settings import settings

logger = logging.getLogger(__name__)

# Create database engine (SQLite needs check_same_thread=False for FastAPI)
_connect_args = {}
if "sqlite" in settings.DATABASE_URL:
    _connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=("sqlite" not in settings.DATABASE_URL),
    echo=settings.DEBUG,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _add_missing_columns_sqlite():
    """Add columns that were added to models after the DB was first created (SQLite only)."""
    if "sqlite" not in settings.DATABASE_URL:
        return
    # Table -> list of (column_name, type_sql)
    additions = {
        "snapshots": [
            ("interval_minutes", "INTEGER"),
            ("bar_time", "DATETIME"),
        ],
        "training_samples": [
            ("interval_minutes", "INTEGER"),
        ],
    }
    with engine.connect() as conn:
        for table, columns in additions.items():
            try:
                r = conn.execute(text(f"PRAGMA table_info({table})"))
                existing = {row[1] for row in r}
            except Exception:
                continue  # Table may not exist yet
            for col_name, col_type in columns:
                if col_name not in existing:
                    try:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"))
                        conn.commit()
                        logger.info("Added column %s.%s", table, col_name)
                    except Exception as e:
                        logger.warning("Could not add %s.%s: %s", table, col_name, e)


def init_db():
    """Initialize database tables and add any missing columns (SQLite)."""
    Base.metadata.create_all(bind=engine)
    _add_missing_columns_sqlite()

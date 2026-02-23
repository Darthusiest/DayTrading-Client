"""Pytest fixtures: test DB, FastAPI TestClient with get_db override."""
import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# File-based SQLite so all connections (including from TestClient request thread) see the same DB.
_TEST_DB = os.environ.get("TEST_DATABASE_URL")
if _TEST_DB is None:
    _test_dir = Path(tempfile.gettempdir()) / "daytrade_tests"
    _test_dir.mkdir(exist_ok=True)
    _TEST_DB = f"sqlite:///{_test_dir}/test.db"


@pytest.fixture
def db_engine():
    """Create a fresh SQLite engine for tests (in-memory or temp file)."""
    connect_args = {}
    if "sqlite" in _TEST_DB:
        connect_args["check_same_thread"] = False
    engine = create_engine(_TEST_DB, connect_args=connect_args)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create tables and yield a session; close after test."""
    from backend.database.db import Base
    from backend.database import models  # noqa: F401 - register all models with Base
    Base.metadata.drop_all(bind=db_engine)
    Base.metadata.create_all(bind=db_engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    """FastAPI TestClient with get_db overridden to use the test db_session."""
    from backend.api.main import app
    from backend.database.db import get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)

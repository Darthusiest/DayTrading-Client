"""Tests for collection API routes."""
import pytest
from unittest.mock import patch

# Routes are mounted at API_V1_PREFIX (e.g. /api/v1)
PREFIX = "/api/v1"


class TestCollectionRun:
    def test_post_run_default_capture_screenshots_true(self, client):
        with patch("backend.api.routes.collection.run_collection") as m_run:
            m_run.return_value = {"collected": 2, "failed": 0, "snapshot_type": "before", "errors": []}
            r = client.post(f"{PREFIX}/collection/run")
        assert r.status_code == 200
        m_run.assert_called_once()
        call_kw = m_run.call_args[1]
        assert call_kw.get("capture_screenshots", True) is True

    def test_post_run_capture_screenshots_false(self, client):
        with patch("backend.api.routes.collection.run_collection") as m_run:
            m_run.return_value = {"collected": 0, "failed": 0, "snapshot_type": "none", "errors": []}
            r = client.post(f"{PREFIX}/collection/run?capture_screenshots=false")
        assert r.status_code == 200
        m_run.assert_called_once()
        call_kw = m_run.call_args[1]
        assert call_kw["capture_screenshots"] is False


class TestCaptureNow:
    def test_valid_interval_200(self, client):
        with patch("backend.api.routes.collection.capture_snapshot_now") as m_capture:
            m_capture.return_value = {"success": True, "snapshot_id": 1, "symbol": "MNQ1!", "interval_minutes": 15}
            r = client.post(f"{PREFIX}/collection/capture-now?symbol=MNQ1!&interval=15")
        assert r.status_code == 200
        assert r.json().get("success") is True

    def test_invalid_interval_400(self, client):
        r = client.post(f"{PREFIX}/collection/capture-now?symbol=MNQ1!&interval=2")
        assert r.status_code == 400
        assert "interval" in r.json().get("detail", "").lower()


class TestIngestDatabento:
    def test_post_no_params(self, client):
        with patch("backend.api.routes.collection.run_ingestion") as m_ingest:
            m_ingest.return_value = {"files_processed": 0, "bars_inserted": 0, "errors": ["No files"]}
            r = client.post(f"{PREFIX}/collection/ingest-databento")
        assert r.status_code == 200
        m_ingest.assert_called_once()
        args, kwargs = m_ingest.call_args
        assert kwargs.get("path_override") is None
        assert kwargs.get("dry_run") is False

    def test_post_with_path_and_dry_run(self, client):
        with patch("backend.api.routes.collection.run_ingestion") as m_ingest:
            m_ingest.return_value = {"files_processed": 1, "bars_inserted": 0, "errors": []}
            r = client.post(f"{PREFIX}/collection/ingest-databento?path=/tmp/foo&dry_run=true")
        assert r.status_code == 200
        m_ingest.assert_called_once()
        kwargs = m_ingest.call_args[1]
        assert str(kwargs["path_override"]) == "/tmp/foo"
        assert kwargs["dry_run"] is True


class TestProcessTrainingData:
    def test_post_aggregates_results(self, client):
        with patch("backend.api.routes.collection.process_training_data_from_snapshots") as m_snap:
            with patch("backend.api.routes.collection.process_training_data_from_session_candles") as m_sess:
                m_snap.return_value = {"created": 5, "skipped": 0, "errors": []}
                m_sess.return_value = {"created": 3, "skipped": 1, "errors": []}
                r = client.post(f"{PREFIX}/collection/process-training-data")
        assert r.status_code == 200
        data = r.json()
        assert data["created"] == 8
        assert data["skipped"] == 1
        assert "before_after" in data
        assert "session_candles" in data
        assert len(data["errors"]) == 0

"""Tests for backend.services.data_collection.collector."""
import pytest
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytz

from backend.services.data_collection.collector import (
    _parse_session_time,
    _intervals_closing_at,
    run_collection,
    run_session_candle_capture,
    capture_snapshot_now,
)


class TestParseSessionTime:
    @pytest.mark.parametrize("time_str,expected", [
        ("09:30", (9, 30)),
        ("16:00", (16, 0)),
        ("7:5", (7, 5)),
        ("0:0", (0, 0)),
        (" 12 : 45 ", (12, 45)),
    ])
    def test_valid_formats(self, time_str, expected):
        assert _parse_session_time(time_str) == expected

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _parse_session_time("")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            _parse_session_time("abc")

    def test_missing_minute_part_raises(self):
        with pytest.raises(ValueError):
            _parse_session_time("10:")


class TestIntervalsClosingAt:
    @pytest.mark.parametrize("hour,minute,session_end_hour,session_end_minute,expected", [
        (9, 31, 16, 0, [1]),
        (9, 35, 16, 0, [1, 5]),
        (9, 45, 16, 0, [1, 5, 15]),
        (10, 0, 16, 0, [1, 5, 15, 60]),
        (16, 0, 16, 0, [1, 5, 15, 60]),
        (16, 30, 16, 0, [1, 5, 15]),  # 60 only if minute <= session_end_minute at end hour
    ])
    def test_intervals(self, hour, minute, session_end_hour, session_end_minute, expected):
        result = _intervals_closing_at(hour, minute, session_end_hour, session_end_minute)
        assert set(result) == set(expected)
        assert 1 in result


class TestRunCollection:
    @pytest.fixture(autouse=True)
    def patch_settings(self, monkeypatch):
        monkeypatch.setattr("backend.services.data_collection.collector.settings.SESSION_TIMEZONE", "America/New_York")
        monkeypatch.setattr("backend.services.data_collection.collector.settings.BEFORE_SNAPSHOT_TIME", "09:30")
        monkeypatch.setattr("backend.services.data_collection.collector.settings.AFTER_SNAPSHOT_TIME", "16:00")
        monkeypatch.setattr("backend.services.data_collection.collector.settings.SYMBOLS", ["MNQ1!", "MES1!"])

    @contextmanager
    def _patch_now(self, year, month, day, hour, minute, second=0):
        """Patch datetime.now in collector to return given time in America/New_York, keep strptime real."""
        session_tz = pytz.timezone("America/New_York")
        fake_now = datetime(year, month, day, hour, minute, second, tzinfo=session_tz)
        with patch("backend.services.data_collection.collector.datetime") as m_dt:
            m_dt.now.return_value = fake_now
            m_dt.strptime = datetime.strptime
            yield

    def test_before_time_returns_before(self, db_session):
        with self._patch_now(2025, 6, 15, 9, 0, 0):
            with patch("backend.services.data_collection.collector.ScreenshotCapture") as MockCapture:
                MockCapture.return_value.capture_chart_screenshot.return_value = Path("/tmp/fake.png")
                result = run_collection(db_session, session_date="2025-06-15", capture_screenshots=True)
        assert result["snapshot_type"] == "before"
        assert result["collected"] == 2
        assert result["failed"] == 0

    def test_after_time_returns_after(self, db_session):
        with self._patch_now(2025, 6, 15, 16, 30, 0):
            with patch("backend.services.data_collection.collector.ScreenshotCapture") as MockCapture:
                MockCapture.return_value.capture_chart_screenshot.return_value = Path("/tmp/fake.png")
                result = run_collection(db_session, session_date="2025-06-15", capture_screenshots=True)
        assert result["snapshot_type"] == "after"
        assert result["collected"] == 2

    def test_between_returns_none(self, db_session):
        with self._patch_now(2025, 6, 15, 12, 0, 0):
            result = run_collection(db_session, session_date="2025-06-15")
        assert result["snapshot_type"] == "none"
        assert result["collected"] == 0
        assert "between" in result["errors"][0].lower() or "between" in str(result["errors"]).lower()

    def test_override_before_ignores_time(self, db_session):
        with self._patch_now(2025, 6, 15, 12, 0, 0):
            with patch("backend.services.data_collection.collector.ScreenshotCapture") as MockCapture:
                MockCapture.return_value.capture_chart_screenshot.return_value = Path("/tmp/fake.png")
                result = run_collection(
                    db_session, session_date="2025-06-15",
                    snapshot_type_override="before", capture_screenshots=True
                )
        assert result["snapshot_type"] == "before"
        assert result["collected"] == 2

    def test_capture_screenshots_false_still_creates_snapshots(self, db_session):
        with self._patch_now(2025, 6, 15, 9, 30, 0):
            result = run_collection(db_session, session_date="2025-06-15", capture_screenshots=False)
        assert result["snapshot_type"] == "before"
        assert result["collected"] == 2
        from backend.database.models import Snapshot
        rows = db_session.query(Snapshot).filter(Snapshot.session_date == "2025-06-15").all()
        assert len(rows) == 2
        assert "price-only" in rows[0].image_path or "price-only" in rows[1].image_path

    def test_screenshot_failure_increments_failed(self, db_session):
        with self._patch_now(2025, 6, 15, 9, 30, 0):
            with patch("backend.services.data_collection.collector.ScreenshotCapture") as MockCapture:
                mock_capture = MockCapture.return_value
                mock_capture.capture_chart_screenshot.side_effect = [None, Path("/tmp/fake.png")]
                result = run_collection(db_session, session_date="2025-06-15", capture_screenshots=True)
        assert result["failed"] >= 1
        assert any("Failed" in e or "fail" in e.lower() for e in result["errors"])


class TestRunSessionCandleCapture:
    def test_disabled_returns_no_op(self, db_session, monkeypatch):
        monkeypatch.setattr("backend.services.data_collection.collector.settings.ENABLE_SESSION_CANDLE_CAPTURE", False)
        result = run_session_candle_capture(db_session)
        assert result["collected"] == 0
        assert "disabled" in result["errors"][0].lower()

    def test_enabled_uses_screenshot_capture(self, db_session, monkeypatch):
        monkeypatch.setattr("backend.services.data_collection.collector.settings.ENABLE_SESSION_CANDLE_CAPTURE", True)
        monkeypatch.setattr("backend.services.data_collection.collector.settings.SESSION_START_TIME", "10:00")
        monkeypatch.setattr("backend.services.data_collection.collector.settings.SESSION_END_TIME", "10:02")
        monkeypatch.setattr("backend.services.data_collection.collector.settings.SYMBOLS", ["MNQ1!"])
        with patch("backend.services.data_collection.collector.ScreenshotCapture") as MockCapture:
            mock_capture = MockCapture.return_value
            mock_capture.capture_chart_screenshot.return_value = Path("/tmp/session_candle.png")
            result = run_session_candle_capture(db_session, session_date="2025-06-15")
        assert "collected" in result
        assert "errors" in result
        mock_capture._init_driver.assert_called_once()
        mock_capture._ensure_logged_in.assert_called_once()


class TestCaptureSnapshotNow:
    @pytest.fixture(autouse=True)
    def patch_symbols(self, monkeypatch):
        monkeypatch.setattr("backend.services.data_collection.collector.settings.SYMBOLS", ["MNQ1!", "MES1!"])

    def test_valid_symbol_and_interval(self, db_session):
        with patch("backend.services.data_collection.collector.ScreenshotCapture") as MockCapture:
            MockCapture.return_value.capture_chart_screenshot.return_value = Path("/tmp/manual.png")
            result = capture_snapshot_now(db_session, symbol="MNQ1!", interval_minutes=15)
        assert result["success"] is True
        assert result["snapshot_id"] is not None
        assert result["symbol"] == "MNQ1!"
        assert result["interval_minutes"] == 15

    def test_invalid_symbol_returns_false(self, db_session):
        result = capture_snapshot_now(db_session, symbol="INVALID", interval_minutes=15)
        assert result["success"] is False
        assert "SYMBOLS" in result["error"] or "not in" in result["error"].lower()

    def test_screenshot_failure_returns_false(self, db_session):
        with patch("backend.services.data_collection.collector.ScreenshotCapture") as MockCapture:
            MockCapture.return_value.capture_chart_screenshot.return_value = None
            result = capture_snapshot_now(db_session, symbol="MNQ1!", interval_minutes=15)
        assert result["success"] is False
        assert "Failed" in result["error"] or "fail" in result["error"].lower()
        from backend.database.models import Snapshot
        count = db_session.query(Snapshot).filter(Snapshot.snapshot_type == "manual").count()
        assert count == 0

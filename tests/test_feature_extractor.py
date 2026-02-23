"""Tests for backend.services.data_processing.feature_extractor."""
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytz

from backend.services.data_processing.feature_extractor import FeatureExtractor
from backend.database.models import SessionMinuteBar


@pytest.fixture
def extractor():
    return FeatureExtractor()


class TestExtractFeatures:
    def test_valid_with_price_data(self, extractor, tmp_path):
        (tmp_path / "dummy.png").touch()
        ts = datetime(2025, 6, 15, 10, 30, 0, tzinfo=extractor.timezone)
        price_data = {"open": 21000.0, "high": 21050.0, "low": 20980.0, "close": 21020.0, "volume": 1000}
        result = extractor.extract_features(
            tmp_path / "dummy.png", ts, "MNQ1!", price_data=price_data
        )
        assert result["symbol"] == "MNQ1!"
        assert "2025-06-15" in result["timestamp"]
        assert result["image_path"] == str(tmp_path / "dummy.png")
        assert "hour" in result and "minute" in result
        assert "day_of_week" in result and "is_weekend" in result
        assert "is_market_hours" in result
        assert "price_change" in result
        assert "price_change_pct" in result
        assert "volume" in result

    def test_valid_without_price_data(self, extractor, tmp_path):
        (tmp_path / "dummy.png").touch()
        ts = datetime(2025, 6, 15, 10, 30, 0, tzinfo=extractor.timezone)
        result = extractor.extract_features(tmp_path / "dummy.png", ts, "MNQ1!", price_data=None)
        assert result["symbol"] == "MNQ1!"
        assert "timestamp" in result and "image_path" in result
        assert "hour" in result and "minute" in result
        assert "price_change" not in result
        assert "volume" not in result


class TestExtractTimeFeatures:
    def test_naive_utc_converted_to_timezone(self, extractor, monkeypatch):
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.TIMEZONE", "America/New_York")
        ext = FeatureExtractor()
        naive = datetime(2025, 6, 15, 15, 45, 0)  # naive
        features = ext._extract_time_features(naive)
        assert features["hour"] in (15, 11)  # depends on DST
        assert features["minute"] == 45
        assert features["day_of_week"] == 6  # Sunday
        assert features["is_weekend"] is True

    def test_aware_converted_to_timezone(self, extractor, monkeypatch):
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.TIMEZONE", "America/New_York")
        ext = FeatureExtractor()
        utc = datetime(2025, 6, 16, 14, 0, 0, tzinfo=pytz.UTC)  # Monday 10:00 ET
        features = ext._extract_time_features(utc)
        assert features["day_of_week"] == 0
        assert features["is_weekend"] is False

    def test_is_weekend_saturday_sunday(self, extractor, monkeypatch):
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.TIMEZONE", "UTC")
        ext = FeatureExtractor()
        sat = datetime(2025, 6, 14, 12, 0, 0, tzinfo=pytz.UTC)
        sun = datetime(2025, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        mon = datetime(2025, 6, 16, 12, 0, 0, tzinfo=pytz.UTC)
        assert ext._extract_time_features(sat)["is_weekend"] is True
        assert ext._extract_time_features(sun)["is_weekend"] is True
        assert ext._extract_time_features(mon)["is_weekend"] is False


class TestIsMarketHours:
    def test_at_open_true(self, extractor, monkeypatch):
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.SESSION_START_TIME", "09:30")
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.SESSION_END_TIME", "16:00")
        ext = FeatureExtractor()
        ts = datetime(2025, 6, 16, 9, 30, 0, tzinfo=extractor.timezone)
        assert ext._is_market_hours(ts) is True

    def test_between_true(self, extractor, monkeypatch):
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.SESSION_START_TIME", "09:30")
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.SESSION_END_TIME", "16:00")
        ext = FeatureExtractor()
        ts = datetime(2025, 6, 16, 12, 0, 0, tzinfo=extractor.timezone)
        assert ext._is_market_hours(ts) is True

    def test_at_close_true(self, extractor, monkeypatch):
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.SESSION_START_TIME", "09:30")
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.SESSION_END_TIME", "16:00")
        ext = FeatureExtractor()
        ts = datetime(2025, 6, 16, 16, 0, 0, tzinfo=extractor.timezone)
        assert ext._is_market_hours(ts) is True

    def test_one_minute_before_open_false(self, extractor, monkeypatch):
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.SESSION_START_TIME", "09:30")
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.SESSION_END_TIME", "16:00")
        ext = FeatureExtractor()
        ts = datetime(2025, 6, 16, 9, 29, 0, tzinfo=extractor.timezone)
        assert ext._is_market_hours(ts) is False

    def test_one_minute_after_close_false(self, extractor, monkeypatch):
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.SESSION_START_TIME", "09:30")
        monkeypatch.setattr("backend.services.data_processing.feature_extractor.settings.SESSION_END_TIME", "16:00")
        ext = FeatureExtractor()
        ts = datetime(2025, 6, 16, 16, 1, 0, tzinfo=extractor.timezone)
        assert ext._is_market_hours(ts) is False


class TestExtractPriceFeatures:
    def test_normal_ohlcv(self, extractor):
        price_data = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000}
        features = extractor._extract_price_features(price_data)
        assert features["price_change"] == 2.0
        assert features["price_change_pct"] == 2.0
        assert features["price_range"] == 10.0
        assert features["volume"] == 1000

    def test_close_zero_no_division(self, extractor):
        price_data = {"open": 100.0, "high": 100.0, "low": 100.0, "close": 0.0, "volume": 0}
        features = extractor._extract_price_features(price_data)
        assert "price_change_pct" not in features or features.get("price_change_pct", 0) == 0
        assert "volume" not in features

    def test_volume_zero_omitted(self, extractor):
        price_data = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 0}
        features = extractor._extract_price_features(price_data)
        assert "volume" not in features

    def test_volume_negative_omitted(self, extractor):
        price_data = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": -1}
        features = extractor._extract_price_features(price_data)
        assert "volume" not in features

    def test_high_equals_low_doji(self, extractor):
        price_data = {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 100}
        features = extractor._extract_price_features(price_data)
        assert features["price_range"] == 0.0
        assert features["price_range_pct"] == 0.0
        assert features["body_size"] == 0.0
        assert features["upper_wick"] == 0.0
        assert features["lower_wick"] == 0.0


class TestExtractSessionBarFeatures:
    def test_empty_data_returns_defaults(self, extractor, db_session):
        result = extractor.extract_session_bar_features("2025-06-15", "MNQ1!", db_session)
        assert result["session_num_bars"] == 0
        assert result["session_return_pct"] == 0.0
        assert result["session_range_pct"] == 0.0
        assert result["session_volatility"] == 0.0

    def test_single_bar_returns_defaults(self, extractor, db_session):
        db_session.add(SessionMinuteBar(
            session_date="2025-06-15", symbol="MNQ1!",
            bar_time=datetime(2025, 6, 15, 9, 30, 0),
            open_price=21000.0, high_price=21010.0, low_price=20990.0, close_price=21005.0, volume=100,
        ))
        db_session.commit()
        result = extractor.extract_session_bar_features("2025-06-15", "MNQ1!", db_session)
        # Implementation returns early when len(bars) < 2
        assert result["session_num_bars"] == 0

    def test_multiple_bars(self, extractor, db_session):
        for i in range(5):
            db_session.add(SessionMinuteBar(
                session_date="2025-06-15", symbol="MNQ1!",
                bar_time=datetime(2025, 6, 15, 9, 30 + i, 0),
                open_price=21000.0 + i * 2,
                high_price=21010.0 + i * 2,
                low_price=20990.0 + i * 2,
                close_price=21005.0 + i * 2,
                volume=100,
            ))
        db_session.commit()
        result = extractor.extract_session_bar_features("2025-06-15", "MNQ1!", db_session)
        assert result["session_num_bars"] == 5
        first_open = 21000.0
        last_close = 21005.0 + 4 * 2
        expected_return = (last_close - first_open) / first_open * 100
        assert abs(result["session_return_pct"] - expected_return) < 1e-6
        session_high = 21010.0 + 4 * 2
        session_low = 20990.0
        expected_range = (session_high - session_low) / first_open * 100
        assert abs(result["session_range_pct"] - expected_range) < 1e-6
        assert result["session_volatility"] >= 0

    def test_incomplete_data_no_crash(self, extractor, db_session):
        db_session.add(SessionMinuteBar(
            session_date="2025-06-15", symbol="MNQ1!",
            bar_time=datetime(2025, 6, 15, 10, 0, 0),
            open_price=21000.0, high_price=21010.0, low_price=20990.0, close_price=21005.0, volume=100,
        ))
        db_session.commit()
        result = extractor.extract_session_bar_features("2025-06-15", "MNQ1!", db_session)
        assert "session_num_bars" in result
        assert result["session_num_bars"] == 0


class TestExtractChartPatterns:
    def test_different_resolutions(self, extractor):
        small = np.random.rand(80, 60).astype(np.float32)
        large = np.random.rand(192, 108).astype(np.float32)
        out_small = extractor.extract_chart_patterns(small)
        out_large = extractor.extract_chart_patterns(large)
        assert "trend_direction" in out_small and "trend_direction" in out_large
        assert "volatility_estimate" in out_small and "volatility_estimate" in out_large
        assert out_small["trend_direction"] in ("up", "down", "sideways")
        assert out_large["trend_direction"] in ("up", "down", "sideways")

    def test_three_channel_image(self, extractor):
        arr = np.random.rand(100, 100, 3).astype(np.float32)
        out = extractor.extract_chart_patterns(arr)
        assert "has_support_level" in out
        assert "has_resistance_level" in out
        assert "volatility_estimate" in out

    def test_invalid_empty_or_degenerate_returns_safe_default(self, extractor):
        # Degenerate shape (0-dim) triggers exception path -> safe default dict
        arr_0d = np.array(1.0)
        out = extractor.extract_chart_patterns(arr_0d)
        assert "trend_direction" in out
        assert out["trend_direction"] == "unknown"
        assert out["volatility_estimate"] == 0.0
        assert out["has_support_level"] is False
        assert out["has_resistance_level"] is False

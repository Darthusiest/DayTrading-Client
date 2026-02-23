"""Tests for predict API routes."""
import io
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from backend.database.models import Prediction

# Routes are mounted at API_V1_PREFIX
PREFIX = "/api/v1"


def _make_png_bytes(width=10, height=10):
    img = Image.new("RGB", (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


class TestPredictPost:
    def test_valid_upload_200(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.api.routes.predict.settings.RAW_DATA_DIR", tmp_path)
        with patch("backend.api.routes.predict.predictor") as m_pred:
            with patch("backend.api.routes.predict.feature_extractor") as m_feat:
                m_pred.model_loaded = True
                m_pred.predict.return_value = {
                    "predicted_price": 21000.0,
                    "probability_hit": 0.75,
                    "model_confidence": 0.75,
                    "learning_score": 75.0,
                }
                m_feat.extract_features.return_value = {"hour": 9, "minute": 30}
                png = _make_png_bytes()
                r = client.post(
                    f"{PREFIX}/predict",
                    files={"file": ("test.png", io.BytesIO(png), "image/png")},
                    data={"symbol": "MNQ1!"},
                )
        if r.status_code != 200:
            raise AssertionError(f"Expected 200, got {r.status_code}: {r.json()}")
        data = r.json()
        assert "prediction_id" in data
        assert data["symbol"] == "MNQ1!"
        assert data["model_predicted_price"] == 21000.0
        assert data["probability_hit"] == 0.75
        assert data["user_expected_price"] is None

    def test_missing_symbol_422(self, client):
        png = _make_png_bytes()
        r = client.post(
            f"{PREFIX}/predict",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
            data={},
        )
        assert r.status_code == 422

    def test_missing_file_422(self, client):
        r = client.post(
            f"{PREFIX}/predict",
            data={"symbol": "MNQ1!"},
        )
        assert r.status_code == 422

    def test_with_expected_price_saved_in_db(self, client, db_session, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.api.routes.predict.settings.RAW_DATA_DIR", tmp_path)
        with patch("backend.api.routes.predict.predictor") as m_pred:
            with patch("backend.api.routes.predict.feature_extractor") as m_feat:
                m_pred.model_loaded = True
                m_pred.predict.return_value = {
                    "predicted_price": 21050.0,
                    "probability_hit": 0.8,
                    "model_confidence": 0.8,
                    "learning_score": 80.0,
                }
                m_feat.extract_features.return_value = {}
                png = _make_png_bytes()
                r = client.post(
                    f"{PREFIX}/predict",
                    files={"file": ("test.png", io.BytesIO(png), "image/png")},
                    data={"symbol": "MNQ1!", "expected_price": "21050.0"},
                )
        assert r.status_code == 200
        assert r.json().get("user_expected_price") == 21050.0
        row = db_session.query(Prediction).order_by(Prediction.id.desc()).first()
        assert row is not None
        assert row.user_expected_price == 21050.0

    def test_without_expected_price_null_in_db(self, client, db_session, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.api.routes.predict.settings.RAW_DATA_DIR", tmp_path)
        with patch("backend.api.routes.predict.predictor") as m_pred:
            with patch("backend.api.routes.predict.feature_extractor") as m_feat:
                m_pred.model_loaded = True
                m_pred.predict.return_value = {
                    "predicted_price": 21000.0,
                    "probability_hit": 0.7,
                    "model_confidence": 0.7,
                    "learning_score": 70.0,
                }
                m_feat.extract_features.return_value = {}
                png = _make_png_bytes()
                r = client.post(
                    f"{PREFIX}/predict",
                    files={"file": ("test.png", io.BytesIO(png), "image/png")},
                    data={"symbol": "MNQ1!"},
                )
        assert r.status_code == 200
        assert r.json().get("user_expected_price") is None
        row = db_session.query(Prediction).order_by(Prediction.id.desc()).first()
        assert row is not None
        assert row.user_expected_price is None


class TestPredictHistory:
    def test_history_empty(self, client):
        r = client.get(f"{PREFIX}/predict/history")
        assert r.status_code == 200
        assert r.json() == []

    def test_history_filter_and_order(self, client, db_session):
        from datetime import datetime
        for i, sym in enumerate(["MNQ1!", "MES1!", "MNQ1!"]):
            db_session.add(Prediction(
                symbol=sym,
                user_expected_price=None,
                model_predicted_price=21000.0 + i,
                probability_hit=0.7,
                screenshot_path=None,
            ))
        db_session.commit()
        r = client.get(f"{PREFIX}/predict/history?symbol=MNQ1!&limit=10")
        assert r.status_code == 200
        items = r.json()
        assert len(items) == 2
        assert all(p["symbol"] == "MNQ1!" for p in items)
        assert items[0]["id"] >= items[1]["id"]

    def test_history_limit(self, client, db_session):
        from datetime import datetime
        for i in range(5):
            db_session.add(Prediction(
                symbol="MNQ1!",
                user_expected_price=None,
                model_predicted_price=21000.0,
                probability_hit=0.7,
                screenshot_path=None,
            ))
        db_session.commit()
        r = client.get(f"{PREFIX}/predict/history?limit=2")
        assert r.status_code == 200
        assert len(r.json()) == 2

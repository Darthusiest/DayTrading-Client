"""Tests for backend.services.ml.inference.predictor."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import numpy as np

from backend.services.ml.inference.predictor import Predictor
from backend.services.ml.models.price_predictor import price_predictor_kwargs_from_settings
from backend.database.models import ModelCheckpoint


@pytest.fixture
def predictor():
    return Predictor(device="cpu")


@patch("backend.services.ml.inference.predictor.PricePredictor")
class TestLoadModel:
    """Mock PricePredictor to avoid loading ResNet (torch cache / network)."""

    def test_existing_checkpoint_in_db_loads(self, mock_price_predictor, predictor, db_session, tmp_path):
        checkpoint_file = tmp_path / "checkpoint_test.pt"
        torch.save({"model_state_dict": {}}, checkpoint_file)
        db_session.add(ModelCheckpoint(
            model_name="test", version="1", checkpoint_path=str(checkpoint_file),
            epoch=1, is_best=True,
        ))
        db_session.commit()
        result = predictor.load_model(db=db_session)
        assert result is True
        assert predictor.model_loaded is True
        assert predictor.model is not None

    def test_no_db_no_file_initializes_fresh(self, mock_price_predictor, predictor, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.ml.inference.predictor.settings.MODELS_DIR", tmp_path)
        result = predictor.load_model(db=None)
        assert result is True
        assert predictor.model_loaded is True
        assert predictor.model is not None

    def test_no_db_but_file_in_models_dir_uses_latest(self, mock_price_predictor, predictor, tmp_path, monkeypatch):
        torch.save({"model_state_dict": {}}, tmp_path / "checkpoint_2.pt")
        monkeypatch.setattr("backend.services.ml.inference.predictor.settings.MODELS_DIR", tmp_path)
        result = predictor.load_model(db=None)
        assert result is True
        assert predictor.model is not None

    def test_checkpoint_path_invalid_or_missing_initializes_fresh(self, mock_price_predictor, predictor):
        result = predictor.load_model(checkpoint_path=Path("/nonexistent/checkpoint.pt"))
        assert result is True
        assert predictor.model_loaded is True
        assert predictor.model is not None

    def test_torch_load_raises_fallback_returns_false(self, mock_price_predictor, predictor, tmp_path):
        (tmp_path / "bad.pt").touch()
        with patch("backend.services.ml.inference.predictor.torch.load", side_effect=RuntimeError("corrupt")):
            result = predictor.load_model(checkpoint_path=tmp_path / "bad.pt")
        assert result is False
        assert predictor.model_loaded is True
        assert predictor.model is not None


class TestPredict:
    def test_valid_preprocess_returns_prediction_dict(self, predictor, tmp_path):
        (tmp_path / "chart.png").touch()
        mock_model = MagicMock()
        mock_model.predict_price.return_value = {
            "predicted_price": np.array([21000.0]),
            "probability": np.array([0.7]),
            "base_confidence": np.array([0.7]),
        }
        predictor.model = mock_model
        predictor.model_loaded = True
        fake_array = np.zeros((224, 224, 3), dtype=np.float32)
        with patch.object(predictor.image_preprocessor, "preprocess", return_value=fake_array):
            result = predictor.predict(tmp_path / "chart.png")
        assert "predicted_price" in result
        assert "probability_hit" in result
        assert "model_confidence" in result
        assert "learning_score" in result
        assert result.get("predicted_price") is not None
        assert result.get("error") is None

    def test_preprocess_returns_none_returns_error_dict(self, predictor, tmp_path):
        (tmp_path / "chart.png").touch()
        predictor.model = MagicMock()
        predictor.model_loaded = True
        with patch.object(predictor.image_preprocessor, "preprocess", return_value=None):
            result = predictor.predict(tmp_path / "chart.png")
        assert "error" in result
        assert result.get("predicted_price") is None

    def test_preprocess_raises_returns_error_dict(self, predictor, tmp_path):
        (tmp_path / "chart.png").touch()
        predictor.model = MagicMock()
        predictor.model_loaded = True
        with patch.object(predictor.image_preprocessor, "preprocess", side_effect=ValueError("bad image")):
            result = predictor.predict(tmp_path / "chart.png")
        assert "error" in result
        assert result.get("predicted_price") is None

    def test_without_expected_price_passes_none_to_model(self, predictor, tmp_path):
        (tmp_path / "chart.png").touch()
        mock_model = MagicMock()
        mock_model.predict_price.return_value = {
            "predicted_price": np.array([21000.0]),
            "probability": np.array([0.7]),
            "base_confidence": np.array([0.7]),
        }
        predictor.model = mock_model
        predictor.model_loaded = True
        fake_array = np.zeros((224, 224, 3), dtype=np.float32)
        with patch.object(predictor.image_preprocessor, "preprocess", return_value=fake_array):
            predictor.predict(tmp_path / "chart.png", expected_price=None)
        call_args = mock_model.predict_price.call_args[0]
        assert call_args[2] is None  # expected_price_tensor

    def test_with_expected_price_passes_tensor_to_model(self, predictor, tmp_path):
        (tmp_path / "chart.png").touch()
        mock_model = MagicMock()
        mock_model.predict_price.return_value = {
            "predicted_price": np.array([21000.0]),
            "probability": np.array([0.8]),
            "base_confidence": np.array([0.8]),
        }
        predictor.model = mock_model
        predictor.model_loaded = True
        fake_array = np.zeros((224, 224, 3), dtype=np.float32)
        with patch.object(predictor.image_preprocessor, "preprocess", return_value=fake_array):
            predictor.predict(tmp_path / "chart.png", expected_price=21050.0)
        call_args = mock_model.predict_price.call_args[0]
        expected_tensor = call_args[2]
        assert expected_tensor is not None
        assert float(expected_tensor[0]) == 21050.0


class TestExtractFeatureVector:
    def test_vector_length_matches_num_features(self, predictor, monkeypatch):
        monkeypatch.setattr("backend.services.ml.inference.predictor.settings.NUM_FEATURES", 18)
        features = {"hour": 10, "minute": 30, "day_of_week": 1}
        vec = predictor._extract_feature_vector(features)
        assert len(vec) == 18

    def test_stable_order(self, predictor, monkeypatch):
        monkeypatch.setattr("backend.services.ml.inference.predictor.settings.NUM_FEATURES", 18)
        features = {"hour": 9, "minute": 31, "day_of_week": 0, "price_change_pct": 0.5,
                    "price_range_pct": 0.2, "session_return_pct": 0.1, "session_range_pct": 0.15,
                    "session_volatility": 0.01, "session_num_bars": 50, "trend_direction": "up",
                    "volatility_estimate": 0.3, "has_support_level": True, "has_resistance_level": False}
        vec1 = predictor._extract_feature_vector(features)
        vec2 = predictor._extract_feature_vector(features)
        assert vec1 == vec2

    def test_missing_keys_use_defaults(self, predictor, monkeypatch):
        monkeypatch.setattr("backend.services.ml.inference.predictor.settings.NUM_FEATURES", 18)
        vec = predictor._extract_feature_vector({})
        assert len(vec) == 18
        assert vec[0] == 0.0
        assert vec[1] == 0.0

"""Inference service for price prediction."""
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
from sqlalchemy.orm import Session
from backend.services.ml.models.price_predictor import PricePredictor
from backend.services.data_processing.image_preprocessor import ImagePreprocessor
from backend.database.models import ModelCheckpoint
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class Predictor:
    """Service for making price predictions."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model: Optional[PricePredictor] = None
        self.image_preprocessor = ImagePreprocessor()
        self.model_loaded = False
    
    def load_model(
        self,
        checkpoint_path: Optional[Path] = None,
        db: Optional[Session] = None
    ) -> bool:
        """
        Load the trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            db: Database session to fetch best checkpoint
        
        Returns:
            True if model loaded successfully
        """
        try:
            # If no checkpoint path provided, get best checkpoint from DB
            if checkpoint_path is None and db is not None:
                best_checkpoint = db.query(ModelCheckpoint).filter(
                    ModelCheckpoint.is_best == True
                ).order_by(ModelCheckpoint.created_at.desc()).first()
                
                if best_checkpoint:
                    checkpoint_path = Path(best_checkpoint.checkpoint_path)
                else:
                    # Try to find latest checkpoint in models directory
                    checkpoints = list(settings.MODELS_DIR.glob("checkpoint_*.pt"))
                    if checkpoints:
                        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    else:
                        logger.warning("No checkpoint found, initializing new model")
                        self.model = PricePredictor()
                        self.model_loaded = True
                        return True
            
            if checkpoint_path is None or not checkpoint_path.exists():
                logger.warning("No checkpoint found, initializing new model")
                self.model = PricePredictor()
                self.model_loaded = True
                return True
            
            # Initialize model
            self.model = PricePredictor()
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"Model loaded from {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Initialize new model as fallback
            self.model = PricePredictor()
            self.model_loaded = True
            return False
    
    def predict(
        self,
        image_path: Path,
        expected_price: Optional[float] = None,
        features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a price prediction from a chart screenshot.
        
        Args:
            image_path: Path to chart screenshot
            expected_price: Optional expected price to calculate probability for
            features: Optional additional features
        
        Returns:
            Dictionary with predictions and metrics
        """
        if not self.model_loaded or self.model is None:
            logger.warning("Model not loaded, attempting to load...")
            self.load_model()
        
        try:
            # Preprocess image
            image_array = self.image_preprocessor.preprocess(image_path)
            if image_array is None:
                raise ValueError(f"Failed to preprocess image: {image_path}")
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Prepare features
            feature_tensor = None
            if features:
                feature_vector = self._extract_feature_vector(features)
                feature_tensor = torch.tensor([feature_vector], dtype=torch.float32).to(self.device)
            
            # Prepare expected price tensor
            expected_price_tensor = None
            if expected_price is not None:
                expected_price_tensor = torch.tensor([expected_price], dtype=torch.float32).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                result = self.model.predict_price(
                    image_tensor,
                    feature_tensor,
                    expected_price_tensor
                )
            
            predicted_price = float(result["predicted_price"][0])
            probability = float(result["probability"][0])
            base_confidence = float(result.get("base_confidence", [probability])[0])
            
            # Calculate learning score (simplified - can be improved)
            learning_score = self._calculate_learning_score(base_confidence)
            
            return {
                "predicted_price": predicted_price,
                "probability_hit": probability,
                "expected_price": expected_price,
                "model_confidence": base_confidence,
                "learning_score": learning_score
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                "error": str(e),
                "predicted_price": None,
                "probability_hit": None
            }
    
    def _extract_feature_vector(self, features: Dict[str, Any]) -> list[float]:
        """Extract numeric features from features dict."""
        feature_list = []
        
        # Time features
        feature_list.append(features.get("hour", 0) / 24.0)
        feature_list.append(features.get("minute", 0) / 60.0)
        feature_list.append(features.get("day_of_week", 0) / 6.0)
        
        # Price features
        feature_list.append(features.get("price_change_pct", 0.0))
        feature_list.append(features.get("price_range_pct", 0.0))
        
        # Pad to fixed size
        target_size = 10
        while len(feature_list) < target_size:
            feature_list.append(0.0)
        return feature_list[:target_size]
    
    def _calculate_learning_score(self, confidence: float) -> float:
        """
        Calculate learning score based on model confidence.
        
        This is a simplified metric - can be improved with actual
        historical performance data.
        """
        # Base score on confidence level
        # Higher confidence = better learning (assuming model is calibrated)
        return confidence * 100.0  # Scale to 0-100

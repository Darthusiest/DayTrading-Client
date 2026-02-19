"""Feature extraction from chart screenshots."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
import pytz
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from chart screenshots and market data."""
    
    def __init__(self):
        self.timezone = pytz.timezone(settings.TIMEZONE)
    
    def extract_features(
        self,
        image_path: Path,
        timestamp: datetime,
        symbol: str,
        price_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract features from chart screenshot and market data.
        
        Args:
            image_path: Path to chart screenshot
            timestamp: Timestamp of the snapshot
            symbol: Trading symbol
            price_data: Optional OHLCV data
        
        Returns:
            Dictionary of extracted features
        """
        features = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "image_path": str(image_path),
        }
        
        # Extract time-based features
        features.update(self._extract_time_features(timestamp))
        
        # Extract price-based features if available
        if price_data:
            features.update(self._extract_price_features(price_data))
        
        return features
    
    def _extract_time_features(self, timestamp: datetime) -> Dict[str, Any]:
        """Extract time-based features."""
        # Convert to timezone-aware if needed
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(self.timezone)
        
        return {
            "hour": timestamp.hour,
            "minute": timestamp.minute,
            "day_of_week": timestamp.weekday(),  # 0=Monday, 6=Sunday
            "day_of_month": timestamp.day,
            "month": timestamp.month,
            "is_weekend": timestamp.weekday() >= 5,
            "is_market_hours": self._is_market_hours(timestamp),
        }
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within NY AM market hours."""
        current_time = timestamp.time()
        before_time = datetime.strptime(settings.BEFORE_SNAPSHOT_TIME, "%H:%M").time()
        after_time = datetime.strptime(settings.AFTER_SNAPSHOT_TIME, "%H:%M").time()
        return before_time <= current_time <= after_time
    
    def _extract_price_features(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from OHLCV price data."""
        features = {}
        
        open_price = price_data.get("open", 0)
        high_price = price_data.get("high", 0)
        low_price = price_data.get("low", 0)
        close_price = price_data.get("close", 0)
        volume = price_data.get("volume", 0)
        
        if close_price > 0:
            # Price change
            features["price_change"] = close_price - open_price
            features["price_change_pct"] = (close_price - open_price) / open_price * 100
            
            # High-low range
            features["price_range"] = high_price - low_price
            features["price_range_pct"] = (high_price - low_price) / close_price * 100
            
            # Body size (candle body)
            features["body_size"] = abs(close_price - open_price)
            features["body_size_pct"] = abs(close_price - open_price) / close_price * 100
            
            # Upper/lower wicks
            features["upper_wick"] = high_price - max(open_price, close_price)
            features["lower_wick"] = min(open_price, close_price) - low_price
        
        if volume > 0:
            features["volume"] = volume
        
        return features
    
    def extract_chart_patterns(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Extract chart patterns using computer vision techniques.
        
        This is a placeholder - actual implementation would use
        CV techniques to detect patterns like support/resistance,
        trend lines, etc.
        
        Args:
            image_array: Preprocessed image as numpy array
        
        Returns:
            Dictionary of detected patterns
        """
        # TODO: Implement actual pattern detection
        # For now, return placeholder structure
        return {
            "has_support_level": False,
            "has_resistance_level": False,
            "trend_direction": "unknown",  # 'up', 'down', 'sideways'
            "volatility_estimate": 0.0,
        }

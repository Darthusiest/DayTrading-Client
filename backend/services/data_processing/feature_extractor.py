"""Feature extraction from chart screenshots."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
from datetime import datetime
import pytz
from backend.config.settings import settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

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
        """Check if timestamp is within session hours (e.g. RTH 9:30–16:00 ET from settings)."""
        current_time = timestamp.time()
        start_time = datetime.strptime(settings.SESSION_START_TIME, "%H:%M").time()
        end_time = datetime.strptime(settings.SESSION_END_TIME, "%H:%M").time()
        return start_time <= current_time <= end_time
    
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
    
    def extract_session_bar_features(
        self,
        session_date: str,
        symbol: str,
        db: "Session",
    ) -> Dict[str, Any]:
        """
        Extract summary features from SessionMinuteBar rows (session start–end from settings).
        Returns dict with session_return_pct, session_range_pct, session_volatility, session_num_bars.
        """
        from backend.database.models import SessionMinuteBar

        features: Dict[str, Any] = {
            "session_return_pct": 0.0,
            "session_range_pct": 0.0,
            "session_volatility": 0.0,
            "session_num_bars": 0,
        }
        try:
            bars = (
                db.query(SessionMinuteBar)
                .filter(
                    SessionMinuteBar.session_date == session_date,
                    SessionMinuteBar.symbol == symbol,
                )
                .order_by(SessionMinuteBar.bar_time)
                .all()
            )
            if not bars or len(bars) < 2:
                return features
            features["session_num_bars"] = len(bars)
            first_open = float(bars[0].open_price)
            last_close = float(bars[-1].close_price)
            session_high = max(float(b.high_price) for b in bars)
            session_low = min(float(b.low_price) for b in bars)
            if first_open > 0:
                features["session_return_pct"] = (last_close - first_open) / first_open * 100
                features["session_range_pct"] = (session_high - session_low) / first_open * 100
            # Minute-to-minute return std as volatility proxy
            returns = []
            for i in range(1, len(bars)):
                prev_c = float(bars[i - 1].close_price)
                curr_c = float(bars[i].close_price)
                if prev_c > 0:
                    returns.append((curr_c - prev_c) / prev_c)
            if returns:
                features["session_volatility"] = float(np.std(returns))
        except Exception as e:
            logger.debug("Could not extract session bar features for %s %s: %s", symbol, session_date, e)
        return features
    
    def extract_chart_patterns(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Extract chart patterns using simple CV: trend from row-wise profile,
        volatility from gradient std, support/resistance from local extrema.
        
        Args:
            image_array: Preprocessed image as numpy array (H, W) or (H, W, C)
        
        Returns:
            Dictionary of detected patterns
        """
        try:
            if image_array.ndim == 3:
                gray = np.mean(image_array, axis=2).astype(np.float32)
            else:
                gray = image_array.astype(np.float32)
            h, w = gray.shape

            # Price-like profile: mean intensity per row (chart often has price on vertical axis)
            profile = np.mean(gray, axis=1)
            profile = profile - np.mean(profile)
            x = np.arange(len(profile), dtype=np.float32)
            # Linear trend: slope
            slope = np.polyfit(x, profile, 1)[0]
            # Normalize slope by profile std for scale-invariance
            profile_std = np.std(profile)
            if profile_std > 1e-6:
                slope_norm = slope / profile_std * 100
            else:
                slope_norm = 0.0

            if slope_norm > 0.5:
                trend_direction = "up"
            elif slope_norm < -0.5:
                trend_direction = "down"
            else:
                trend_direction = "sideways"

            # Volatility: std of row-to-row differences (price change proxy)
            diffs = np.diff(profile)
            volatility_estimate = float(np.std(diffs)) if len(diffs) > 0 else 0.0
            # Normalize to [0, 1] range for typical preprocessed images
            volatility_estimate = min(1.0, volatility_estimate / 20.0)

            # Local extrema in profile (support = local min, resistance = local max)
            k = max(2, len(profile) // 15)
            extended = np.concatenate([[profile[0]] * k, profile, [profile[-1]] * k])
            local_min = np.array(
                [np.min(extended[i : i + 2 * k + 1]) for i in range(len(profile))]
            )
            local_max = np.array(
                [np.max(extended[i : i + 2 * k + 1]) for i in range(len(profile))]
            )
            has_support = (
                np.any(profile <= local_min + 1e-6)
                and np.mean(profile[: h // 3]) <= np.mean(profile) + 1e-6
            )
            has_resistance = (
                np.any(profile >= local_max - 1e-6)
                and np.mean(profile[-h // 3 :]) >= np.mean(profile) - 1e-6
            )

            return {
                "has_support_level": bool(has_support),
                "has_resistance_level": bool(has_resistance),
                "trend_direction": trend_direction,
                "volatility_estimate": volatility_estimate,
            }
        except Exception as e:
            logger.debug("Chart pattern extraction failed: %s", e)
            return {
                "has_support_level": False,
                "has_resistance_level": False,
                "trend_direction": "unknown",
                "volatility_estimate": 0.0,
            }

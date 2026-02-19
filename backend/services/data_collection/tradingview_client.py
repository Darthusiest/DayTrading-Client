"""Polygon.io client for fetching market data."""
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import pytz
from polygon import RESTClient
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class PolygonClient:
    """Client for interacting with Polygon.io API."""
    
    def __init__(self):
        self.api_key = settings.POLYGON_API_KEY
        if not self.api_key:
            logger.warning("Polygon.io API key not set. Set POLYGON_API_KEY environment variable.")
        self.client = RESTClient(self.api_key) if self.api_key else None
        self.timezone = pytz.timezone(settings.TIMEZONE)
        self.symbol_map = settings.POLYGON_SYMBOL_MAP
    
    def _map_symbol(self, symbol: str) -> str:
        """Map user-facing symbol to Polygon.io format."""
        return self.symbol_map.get(symbol, symbol)
    
    def get_price_data(
        self,
        symbol: str,
        timestamp: datetime,
        timeframe: str = "1"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch OHLCV price data for a symbol at a specific timestamp.
        
        Args:
            symbol: Trading symbol (e.g., 'NQ1!', 'ES1!')
            timestamp: Datetime to fetch data for
            timeframe: Timeframe in minutes (default: 1 minute)
        
        Returns:
            Dictionary with OHLCV data or None if failed
        """
        if not self.client:
            logger.error("Polygon.io client not initialized. Check API key.")
            return None
        
        try:
            # Map symbol to Polygon.io format
            polygon_symbol = self._map_symbol(symbol)
            
            # Convert timeframe to Polygon.io format
            # Polygon.io uses multiplier and timespan
            multiplier = int(timeframe)
            timespan = "minute"  # Can be: minute, hour, day, week, month, quarter, year
            
            # Convert timestamp to timezone-aware if needed
            if timestamp.tzinfo is None:
                timestamp = self.timezone.localize(timestamp)
            else:
                timestamp = timestamp.astimezone(self.timezone)
            
            # Get data for the specific minute
            # Polygon.io aggregates endpoint requires from/to dates
            from_time = timestamp
            to_time = timestamp + timedelta(minutes=1)
            
            # Convert to Unix timestamp in milliseconds
            from_ts = int(from_time.timestamp() * 1000)
            to_ts = int(to_time.timestamp() * 1000)
            
            # Fetch aggregates
            aggs = self.client.get_aggs(
                ticker=polygon_symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_ts,
                to=to_ts,
                limit=1
            )
            
            if aggs and len(aggs) > 0:
                agg = aggs[0]
                return {
                    "symbol": symbol,
                    "timestamp": timestamp.isoformat(),
                    "open": float(agg.open) if agg.open else 0.0,
                    "high": float(agg.high) if agg.high else 0.0,
                    "low": float(agg.low) if agg.low else 0.0,
                    "close": float(agg.close) if agg.close else 0.0,
                    "volume": int(agg.volume) if agg.volume else 0,
                    "timeframe": timeframe,
                    "vwap": float(agg.vwap) if hasattr(agg, 'vwap') and agg.vwap else None
                }
            else:
                logger.warning(f"No data returned from Polygon.io for {symbol} at {timestamp}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching price data from Polygon.io for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/latest price for a symbol."""
        if not self.client:
            logger.error("Polygon.io client not initialized. Check API key.")
            return None
        
        try:
            polygon_symbol = self._map_symbol(symbol)
            
            # Get previous close (most recent available price)
            prev_close = self.client.get_previous_close_agg(ticker=polygon_symbol)
            
            if prev_close and hasattr(prev_close, 'close'):
                return float(prev_close.close)
            
            # Fallback: get latest trade
            try:
                trades = self.client.list_trades(ticker=polygon_symbol, limit=1)
                if trades and len(trades) > 0:
                    return float(trades[0].price)
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching current price from Polygon.io for {symbol}: {e}")
            return None
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        multiplier: int = 1,
        timespan: str = "minute"
    ) -> Optional[list[Dict[str, Any]]]:
        """
        Get historical OHLCV data for a date range.
        
        Args:
            symbol: Trading symbol
            start_date: Start datetime
            end_date: End datetime
            multiplier: Size of timespan multiplier
            timespan: Size of time window (minute, hour, day, etc.)
        
        Returns:
            List of OHLCV dictionaries
        """
        if not self.client:
            logger.error("Polygon.io client not initialized. Check API key.")
            return None
        
        try:
            polygon_symbol = self._map_symbol(symbol)
            
            # Convert to Unix timestamps in milliseconds
            from_ts = int(start_date.timestamp() * 1000)
            to_ts = int(end_date.timestamp() * 1000)
            
            # Fetch aggregates
            aggs = self.client.get_aggs(
                ticker=polygon_symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_ts,
                to=to_ts,
                limit=50000  # Max limit
            )
            
            if not aggs:
                return None
            
            results = []
            for agg in aggs:
                results.append({
                    "symbol": symbol,
                    "timestamp": datetime.fromtimestamp(agg.timestamp / 1000, tz=self.timezone).isoformat(),
                    "open": float(agg.open) if agg.open else 0.0,
                    "high": float(agg.high) if agg.high else 0.0,
                    "low": float(agg.low) if agg.low else 0.0,
                    "close": float(agg.close) if agg.close else 0.0,
                    "volume": int(agg.volume) if agg.volume else 0,
                    "vwap": float(agg.vwap) if hasattr(agg, 'vwap') and agg.vwap else None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching historical data from Polygon.io for {symbol}: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        # NY AM session: 6:30 AM - 8:00 AM PST
        now = datetime.now(self.timezone)
        current_time = now.time()
        
        # Check if it's a weekday (Monday-Friday)
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if within NY AM hours (6:30 AM - 8:00 AM PST)
        before_time = datetime.strptime(settings.BEFORE_SNAPSHOT_TIME, "%H:%M").time()
        after_time = datetime.strptime(settings.AFTER_SNAPSHOT_TIME, "%H:%M").time()
        
        return before_time <= current_time <= after_time
    
    def get_session_date(self) -> str:
        """Get current session date in YYYY-MM-DD format."""
        now = datetime.now(self.timezone)
        return now.strftime("%Y-%m-%d")


# Alias for backward compatibility
TradingViewClient = PolygonClient

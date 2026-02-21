# Polygon.io Integration Guide

## Overview

The Day Trading AI Agent backend uses Polygon.io for fetching real-time and historical market data (OHLCV) for Nasdaq and S&P 500 futures contracts.

## API Key Setup

1. **Get your API key**:
   - Sign up at https://polygon.io/
   - Navigate to Dashboard → Keys
   - Copy your API key

2. **Add to environment**:
   ```bash
   # In your .env file
   POLYGON_API_KEY=your_polygon_api_key
   ```

## Symbol Mapping

The system maps user-facing symbols to Polygon.io format:

- `MNQ1!` → `C:MNQ1` (Micro E-mini Nasdaq continuous)
- `MES1!` → `C:MES1` (Micro E-mini S&P 500 continuous)

The `C:` prefix indicates continuous contracts, which automatically roll to the front month.

## Usage

### Basic Price Data Fetching

```python
from backend.services.data_collection.tradingview_client import PolygonClient
from datetime import datetime
import pytz

client = PolygonClient()

# Get price data for a specific timestamp
timestamp = datetime.now(pytz.timezone("America/Los_Angeles"))
price_data = client.get_price_data("MNQ1!", timestamp, timeframe="1")

if price_data:
    print(f"Open: {price_data['open']}")
    print(f"High: {price_data['high']}")
    print(f"Low: {price_data['low']}")
    print(f"Close: {price_data['close']}")
    print(f"Volume: {price_data['volume']}")
```

### Get Current Price

```python
current_price = client.get_current_price("MNQ1!")
print(f"Current price: {current_price}")
```

### Get Historical Data

```python
from datetime import datetime, timedelta
import pytz

timezone = pytz.timezone("America/Los_Angeles")
start_date = datetime.now(timezone) - timedelta(days=7)
end_date = datetime.now(timezone)

historical_data = client.get_historical_data(
    symbol="MNQ1!",
    start_date=start_date,
    end_date=end_date,
    multiplier=1,      # 1 minute bars
    timespan="minute"  # minute, hour, day, week, month
)

for bar in historical_data:
    print(f"{bar['timestamp']}: Close = {bar['close']}")
```

## WebSocket (real-time stream)

When the API server runs, an optional **Polygon WebSocket** connection streams minute-aggregate bars for the configured symbols (C:MNQ1, C:MES1). This runs in a background thread and keeps the latest bar per symbol in memory.

- **Enable/disable**: Set `ENABLE_POLYGON_WEBSOCKET=False` in `.env` to turn it off (default: true when `POLYGON_API_KEY` is set).
- **Live prices endpoint**: `GET /api/v1/live/prices` returns the latest minute bar (OHLCV + timestamps) per symbol from the stream. Empty or partial if the stream has not connected or no data has arrived yet.
- **Subscriptions**: The client subscribes to `AM.C:MNQ1` and `AM.C:MES1` (minute aggregates for the same symbols used by the REST client). Requires a Polygon plan that includes futures WebSocket access.

## API Methods

### `get_price_data(symbol, timestamp, timeframe="1")`

Fetches OHLCV data for a specific timestamp.

**Parameters:**
- `symbol`: Trading symbol ("MNQ1!" or "MES1!")
- `timestamp`: Datetime object (timezone-aware)
- `timeframe`: Timeframe in minutes (default: "1")

**Returns:**
- Dictionary with OHLCV data or `None` if failed

### `get_current_price(symbol)`

Gets the current/latest price for a symbol.

**Parameters:**
- `symbol`: Trading symbol

**Returns:**
- Float price or `None` if failed

### `get_historical_data(symbol, start_date, end_date, multiplier=1, timespan="minute")`

Gets historical OHLCV data for a date range.

**Parameters:**
- `symbol`: Trading symbol
- `start_date`: Start datetime
- `end_date`: End datetime
- `multiplier`: Size of timespan multiplier (default: 1)
- `timespan`: Size of time window - "minute", "hour", "day", "week", "month" (default: "minute")

**Returns:**
- List of OHLCV dictionaries or `None` if failed

## Error Handling

The client includes comprehensive error handling and logging:

- Invalid API key: Logs warning and returns `None`
- Network errors: Logs error and returns `None`
- Invalid symbols: Polygon.io API will return error, logged and handled
- Rate limiting: Polygon.io has rate limits based on your plan

## Rate Limits

Polygon.io has different rate limits based on your subscription plan:

- **Starter**: 5 API calls per minute
- **Developer**: 200 API calls per minute
- **Advanced**: 1,000 API calls per minute

Check your plan limits at https://polygon.io/pricing

## Troubleshooting

### "Polygon.io client not initialized"

**Solution**: Make sure `POLYGON_API_KEY` is set in your `.env` file and the environment is loaded correctly.

### "No data returned from Polygon.io"

**Possible causes**:
1. Market is closed (no data for that timestamp)
2. Invalid symbol format
3. Date range too large (exceeds API limits)
4. API rate limit exceeded

**Solution**: Check logs for specific error messages, verify symbol format, and ensure market hours.

### Symbol not found

**Solution**: Verify the symbol exists on Polygon.io. Futures symbols may need different formatting. Check Polygon.io documentation for symbol formats.

## Resources

- [Polygon.io Documentation](https://polygon.io/docs)
- [Polygon.io Python SDK](https://github.com/polygon-io/client-python)
- [Polygon.io API Reference](https://polygon.io/docs/stocks/getting-started)

"""Live market data from Polygon WebSocket stream."""
from fastapi import APIRouter
from backend.services.data_collection.polygon_websocket import get_latest_bars

router = APIRouter(prefix="/live", tags=["live"])


@router.get("/prices")
def get_live_prices():
    """
    Return the latest minute-aggregate bar per symbol from the Polygon WebSocket stream.

    When the WebSocket is enabled and connected, this returns real-time OHLCV for
    MNQ1! and MES1! (and any other configured symbols). If the stream is not running
    or no data has been received yet, the response may be empty or partial.
    """
    return {"prices": get_latest_bars()}

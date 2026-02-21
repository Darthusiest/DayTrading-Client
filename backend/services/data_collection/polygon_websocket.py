"""Polygon.io WebSocket client for real-time futures minute aggregates (e.g. C:MNQ1, C:MES1)."""
import logging
import threading
from typing import Any, Dict, List, Optional

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# Optional: only import when WebSocket is enabled so REST-only setups don't need websockets
try:
    from polygon import WebSocketClient
    from polygon.websocket.models import WebSocketMessage, EquityAgg
    POLYGON_WS_AVAILABLE = True
except Exception as e:
    logger.debug("Polygon WebSocket client not available: %s", e)
    WebSocketClient = None
    WebSocketMessage = None
    EquityAgg = None
    POLYGON_WS_AVAILABLE = False


# In-memory store of latest minute bar per symbol (user-facing symbol, e.g. MNQ1!)
_latest_bars: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_ws_thread: Optional[threading.Thread] = None
_ws_client: Optional["WebSocketClient"] = None
_ws_stop = threading.Event()


def _get_polygon_subscriptions() -> List[str]:
    """Subscription params for minute aggregates: AM.<ticker> per symbol."""
    tickers = list(settings.POLYGON_SYMBOL_MAP.values())
    return [f"AM.{t}" for t in tickers]


def _reverse_symbol_map() -> Dict[str, str]:
    """Polygon ticker -> user-facing symbol (e.g. C:MNQ1 -> MNQ1!)."""
    return {v: k for k, v in settings.POLYGON_SYMBOL_MAP.items()}


def _handle_ws_message(msgs: List[WebSocketMessage]) -> None:
    """Process parsed WebSocket messages; update _latest_bars."""
    rev = _reverse_symbol_map()
    for m in msgs:
        if EquityAgg is not None and isinstance(m, EquityAgg):
            sym = getattr(m, "symbol", None) or getattr(m, "sym", None)
            if not sym:
                continue
            user_sym = rev.get(sym, sym)
            bar = {
                "symbol": user_sym,
                "open": getattr(m, "open", None),
                "high": getattr(m, "high", None),
                "low": getattr(m, "low", None),
                "close": getattr(m, "close", None),
                "volume": getattr(m, "volume", None) or getattr(m, "accumulated_volume", None),
                "start_timestamp": getattr(m, "start_timestamp", None),
                "end_timestamp": getattr(m, "end_timestamp", None),
            }
            with _lock:
                _latest_bars[user_sym] = bar
            logger.debug("WebSocket bar %s: close=%s", user_sym, bar.get("close"))


def _run_ws_loop() -> None:
    """Run WebSocket client in this thread (blocking)."""
    global _ws_client
    if not POLYGON_WS_AVAILABLE or not settings.POLYGON_API_KEY:
        logger.warning("Polygon WebSocket skipped: client unavailable or no API key")
        return
    subs = _get_polygon_subscriptions()
    if not subs:
        return
    try:
        # market="futures" for C:MNQ1, C:MES1 (polygon-api-client may not have Market.Futures enum)
        _ws_client = WebSocketClient(
            api_key=settings.POLYGON_API_KEY,
            market="futures",
            subscriptions=subs,
            max_reconnects=10,
            verbose=logger.isEnabledFor(logging.DEBUG),
        )
        logger.info("Polygon WebSocket connecting (subscriptions: %s)", subs)
        _ws_client.run(_handle_ws_message)
    except Exception as e:
        logger.exception("Polygon WebSocket error: %s", e)
    finally:
        _ws_client = None
        logger.info("Polygon WebSocket stopped")


def start_polygon_websocket() -> bool:
    """Start the Polygon WebSocket in a background thread. Returns True if started."""
    global _ws_thread
    if not getattr(settings, "ENABLE_POLYGON_WEBSOCKET", True):
        logger.info("Polygon WebSocket disabled (ENABLE_POLYGON_WEBSOCKET=False)")
        return False
    if not settings.POLYGON_API_KEY:
        logger.warning("Polygon WebSocket skipped: POLYGON_API_KEY not set")
        return False
    if not POLYGON_WS_AVAILABLE:
        logger.warning("Polygon WebSocket skipped: websocket client not available")
        return False
    if _ws_thread is not None and _ws_thread.is_alive():
        logger.debug("Polygon WebSocket already running")
        return True
    _ws_stop.clear()
    _ws_thread = threading.Thread(target=_run_ws_loop, name="polygon-websocket", daemon=True)
    _ws_thread.start()
    return True


def stop_polygon_websocket() -> None:
    """Stop the WebSocket (best-effort; client may need to be closed from inside the loop)."""
    global _ws_client
    _ws_stop.set()
    if _ws_client is not None and hasattr(_ws_client, "websocket") and _ws_client.websocket:
        try:
            import asyncio
            asyncio.run(_ws_client.close())
        except Exception as e:
            logger.debug("Error closing WebSocket: %s", e)
    _ws_client = None


def get_latest_bars() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the latest minute bar per symbol (from WebSocket stream)."""
    with _lock:
        return dict(_latest_bars)

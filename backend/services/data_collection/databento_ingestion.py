"""Ingest Databento OHLCV-1m batch files into SessionMinuteBar."""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import databento as db
import pytz
from sqlalchemy.orm import Session

from backend.config.settings import settings
from backend.database.models import SessionMinuteBar

logger = logging.getLogger(__name__)


def _build_instrument_id_to_raw_symbol(store: db.DBNStore) -> dict[int, str]:
    """Build mapping instrument_id -> raw_symbol from store.mappings."""
    out: dict[int, str] = {}
    for raw_symbol, intervals in (store.mappings or {}).items():
        for iv in intervals:
            sid = iv.get("symbol")
            if sid is not None:
                out[int(sid)] = raw_symbol
    return out


def _raw_symbol_to_app_symbol(raw_symbol: str) -> Optional[str]:
    """Map Databento raw_symbol to app symbol (e.g. MNQ1!, MES1!) via settings.DATABENTO_SYMBOL_MAP."""
    for pattern, app_sym in settings.DATABENTO_SYMBOL_MAP.items():
        if pattern in raw_symbol:
            return app_sym
    return None


def _in_rth(
    dt_utc: datetime,
    session_tz: pytz.BaseTzInfo,
    start_h: int,
    start_m: int,
    end_h: int,
    end_m: int,
) -> tuple[bool, Optional[str], Optional[datetime]]:
    """
    Convert UTC bar time to session TZ and check if within RTH.
    Returns (in_range, session_date_YYYY_MM_DD, naive_bar_time_in_session_tz).
    """
    if dt_utc.tzinfo is None:
        dt_utc = pytz.UTC.localize(dt_utc)
    local = dt_utc.astimezone(session_tz)
    session_date = local.strftime("%Y-%m-%d")
    t = local.time()
    start_ok = (local.hour, local.minute) >= (start_h, start_m)
    end_ok = (local.hour, local.minute) <= (end_h, end_m)
    if not (start_ok and end_ok):
        return False, None, None
    naive = local.replace(tzinfo=None)
    return True, session_date, naive


def _discover_dbn_files(root: Path, path_override: Optional[Path] = None) -> list[Path]:
    """Discover *.dbn.zst files under root, or use path_override (file or dir)."""
    if path_override is not None:
        if path_override.is_file():
            return [path_override] if path_override.suffix == ".zst" and ".dbn" in path_override.name else []
        if path_override.is_dir():
            return sorted(path_override.rglob("*.dbn.zst"))
        return []
    return sorted(root.rglob("*.dbn.zst"))


def ingest_from_store(
    store: db.DBNStore,
    session_tz: pytz.BaseTzInfo,
    start_h: int,
    start_m: int,
    end_h: int,
    end_m: int,
    id_to_raw: dict[int, str],
    dry_run: bool = False,
) -> list[dict]:
    """
    Read OHLCV-1m records from store, filter RTH, map to app symbols.
    Returns list of dicts ready for SessionMinuteBar (session_date, symbol, bar_time, open_price, ...).
    """
    bars: list[dict] = []
    stats = {"skipped_unknown": 0, "skipped_outside_rth": 0}

    def callback(rec):
        if "OHLCV" not in type(rec).__name__:
            return
        instrument_id = getattr(rec, "instrument_id", None)
        if instrument_id is None:
            return
        raw_symbol = id_to_raw.get(int(instrument_id))
        if not raw_symbol:
            stats["skipped_unknown"] += 1
            return
        app_symbol = _raw_symbol_to_app_symbol(raw_symbol)
        if app_symbol not in settings.SYMBOLS:
            return
        ts = getattr(rec, "pretty_ts_event", None)
        if ts is None and hasattr(rec, "ts_event"):
            ns = rec.ts_event
            ts = datetime.utcfromtimestamp(ns / 1e9)
            ts = pytz.UTC.localize(ts)
        if ts is None:
            return
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = pytz.UTC.localize(ts)
        in_range, session_date, bar_time = _in_rth(ts, session_tz, start_h, start_m, end_h, end_m)
        if not in_range or session_date is None or bar_time is None:
            stats["skipped_outside_rth"] += 1
            return
        open_p = getattr(rec, "pretty_open", None)
        if open_p is None and getattr(rec, "open", None) is not None:
            open_p = rec.open / 1e9
        high_p = getattr(rec, "pretty_high", None) or (rec.high / 1e9 if getattr(rec, "high", None) else 0)
        low_p = getattr(rec, "pretty_low", None) or (rec.low / 1e9 if getattr(rec, "low", None) else 0)
        close_p = getattr(rec, "pretty_close", None) or (rec.close / 1e9 if getattr(rec, "close", None) else 0)
        vol = getattr(rec, "volume", None) or 0
        bars.append({
            "session_date": session_date,
            "symbol": app_symbol,
            "bar_time": bar_time,
            "open_price": float(open_p or 0),
            "high_price": float(high_p),
            "low_price": float(low_p),
            "close_price": float(close_p),
            "volume": int(vol) if vol is not None else None,
        })

    try:
        store.replay(callback)
    except Exception as e:
        logger.exception("Replay failed: %s", e)
        raise
    if stats["skipped_unknown"]:
        logger.debug("Skipped %s records with unknown instrument_id", stats["skipped_unknown"])
    if stats["skipped_outside_rth"]:
        logger.debug("Skipped %s records outside RTH", stats["skipped_outside_rth"])
    return bars


def run_ingestion(
    db_session: Session,
    path_override: Optional[Path] = None,
    dry_run: bool = False,
) -> dict:
    """
    Discover .dbn.zst files under settings.DATABENTO_RAW_DIR (or path_override),
    decode OHLCV-1m, map symbols, filter RTH, and insert into SessionMinuteBar.
    Idempotent: for each (session_date, symbol) present in the batch, deletes existing bars then inserts.
    Returns summary: files_processed, bars_inserted, errors.
    """
    root = Path(settings.DATABENTO_RAW_DIR)
    files = _discover_dbn_files(root, path_override)
    if not files:
        return {"files_processed": 0, "bars_inserted": 0, "errors": ["No .dbn.zst files found"]}

    session_tz = pytz.timezone(settings.SESSION_TIMEZONE)
    start_parts = settings.SESSION_START_TIME.strip().split(":")
    end_parts = settings.SESSION_END_TIME.strip().split(":")
    start_h, start_m = int(start_parts[0]), int(start_parts[1]) if len(start_parts) > 1 else 0
    end_h, end_m = int(end_parts[0]), int(end_parts[1]) if len(end_parts) > 1 else 0

    total_bars = 0
    errors: list[str] = []

    for filepath in files:
        try:
            store = db.DBNStore.from_file(str(filepath))
            if getattr(store, "schema", None) and "ohlcv" not in str(store.schema).lower():
                logger.warning("Skip %s: schema %s is not OHLCV", filepath.name, store.schema)
                continue
            id_to_raw = _build_instrument_id_to_raw_symbol(store)
            bars = ingest_from_store(
                store, session_tz, start_h, start_m, end_h, end_m, id_to_raw, dry_run=dry_run
            )
            if dry_run:
                total_bars += len(bars)
                continue
            if not bars:
                continue
            # Group by (session_date, symbol) for idempotent replace
            by_key: dict[tuple[str, str], list[dict]] = {}
            for b in bars:
                key = (b["session_date"], b["symbol"])
                by_key.setdefault(key, []).append(b)
            for (session_date, symbol), group in by_key.items():
                try:
                    db_session.query(SessionMinuteBar).filter(
                        SessionMinuteBar.session_date == session_date,
                        SessionMinuteBar.symbol == symbol,
                    ).delete()
                    for b in group:
                        db_session.add(SessionMinuteBar(**b))
                    total_bars += len(group)
                except Exception as e:
                    db_session.rollback()
                    errors.append(f"{session_date} {symbol}: {e}")
                    logger.exception("Insert failed for %s %s", session_date, symbol)
                    raise
            db_session.commit()
        except Exception as e:
            errors.append(f"{filepath.name}: {e}")
            logger.exception("Process %s failed: %s", filepath, e)
            db_session.rollback()

    return {
        "files_processed": len(files),
        "bars_inserted": total_bars,
        "errors": errors,
    }

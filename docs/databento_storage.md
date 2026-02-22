# Databento data storage

## Where to put the batch download files

- **Raw downloads (compressed):**  
  Put the files you get from Databento (e.g. `glbx-mdp3-YYYYMMDD.ohlcv-1m.dbn.zst` and `condition.json`) in:

  **`data/databento/raw/`**

  This path is defined in config as `settings.DATABENTO_RAW_DIR`. The app creates this directory on startup.

- **Processed data (canonical):**  
  After you run an ingestion step that reads the `.dbn.zst` files (e.g. with the `dbn` Python library), the minute bars are stored in the **database** in the **`session_minute_bars`** table (`SessionMinuteBar` model). That is the single source of truth for OHLCV-1m data used by training, labels, and evaluation.

## Summary

| What | Location |
|------|----------|
| Raw batch files (`.dbn.zst`, `condition.json`) | `data/databento/raw/` |
| Ingested minute bars | DB table `session_minute_bars` |
| Other app data (screenshots, processed images, DB file) | `data/raw/`, `data/processed/`, `data/daytrade.db` |

## Next step

Implement an ingestion script or service that:

1. Reads `.dbn.zst` files from `settings.DATABENTO_RAW_DIR`.
2. Decodes OHLCV-1m records (Databento schema `ohlcv-1m`).
3. Maps instrument IDs to your symbols (e.g. MNQ, MES).
4. Filters to RTH (e.g. 9:30–16:00 ET) if the files contain full globex.
5. Inserts/upserts into `SessionMinuteBar` (session_date, symbol, bar_time, open/high/low/close, volume).

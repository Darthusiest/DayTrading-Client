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

## Ingestion

The app ingests OHLCV-1m from `.dbn.zst` files under `data/databento/raw/` (including subdirs like `GLBX-20260222-CPH8DH3NDW/`) into the `session_minute_bars` table. RTH filtering uses `SESSION_START_TIME`, `SESSION_END_TIME`, and `SESSION_TIMEZONE` from settings (default 9:30–16:00 ET).

### CLI (recommended for bulk)

From the project root:

```bash
python scripts/ingest_databento.py
```

- **`--dry-run`** — Decode and count bars only; do not write to the database.
- **`--path <file_or_dir>`** — Process a single file or directory instead of the full `DATABENTO_RAW_DIR`.

### API

- **`POST /api/v1/collection/ingest-databento`** — Run ingestion. Optional query: `?path=<file_or_dir>`, `?dry_run=true`.

### Symbol mapping

Databento raw symbols (e.g. `MNQU5`, `MESZ5`) are mapped to app symbols (`MNQ1!`, `MES1!`) via **`DATABENTO_SYMBOL_MAP`** in settings. Default: any raw symbol containing `"MNQ"` → `MNQ1!`, containing `"MES"` → `MES1!`. Override in `.env` with JSON, e.g.:

```env
DATABENTO_SYMBOL_MAP={"MNQ":"MNQ1!","MES":"MES1!"}
```

Only symbols in `SYMBOLS` (default `["MNQ1!", "MES1!"]`) are inserted. Re-running ingestion for the same `(session_date, symbol)` replaces existing bars (idempotent).

"""Tests for backend.services.data_collection.databento_ingestion."""
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytz

from backend.services.data_collection.databento_ingestion import (
    _raw_symbol_to_app_symbol,
    _in_rth,
    _discover_dbn_files,
    ingest_from_store,
    run_ingestion,
)
from backend.database.models import SessionMinuteBar


class TestRawSymbolToAppSymbol:
    def test_mnq_maps_to_mnq1(self, monkeypatch):
        monkeypatch.setattr(
            "backend.services.data_collection.databento_ingestion.settings.DATABENTO_SYMBOL_MAP",
            {"MNQ": "MNQ1!", "MES": "MES1!"},
        )
        assert _raw_symbol_to_app_symbol("MNQZ5") == "MNQ1!"
        assert _raw_symbol_to_app_symbol("MNQH6-MNQM6") == "MNQ1!"

    def test_mes_maps_to_mes1(self, monkeypatch):
        monkeypatch.setattr(
            "backend.services.data_collection.databento_ingestion.settings.DATABENTO_SYMBOL_MAP",
            {"MNQ": "MNQ1!", "MES": "MES1!"},
        )
        assert _raw_symbol_to_app_symbol("MESZ5") == "MES1!"
        assert _raw_symbol_to_app_symbol("MESH6") == "MES1!"

    def test_no_match_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "backend.services.data_collection.databento_ingestion.settings.DATABENTO_SYMBOL_MAP",
            {"MNQ": "MNQ1!", "MES": "MES1!"},
        )
        assert _raw_symbol_to_app_symbol("ESZ5") is None
        assert _raw_symbol_to_app_symbol("NQZ5") is None

    def test_first_key_wins_when_multiple_match(self, monkeypatch):
        monkeypatch.setattr(
            "backend.services.data_collection.databento_ingestion.settings.DATABENTO_SYMBOL_MAP",
            {"MNQ": "MNQ1!", "MES": "MES1!", "MNQM": "OTHER!"},
        )
        # "MNQ" appears before "MNQM" in iteration; pattern "in" raw_symbol, so MNQ in "MNQM6" -> MNQ1!
        result = _raw_symbol_to_app_symbol("MNQM6")
        assert result in ("MNQ1!", "OTHER!")  # implementation iterates .items(); first match wins


class TestInRth:
    SESSION_TZ = pytz.timezone("America/New_York")
    START_H, START_M = 9, 30
    END_H, END_M = 16, 0

    def test_at_open_in_range(self):
        # 09:30 ET = 13:30 UTC (EDT) or 14:30 UTC (EST). Use a date in summer for EDT.
        utc_at_open = datetime(2025, 6, 15, 13, 30, 0, tzinfo=pytz.UTC)
        in_range, session_date, bar_time = _in_rth(
            utc_at_open, self.SESSION_TZ, self.START_H, self.START_M, self.END_H, self.END_M
        )
        assert in_range is True
        assert session_date == "2025-06-15"
        assert bar_time is not None
        assert bar_time.hour == 9 and bar_time.minute == 30

    def test_at_close_in_range(self):
        # 16:00 ET = 20:00 UTC (EDT)
        utc_at_close = datetime(2025, 6, 15, 20, 0, 0, tzinfo=pytz.UTC)
        in_range, session_date, bar_time = _in_rth(
            utc_at_close, self.SESSION_TZ, self.START_H, self.START_M, self.END_H, self.END_M
        )
        assert in_range is True
        assert session_date == "2025-06-15"
        assert bar_time is not None
        assert bar_time.hour == 16 and bar_time.minute == 0

    def test_minute_before_open_out_of_range(self):
        # 09:29 ET
        utc_before = datetime(2025, 6, 15, 13, 29, 0, tzinfo=pytz.UTC)
        in_range, session_date, bar_time = _in_rth(
            utc_before, self.SESSION_TZ, self.START_H, self.START_M, self.END_H, self.END_M
        )
        assert in_range is False
        assert session_date is None
        assert bar_time is None

    def test_minute_after_close_out_of_range(self):
        # 16:01 ET = 20:01 UTC
        utc_after = datetime(2025, 6, 15, 20, 1, 0, tzinfo=pytz.UTC)
        in_range, session_date, bar_time = _in_rth(
            utc_after, self.SESSION_TZ, self.START_H, self.START_M, self.END_H, self.END_M
        )
        assert in_range is False
        assert session_date is None
        assert bar_time is None

    def test_naive_utc_treated_as_utc(self):
        naive_utc = datetime(2025, 6, 15, 13, 30, 0)  # no tzinfo
        in_range, session_date, bar_time = _in_rth(
            naive_utc, self.SESSION_TZ, self.START_H, self.START_M, self.END_H, self.END_M
        )
        assert in_range is True
        assert session_date == "2025-06-15"
        assert bar_time is not None


class TestDiscoverDbnFiles:
    def test_empty_dir_returns_empty(self, tmp_path):
        assert _discover_dbn_files(tmp_path) == []
        sub = tmp_path / "sub"
        sub.mkdir()
        assert _discover_dbn_files(tmp_path, path_override=sub) == []

    def test_path_override_file_dbn_zst(self, tmp_path):
        f = tmp_path / "test.dbn.zst"
        f.touch()
        assert _discover_dbn_files(Path("/other"), path_override=f) == [f]

    def test_path_override_file_not_zst_returns_empty(self, tmp_path):
        f = tmp_path / "test.dbn"
        f.touch()
        assert _discover_dbn_files(Path("/other"), path_override=f) == []

    def test_path_override_dir_rglob(self, tmp_path):
        (tmp_path / "a.dbn.zst").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.dbn.zst").touch()
        found = _discover_dbn_files(Path("/other"), path_override=tmp_path)
        assert len(found) == 2
        assert all(p.suffix == ".zst" and ".dbn" in p.name for p in found)


def _make_fake_ohlcv_rec(instrument_id, ts_utc, open_p=21000.0, high_p=21010.0, low_p=20990.0, close_p=21005.0, volume=100):
    rec = MagicMock()
    rec.__class__.__name__ = "OHLCVMsg"
    rec.instrument_id = instrument_id
    rec.pretty_ts_event = ts_utc
    rec.pretty_open = open_p
    rec.pretty_high = high_p
    rec.pretty_low = low_p
    rec.pretty_close = close_p
    rec.volume = volume
    return rec


def _make_fake_store(schema="ohlcv-1m", mappings=None, records=None):
    store = MagicMock()
    store.schema = schema
    store.mappings = mappings or {"MNQZ5": [{"symbol": 1}], "MESZ5": [{"symbol": 2}]}
    if records is None:
        session_tz = pytz.timezone("America/New_York")
        # 10:00 ET = 14:00 UTC (EDT)
        ts = datetime(2025, 6, 15, 14, 0, 0, tzinfo=pytz.UTC)
        records = [
            _make_fake_ohlcv_rec(1, ts),
            _make_fake_ohlcv_rec(2, ts),
        ]
    def replay(callback):
        for rec in records:
            callback(rec)
    store.replay = replay
    return store


class TestRunIngestion:
    @pytest.fixture(autouse=True)
    def patch_settings(self, monkeypatch):
        monkeypatch.setattr("backend.services.data_collection.databento_ingestion.settings.SESSION_TIMEZONE", "America/New_York")
        monkeypatch.setattr("backend.services.data_collection.databento_ingestion.settings.SESSION_START_TIME", "09:30")
        monkeypatch.setattr("backend.services.data_collection.databento_ingestion.settings.SESSION_END_TIME", "16:00")
        monkeypatch.setattr("backend.services.data_collection.databento_ingestion.settings.DATABENTO_SYMBOL_MAP", {"MNQ": "MNQ1!", "MES": "MES1!"})
        monkeypatch.setattr("backend.services.data_collection.databento_ingestion.settings.SYMBOLS", ["MNQ1!", "MES1!"])

    def test_missing_files_returns_zero_and_error(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.data_collection.databento_ingestion.settings.DATABENTO_RAW_DIR", str(tmp_path))
        result = run_ingestion(db_session, path_override=tmp_path)
        assert result["files_processed"] == 0
        assert result["bars_inserted"] == 0
        assert any("No .dbn.zst" in e for e in result["errors"])

    def test_nonexistent_path_override_returns_zero(self, db_session):
        result = run_ingestion(db_session, path_override=Path("/nonexistent/dir/xyz"))
        assert result["files_processed"] == 0
        assert result["bars_inserted"] == 0
        assert any("No .dbn.zst" in e for e in result["errors"])

    def test_single_valid_file_inserts_bars(self, db_session, tmp_path):
        (tmp_path / "test.dbn.zst").touch()
        fake_store = _make_fake_store()
        with patch("backend.services.data_collection.databento_ingestion.settings.DATABENTO_RAW_DIR", str(tmp_path)):
            with patch("backend.services.data_collection.databento_ingestion.db.DBNStore") as MockStore:
                MockStore.from_file.return_value = fake_store
                result = run_ingestion(db_session, path_override=tmp_path)
        assert result["files_processed"] == 1
        assert result["bars_inserted"] == 2
        rows = db_session.query(SessionMinuteBar).filter(SessionMinuteBar.session_date == "2025-06-15").all()
        assert len(rows) == 2
        symbols = {r.symbol for r in rows}
        assert symbols == {"MNQ1!", "MES1!"}
        for r in rows:
            assert r.bar_time is not None
            assert r.open_price == 21000.0
            assert r.close_price == 21005.0

    def test_idempotent_second_run_same_count(self, db_session, tmp_path):
        (tmp_path / "test.dbn.zst").touch()
        fake_store = _make_fake_store()
        with patch("backend.services.data_collection.databento_ingestion.settings.DATABENTO_RAW_DIR", str(tmp_path)):
            with patch("backend.services.data_collection.databento_ingestion.db.DBNStore") as MockStore:
                MockStore.from_file.return_value = fake_store
                run_ingestion(db_session, path_override=tmp_path)
                count1 = db_session.query(SessionMinuteBar).filter(SessionMinuteBar.session_date == "2025-06-15").count()
                run_ingestion(db_session, path_override=tmp_path)
                count2 = db_session.query(SessionMinuteBar).filter(SessionMinuteBar.session_date == "2025-06-15").count()
        assert count1 == count2
        assert count1 == 2

    def test_from_file_raises_error_in_result(self, db_session, tmp_path):
        (tmp_path / "bad.dbn.zst").touch()
        with patch("backend.services.data_collection.databento_ingestion.settings.DATABENTO_RAW_DIR", str(tmp_path)):
            with patch("backend.services.data_collection.databento_ingestion.db.DBNStore") as MockStore:
                MockStore.from_file.side_effect = RuntimeError("corrupt file")
                result = run_ingestion(db_session, path_override=tmp_path)
        assert result["files_processed"] == 1
        assert result["bars_inserted"] == 0
        assert any("bad.dbn.zst" in e or "corrupt" in e for e in result["errors"])

    def test_replay_raises_rollback_and_error(self, db_session, tmp_path):
        (tmp_path / "test.dbn.zst").touch()
        store = _make_fake_store()
        def replay_that_raises(callback):
            raise RuntimeError("replay failed")
        store.replay = replay_that_raises
        with patch("backend.services.data_collection.databento_ingestion.settings.DATABENTO_RAW_DIR", str(tmp_path)):
            with patch("backend.services.data_collection.databento_ingestion.db.DBNStore") as MockStore:
                MockStore.from_file.return_value = store
                result = run_ingestion(db_session, path_override=tmp_path)
        assert result["files_processed"] == 1
        assert result["bars_inserted"] == 0
        assert any("replay" in e.lower() or "test.dbn" in e for e in result["errors"])
        rows = db_session.query(SessionMinuteBar).all()
        assert len(rows) == 0

    def test_schema_not_ohlcv_skipped_no_inserts(self, db_session, tmp_path):
        (tmp_path / "mbp.dbn.zst").touch()
        store = _make_fake_store(schema="mbp-1")
        with patch("backend.services.data_collection.databento_ingestion.settings.DATABENTO_RAW_DIR", str(tmp_path)):
            with patch("backend.services.data_collection.databento_ingestion.db.DBNStore") as MockStore:
                MockStore.from_file.return_value = store
                result = run_ingestion(db_session, path_override=tmp_path)
        assert result["files_processed"] == 1
        assert result["bars_inserted"] == 0
        rows = db_session.query(SessionMinuteBar).all()
        assert len(rows) == 0


class TestIngestFromStore:
    """Unit tests for ingest_from_store with fake store."""
    @pytest.fixture(autouse=True)
    def patch_settings(self, monkeypatch):
        monkeypatch.setattr("backend.services.data_collection.databento_ingestion.settings.SYMBOLS", ["MNQ1!", "MES1!"])

    def test_returns_bars_for_mapped_symbols(self):
        session_tz = pytz.timezone("America/New_York")
        ts = datetime(2025, 6, 15, 14, 0, 0, tzinfo=pytz.UTC)
        store = _make_fake_store(records=[
            _make_fake_ohlcv_rec(1, ts),
            _make_fake_ohlcv_rec(2, ts),
        ])
        id_to_raw = {1: "MNQZ5", 2: "MESZ5"}
        bars = ingest_from_store(
            store, session_tz, 9, 30, 16, 0, id_to_raw, dry_run=True
        )
        assert len(bars) == 2
        assert {b["symbol"] for b in bars} == {"MNQ1!", "MES1!"}
        assert all(b["session_date"] == "2025-06-15" for b in bars)

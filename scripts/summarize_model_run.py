"""Summarize latest model artifacts into a run folder + CSV leaderboard.

Usage:
  python scripts/summarize_model_run.py --run-id 20260309_210000
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, Optional

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config.settings import settings  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _nested(d: Dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    try:
        if cur is None:
            return None
        return float(cur)
    except Exception:
        return None


def _latest_backtest_summary(models_dir: Path) -> Optional[Path]:
    candidates = list(models_dir.glob("event_hour_backtest_*_summary.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write compact run summary + leaderboard row.")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier (default: UTC timestamp)")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=settings.MODELS_DIR,
        help="Models directory containing metrics/backtests",
    )
    args = parser.parse_args()

    models_dir = args.models_dir
    run_id = args.run_id.strip() or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = models_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    next_metrics_path = models_dir / "next_minute_metrics.json"
    cont_metrics_path = models_dir / "event_hour_continuation_metrics.json"
    rev_metrics_path = models_dir / "event_hour_reversal_metrics.json"
    backtest_summary_path = _latest_backtest_summary(models_dir)

    next_metrics = _load_json(next_metrics_path)
    cont_metrics = _load_json(cont_metrics_path)
    rev_metrics = _load_json(rev_metrics_path)
    backtest_summary = _load_json(backtest_summary_path) if backtest_summary_path else {}

    row = {
        "run_id": run_id,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "next_test_direction_5m_accuracy": _nested(next_metrics, "test", "direction_5m_accuracy"),
        "next_test_direction_5m_macro_f1": _nested(next_metrics, "test", "direction_5m_macro_f1"),
        "event_cont_test_f1": _nested(cont_metrics, "best_fold_metrics", "test", "f1")
        if cont_metrics.get("walk_forward")
        else _nested(cont_metrics, "test", "f1"),
        "event_rev_test_f1": _nested(rev_metrics, "best_fold_metrics", "test", "f1")
        if rev_metrics.get("walk_forward")
        else _nested(rev_metrics, "test", "f1"),
        "backtest_trades": backtest_summary.get("trades"),
        "backtest_hit_rate": backtest_summary.get("hit_rate"),
        "backtest_total_pnl": backtest_summary.get("total_pnl"),
        "backtest_max_drawdown": backtest_summary.get("max_drawdown"),
        "backtest_trade_rate": backtest_summary.get("trade_rate"),
        "backtest_summary_file": str(backtest_summary_path) if backtest_summary_path else "",
    }

    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w") as f:
        json.dump(row, f, indent=2)

    for p in (next_metrics_path, cont_metrics_path, rev_metrics_path, backtest_summary_path):
        if p and p.is_file():
            copy2(p, run_dir / p.name)

    leaderboard_path = models_dir / "runs" / "leaderboard.csv"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not leaderboard_path.is_file()
    with leaderboard_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved run summary: {summary_path}")
    print(f"Updated leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()

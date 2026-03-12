"""Sweep min-confidence / min-continuation-prob / min-reversal-prob and report PnL, hit-rate, trades."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import settings
from backend.services.trading.strategy import RiskConfig
from scripts.backtest_event_hour import run_backtest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep event-hour backtest thresholds and write results to CSV."
    )
    parser.add_argument(
        "--min-confidence",
        type=str,
        default="0.0,0.50,0.53,0.55",
        help="Comma-separated min-confidence values to try",
    )
    parser.add_argument(
        "--min-continuation-prob",
        type=str,
        default="0.0,0.50",
        help="Comma-separated min-continuation-prob values",
    )
    parser.add_argument(
        "--min-reversal-prob",
        type=str,
        default="0.0,0.51",
        help="Comma-separated min-reversal-prob values",
    )
    parser.add_argument(
        "--min-action-margin",
        type=str,
        default="0.0,0.02",
        help="Comma-separated min-action-margin values",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="event_hour_sweep",
        help="Backtest output prefix (sweep overwrites this each run)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Output CSV path (default: data/models/event_hour_threshold_sweep.csv)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="total_pnl",
        choices=["total_pnl", "sharpe", "hit_rate", "profit_factor"],
        help="Metric to rank by for top-10 table (total_pnl, sharpe, hit_rate, profit_factor)",
    )
    args = parser.parse_args()

    def parse_floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    conf_vals = parse_floats(args.min_confidence)
    cont_vals = parse_floats(args.min_continuation_prob)
    rev_vals = parse_floats(args.min_reversal_prob)
    margin_vals = parse_floats(args.min_action_margin)

    risk = RiskConfig()
    rows: list[dict] = []
    total = len(conf_vals) * len(cont_vals) * len(rev_vals) * len(margin_vals)
    n = 0
    for min_conf in conf_vals:
        for min_cont in cont_vals:
            for min_rev in rev_vals:
                for min_margin in margin_vals:
                    n += 1
                    print(f"Run {n}/{total}: min_conf={min_conf} min_cont={min_cont} min_rev={min_rev} margin={min_margin}")
                    try:
                        result = run_backtest(
                            prefix=args.prefix,
                            batch_size=512,
                            risk=risk,
                            min_confidence=min_conf,
                            min_action_margin=min_margin,
                            min_continuation_prob=min_cont,
                            min_reversal_prob=min_rev,
                        )
                        s = result["summary"]
                        rows.append({
                            "min_confidence": min_conf,
                            "min_continuation_prob": min_cont,
                            "min_reversal_prob": min_rev,
                            "min_action_margin": min_margin,
                            "trades": s["trades"],
                            "hit_rate": round(s["hit_rate"], 4),
                            "total_pnl": round(s["total_pnl"], 2),
                            "max_drawdown": round(s["max_drawdown"], 2),
                            "ending_capital": round(s["ending_capital"], 2),
                            "sharpe": round(s.get("sharpe", 0), 4),
                            "sortino": round(s.get("sortino", 0), 4),
                            "calmar": round(s.get("calmar", 0), 4),
                            "profit_factor": round(s.get("profit_factor", 0), 4),
                            "bootstrap_pvalue": round(s.get("bootstrap_pnl_pvalue", 0.5), 4),
                        })
                    except Exception as e:
                        print(f"  Error: {e}")
                        rows.append({
                            "min_confidence": min_conf,
                            "min_continuation_prob": min_cont,
                            "min_reversal_prob": min_rev,
                            "min_action_margin": min_margin,
                            "trades": -1,
                            "hit_rate": -1,
                            "total_pnl": float("nan"),
                            "max_drawdown": float("nan"),
                            "ending_capital": float("nan"),
                            "sharpe": float("nan"),
                            "sortino": float("nan"),
                            "calmar": float("nan"),
                            "profit_factor": float("nan"),
                            "bootstrap_pvalue": float("nan"),
                        })

    if not rows:
        print("No results.")
        return 1

    out_path = Path(args.out_csv) if args.out_csv else settings.MODELS_DIR / "event_hour_threshold_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")

    # Print table: sort by selected metric (higher is better for all)
    valid = [r for r in rows if r["trades"] >= 0]
    metric = args.metric
    by_metric = sorted(valid, key=lambda r: r.get(metric, 0) if not (isinstance(r.get(metric), float) and (r.get(metric) != r.get(metric))) else -1e9, reverse=True)
    print(f"\nTop 10 by {metric}:")
    print(f"{'min_conf':<8} {'min_cont':<8} {'min_rev':<8} {'margin':<8} {'trades':<8} {'hit_rate':<8} {'total_pnl':<12} {'sharpe':<8} {'pf':<8}")
    print("-" * 88)
    for r in by_metric[:10]:
        sharpe = r.get("sharpe", 0)
        pf = r.get("profit_factor", 0)
        print(f"{r['min_confidence']:<8} {r['min_continuation_prob']:<8} {r['min_reversal_prob']:<8} {r['min_action_margin']:<8} {r['trades']:<8} {r['hit_rate']:<8} {r['total_pnl']:<12} {sharpe:<8.4f} {pf:<8.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

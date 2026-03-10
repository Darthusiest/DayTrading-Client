# Event-hour model backtest and thresholds

## Output files (on server: `data/models/`)

After running the backtest you get:

- **`event_hour_backtest_summary.json`** – one-shot stats and filter reasons
- **`event_hour_backtest_trades.json`** – per-trade list
- **`event_hour_backtest_curves.png`** – equity, drawdown, by-action and by-event-group
- **`event_hour_backtest_chart_data.json`** – equity curve, drawdown, trade PnL, by_event_group for charts

### Inspecting the summary

```bash
# From repo root
cat data/models/event_hour_backtest_summary.json | python -m json.tool
```

Useful fields:

| Field | Meaning |
|-------|--------|
| `trades` | Number of trades taken |
| `hit_rate` | Fraction of trades with positive PnL |
| `total_pnl` | Sum of net PnL over all trades |
| `max_drawdown` | Max peak-to-trough drawdown |
| `signals_filtered` | How many events were filtered out (no trade) |
| `filter_reason_counts` | Counts per filter: `below_confidence`, `below_model_threshold`, `below_margin`, etc. |
| `by_event_group` | Per group (PDH_PDL, ORB, ATR, IMP, BOS): `trades`, `hit_rate`, `total_pnl` |
| `min_confidence`, `min_continuation_prob`, `min_reversal_prob` | Thresholds used for this run |

### Inspecting the trades

```bash
# First few trades (pretty-printed)
head -n 100 data/models/event_hour_backtest_trades.json

# Count trades
python -c "import json; d=json.load(open('data/models/event_hour_backtest_trades.json')); print(len(d))"
```

Each trade has: `entry_time`, `exit_time`, `direction`, `confidence`, `net_pnl`, `return_pct`, etc.

---

## Trying different thresholds

Single run with custom thresholds (from repo root):

```bash
python -m scripts.backtest_event_hour \
  --min-confidence 0.55 \
  --min-continuation-prob 0.52 \
  --min-reversal-prob 0.53 \
  --min-action-margin 0.02
```

Go-forward defaults (higher bars):

```bash
python -m scripts.run_event_backtest_go_forward
# uses: min-confidence=0.53, min-action-margin=0.02, min-continuation-prob=0.50, min-reversal-prob=0.51
```

---

## Threshold sweep (grid search)

Sweep several values and write a CSV of results (trades, hit_rate, total_pnl, etc.):

```bash
python -m scripts.sweep_event_hour_thresholds
```

Defaults try:

- `--min-confidence`: 0.0, 0.50, 0.53, 0.55  
- `--min-continuation-prob`: 0.0, 0.50  
- `--min-reversal-prob`: 0.0, 0.51  
- `--min-action-margin`: 0.0, 0.02  

Override with comma-separated lists:

```bash
python -m scripts.sweep_event_hour_thresholds \
  --min-confidence 0.50,0.53,0.55 \
  --min-continuation-prob 0.50 \
  --min-reversal-prob 0.51 \
  --out-csv data/models/my_sweep.csv
```

Output: `data/models/event_hour_threshold_sweep.csv` (or `--out-csv`) and a “Top 10 by total_pnl” table in the terminal.

---

## Dependencies

- **Optuna** is in `requirements.txt` (`optuna>=3.0.0`). If a fresh env is missing it: `pip install optuna`.
- Run all scripts from the **project root** using `python -m scripts.<script_name>` so `backend` and `scripts` resolve.

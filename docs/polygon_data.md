# Polygon.io Data and Free Tier

## Is it free? What’s the catch?

Polygon has a **free Basic tier** with real limits:

| Free tier | Paid (e.g. Starter $29/mo) |
|-----------|----------------------------|
| **5 API calls per minute** | Unlimited calls |
| **End-of-day data only** (no intraday for free) | 15-min delayed or real-time |
| **Personal / non-commercial use only** | Commercial use allowed per plan |
| 2 years historical, 100% market coverage | Same or better |

So for **free** you get: very low rate limit, and only **daily** bars (no minute bars). For minute bars during the 6:30–8:00 session you need a **paid** plan (Starter or above). Their terms also require “Non-Professional” status for the free tier.

**Bottom line:** Free is fine for testing and daily data; for this app’s **intraday** (minute) data and session bars you need a paid Polygon subscription.

---

## Example: data this app gets from Polygon

The backend maps MNQ/MES to Polygon symbols (e.g. `C:MNQ1`, `C:MES1`) and calls the **aggregates** API. One bar looks like this:

### Single bar (e.g. `get_price_data` or one bar from `get_historical_data`)

```json
{
  "symbol": "MNQ1!",
  "timestamp": "2026-02-21T06:31:00-08:00",
  "open": 25074.5,
  "high": 25076.0,
  "low": 25073.25,
  "close": 25075.0,
  "volume": 1247,
  "timeframe": "1",
  "vwap": 25074.8
}
```

- **open, high, low, close**: prices for that bar (minute or day).
- **volume**: number of contracts (or shares).
- **vwap**: volume-weighted average price (when the API provides it).

### Minute bars for 6:30–8:00 (e.g. `get_historical_data` → SessionMinuteBar)

Same shape, one dict per minute:

```json
[
  {
    "symbol": "MNQ1!",
    "timestamp": "2026-02-21T06:30:00-08:00",
    "open": 25072.0,
    "high": 25074.0,
    "low": 25071.5,
    "close": 25073.5,
    "volume": 892,
    "vwap": 25072.9
  },
  {
    "symbol": "MNQ1!",
    "timestamp": "2026-02-21T06:31:00-08:00",
    "open": 25073.5,
    "high": 25076.0,
    "low": 25073.0,
    "close": 25075.0,
    "volume": 1103,
    "vwap": 25074.5
  }
]
```

That’s the data used for labels (e.g. session high/low/close) and for attaching price to snapshots.

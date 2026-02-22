# Database Migrations

For existing databases, run the following SQL when the schema changes.

## Add interval_minutes and bar_time to snapshots (multi-timeframe session capture)

When upgrading to support per-candle capture (1m, 5m, 15m, 1h) between 6:30–8:00:

```sql
ALTER TABLE snapshots ADD COLUMN interval_minutes INTEGER NULL;
ALTER TABLE snapshots ADD COLUMN bar_time TIMESTAMP NULL;
CREATE INDEX ix_snapshots_interval_minutes ON snapshots (interval_minutes);
CREATE INDEX ix_snapshots_bar_time ON snapshots (bar_time);
```

Existing rows (before/after/manual) keep `interval_minutes` and `bar_time` as NULL.

## Add interval_minutes to training_samples (session-candle training)

When upgrading to support training from session candle pairs (per-interval before/after):

```sql
ALTER TABLE training_samples ADD COLUMN interval_minutes INTEGER NULL;
CREATE INDEX ix_training_samples_interval_minutes ON training_samples (interval_minutes);
```

## Make `user_expected_price` nullable (predictions table)

When upgrading to support optional expected price (screenshot-only price estimate):

```sql
ALTER TABLE predictions ALTER COLUMN user_expected_price DROP NOT NULL;
```

This allows `POST /api/v1/predict` to be called with only screenshot + symbol (no expected price) and still save a prediction with `model_predicted_price` as the estimate.

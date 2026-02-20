# Database Migrations

For existing databases, run the following SQL when the schema changes.

## Make `user_expected_price` nullable (predictions table)

When upgrading to support optional expected price (screenshot-only price estimate):

```sql
ALTER TABLE predictions ALTER COLUMN user_expected_price DROP NOT NULL;
```

This allows `POST /api/v1/predict` to be called with only screenshot + symbol (no expected price) and still save a prediction with `model_predicted_price` as the estimate.

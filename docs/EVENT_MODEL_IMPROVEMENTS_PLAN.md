# Event Model Improvements: Implementation Plan

This plan details how to implement the data/labeling and training/tuning improvements for the event-hour continuation and reversal models. Each section maps to the current codebase and includes specific file changes, new settings, and suggested order of execution.

---

## Part 1: Data & Labeling Improvements

### 1.1 Tighter Label Definitions

**Current state** (in `scripts/train_event_hour_models.py`):
- Labels use `ret_60 = (closes[i + horizon_minutes] - closes[i]) / closes[i]` with no exclusion of the event bar itself.
- Band: `band = max(label_min_band, label_band_k * atr_pct_i)`; label is continuation/reversal only if `|ret_60| > band`.
- No separation between “event spike” (first few minutes) and “clean post-event” move.
- No magnitude threshold tied to economic significance (e.g., spread).

**Planned changes**:

| Task | Location | Description |
|------|----------|-------------|
| **Exclude event spike from label window** | `_build_event_dataset()` | Add config `EVENT_LABEL_SKIP_FIRST_MINUTES` (default 5). Use `ret_60 = (close[i + skip + horizon_minutes] - close[i + skip]) / close[i + skip]` instead of starting at bar `i`. This avoids counting the initial spike in the label. |
| **Minimum gap from previous event** | `_build_event_dataset()` | Add `EVENT_MIN_GAP_FROM_PREV_EVENT` (default 15). When iterating events, skip bar `i` if another event of the same type fired within the last N minutes. Prevents overlapping labels. |
| **Direction + magnitude labels** | `_build_event_dataset()` | Add `EVENT_LABEL_MIN_MOVE_ATR_K` (e.g., 0.5). Only label as continuation/reversal if `|ret_60| > max(band, k * ATR_i / close_i)`. Ensures the move is economically meaningful relative to volatility. Optionally add `EVENT_LABEL_MIN_MOVE_SPREAD_K` if spread data is available (proxy: ATR * 0.02). |
| **Horizon as a parameter** | `_build_event_dataset()` | Add `EVENT_LABEL_HORIZON_MINUTES` (default 60). Allow 45, 60, 90 as discrete options for ablation. Cache key must include horizon. |
| **EVENT_FEATURE_VERSION bump** | Top of script | Increment to 4 when label logic changes, so caches invalidate. |

**Files to edit**:
- `scripts/train_event_hour_models.py` (lines ~764–900)
- `backend/config/settings.py` (new `EVENT_*` settings)

**Validation**:
- Compare sample counts before/after with `EVENT_REBUILD_CACHE=True`.
- Log distribution of `|ret_60|` and label balance per fold.

---

### 1.2 Regime / Session Filtering

**Current state**:
- `bar_features.py` already has `_time_features()` (hours, minutes, day_of_week, minutes_since_open, is_ny_open, is_power_hour, session_phase) and `_regime_and_microstructure()` (vol_regime, trend_strength, etc.).
- `build_session_feature_matrix()` returns a matrix that includes these (see `bar_features.py`).
- Event model uses `all_feat` from `build_session_feature_matrix()` but does not explicitly add regime buckets or event-type/importance features.

**Planned changes**:

| Task | Location | Description |
|------|----------|-------------|
| **Add regime features to event input** | `_build_event_dataset()` + `bar_features.py` | Ensure `vol_regime`, `session_phase`, `is_ny_open`, `is_power_hour` are in the feature matrix passed to the LSTM. If not already present, concatenate them in `all_feat_aug`. |
| **Rolling 30/60m realized vol buckets** | `bar_features.py` or inline | Add `realized_vol_30m`, `realized_vol_60m` (std of 1m returns over window). Bucket into tertiles (0/0.5/1) and append to features. |
| **Event-type / importance features** | `_build_event_dataset()` | Extend `event_feat_row` with: (1) one-hot or embedding index for event type (PDH/PDL, ORB, BOS, ATR, IMP), (2) optional importance: ORB/BOS/PDH-PDL = “high”, ATR/IMP = “medium”. Add settings `EVENT_ENABLE_REGIME_FEATURES`, `EVENT_ENABLE_EVENT_TYPE_FEATURES`. |
| **Train separate models by regime (optional)** | New script or mode | Add `EVENT_TRAIN_BY_REGIME` (default False). If True, build three datasets: low-vol, mid-vol, high-vol (by `vol_regime` tertile) and train three continuation + three reversal models. Inference picks model by current regime. More complex; do after single-model regime features are validated. |

**Files to edit**:
- `backend/services/ml/bar_features.py` (ensure regime features in output)
- `scripts/train_event_hour_models.py` (concat regime feats, extend event_feat_row)
- `backend/config/settings.py`

**Validation**:
- Check feature dimension before/after and that no NaNs.
- Compare validation F1 by regime bucket (low/mid/high vol).

---

### 1.3 Better Negative Examples

**Current state**:
- Only event times produce samples. There are no explicit “non-event” or “near-miss” samples.
- `sign == 0` (i.e., `|ret_60| <= band`) yields `cont=0`, `rev=0`; these are still included as negatives for both heads.
- Dataset can be heavily imbalanced (many continuation negatives, fewer reversal positives).

**Planned changes**:

| Task | Location | Description |
|------|----------|-------------|
| **Near-miss non-events** | `_build_event_dataset()` | Add `EVENT_ADD_NEAR_MISS_SAMPLES` (default False). For non-event bars `i` where: (1) price/vol profile is similar to event bars (e.g., `range_over_atr[i]` in top 30%, `vol_z[i]` in top 30%), (2) no event fired at `i`, (3) `|ret_60| <= band` → add as explicit negative (cont=0, rev=0) with same feature structure. Cap at `EVENT_NEAR_MISS_MAX_PER_SESSION` (e.g., 20) to avoid blowing up dataset. |
| **Down-sample easy negatives** | `_build_event_dataset()` or Dataset | Add `EVENT_DOWNSAMPLE_EASY_NEG_RATIO` (default 1.0 = no downsampling). For samples with cont=0 and rev=0 where `|ret_60|` is very small (e.g., < 0.5 * band), randomly drop with probability `1 - EVENT_DOWNSAMPLE_EASY_NEG_RATIO`. This focuses training on harder negatives. |
| **Balance continuation vs reversal** | `_train_one()` | Already have `pos_weight` and focal loss. Optionally add `EVENT_CONT_REV_SAMPLE_WEIGHTS` to explicitly oversample the rarer head (e.g., reversal) via a weighted sampler if needed. |

**Files to edit**:
- `scripts/train_event_hour_models.py`
- `backend/config/settings.py`

**Validation**:
- Compare label distribution before/after.
- Check that near-miss samples improve validation performance on “no-trade” decisions (precision of low-prob predictions).

---

## Part 2: Training & Tuning Process

### 2.1 Cost-Aware Objective

**Current state**:
- Loss: `_BinaryFocalLoss` or `BCEWithLogitsLoss` with `pos_weight`.
- Early stopping and fold selection use validation F1 (`EVENT_EARLY_STOP_METRIC`).
- No explicit PnL or cost weighting in the loss.

**Planned changes**:

| Task | Location | Description |
|------|----------|-------------|
| **Weighted loss by magnitude** | `_train_one()` | Add `EVENT_LOSS_WEIGHT_BY_MAGNITUDE` (default False). When True, weight each sample’s loss by `1 + |ret_60| / atr_pct` so that wrong predictions on large moves are penalized more. Requires passing `forward_return_60m` and `atr` (or a proxy) into the training loop; may need to extend `EventHourDataset` or use a custom sampler. |
| **Validation PnL proxy metric** | `_train_one()` + new helper | Add `EVENT_VAL_PNL_PROXY` (default False). On validation, compute a proxy: for each sample with pred >= threshold, add `sign(pred) * sign(actual) * |ret_60|` (simplified PnL). Use this as an alternative early-stop metric via `EVENT_EARLY_STOP_METRIC=val_pnl_proxy`. |
| **Custom metric for fold selection** | Walk-forward logic | Extend `EVENT_EARLY_STOP_METRIC` to support `val_pnl_proxy` or `val_cost_weighted_f1`. Select best fold by this metric instead of raw F1 when enabled. |

**Files to edit**:
- `scripts/train_event_hour_models.py` (`_train_one`, `_evaluate`, fold selection)
- `backend/services/ml/event_hour.py` (optional: extend Dataset to expose `forward_return_60m` for weighted loss)
- `backend/config/settings.py`

**Validation**:
- Compare fold selection when using `f1` vs `val_pnl_proxy`; check if backtest PnL improves.

---

### 2.2 Hyperparameter Search

**Current state**:
- Single config from `settings` (EVENT_LR, EVENT_EPOCHS, EVENT_LSTM_HIDDEN_SIZE, etc.).
- No automated search over hyperparameters.

**Planned changes**:

| Task | Location | Description |
|------|----------|-------------|
| **Optuna integration script** | New: `scripts/tune_event_hour_models.py` | Create a script that: (1) loads bars and builds dataset (or uses cached), (2) runs Optuna study over: `lr` (log, 1e-4–1e-2), `hidden_size` (64, 128, 256), `num_layers` (2, 3), `dropout` (0.05, 0.1, 0.2), `weight_decay` (1e-6, 1e-5, 1e-4), `focal_gamma` (1.0, 2.0, 3.0), `pos_weight_scale` (1.0, 1.5, 2.0), `label_horizon` (45, 60, 90). Each trial trains one fold (e.g., last WF fold) and returns validation metric. Uses `EVENT_EARLY_STOP_METRIC` as objective. |
| **Constraint: one trial per walk-forward window** | `tune_event_hour_models.py` | To avoid look-ahead, each trial uses a fixed train/val split (e.g., last fold). No test data in tuning. |
| **Save best params to .env.example or config** | Post-tune | Log best params to a `event_hour_tuned_config.json` for manual copy into .env. |

**Files to create**:
- `scripts/tune_event_hour_models.py`

**Dependencies**:
- Add `optuna` to `requirements.txt` if not present.

**Validation**:
- Run a small study (e.g., 20 trials) and confirm improvement over default.

---

### 2.3 Regularization & Robustness

**Current state**:
- `EVENT_WEIGHT_DECAY=1e-5`, `EVENT_DROPOUT=0.1`, `EVENT_GRAD_CLIP_NORM=1.0`.
- Early stopping on validation metric with `EVENT_EARLY_STOP_PATIENCE=8`.

**Planned changes**:

| Task | Location | Description |
|------|----------|-------------|
| **Stronger L2** | `backend/config/settings.py` | Add `EVENT_WEIGHT_DECAY` range in tuning; consider default 1e-4 for overfit-prone runs. |
| **Label smoothing (optional)** | `_train_one()` | Add `EVENT_LABEL_SMOOTHING` (default 0.0). Replace hard 0/1 with `y_smooth = y * (1 - eps) + 0.5 * eps` for binary targets. |
| **Stricter early stopping** | `_train_one()` | Add `EVENT_EARLY_STOP_MIN_DELTA` (default 0.0). Require improvement > delta to reset patience. |
| **Gradient clipping** | Already present | Ensure `EVENT_GRAD_CLIP_NORM` is applied in training step (verify in `_train_one`). |

**Files to edit**:
- `scripts/train_event_hour_models.py`
- `backend/config/settings.py`

**Validation**:
- Compare train vs val loss curves; reduce overfitting if visible.

---

### 2.4 Calibration

**Current state**:
- Temperature scaling is already implemented (`_calibrate_temperature()` in `train_event_hour_models.py`).
- Calibration JSON is saved per model (`event_hour_continuation_calibration.json`, `event_hour_reversal_calibration.json`).
- Backtest and API use `sigmoid(logits / temperature)`.

**Planned changes**:

| Task | Location | Description |
|------|----------|-------------|
| **Refined temperature grid** | `_calibrate_temperature()` | Extend grid to `[0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]` for finer search. Optionally use scipy.optimize or Optuna for continuous T. |
| **Platt scaling (optional)** | New helper | Add `EVENT_CALIBRATION_METHOD`: `temperature` (default) or `platt`. Platt: fit `P(y=1) = 1/(1+exp(A*logit+B))` on validation. Store A, B in calibration JSON. |
| **Brier score logging** | `_train_one()` or eval | Add Brier score to metrics JSON: `brier = mean((p - y)^2)`. Log per fold for monitoring. |
| **Reliability curves** | New: `scripts/plot_event_calibration.py` | Script to plot reliability (expected vs observed) for continuation and reversal on validation set. Save to `data/models/event_hour_*_reliability.png`. |

**Files to edit**:
- `scripts/train_event_hour_models.py` (`_calibrate_temperature`, metrics)
- `backend/config/settings.py`
- New: `scripts/plot_event_calibration.py`

**Validation**:
- Compare Brier before/after calibration; inspect reliability plots.

---

## Suggested Order of Implementation

1. **Phase 1 – Low risk, high impact**
   - 1.1: Exclude event spike (`EVENT_LABEL_SKIP_FIRST_MINUTES`)
   - 1.1: Min gap from previous event
   - 2.3: Verify gradient clipping, add `EVENT_EARLY_STOP_MIN_DELTA`
   - 2.4: Brier score logging, refined temperature grid

2. **Phase 2 – Feature & regime**
   - 1.2: Add regime features (vol_regime, session_phase) to event input
   - 1.2: Realized vol buckets (30m, 60m)
   - 1.2: Event-type features in event_feat_row

3. **Phase 3 – Labels & negatives**
   - 1.1: Direction + magnitude labels (`EVENT_LABEL_MIN_MOVE_ATR_K`)
   - 1.3: Near-miss non-events (optional)
   - 1.3: Down-sample easy negatives (optional)

4. **Phase 4 – Training pipeline**
   - 2.1: Cost-aware weighted loss
   - 2.1: Validation PnL proxy metric
   - 2.2: Optuna tuning script

5. **Phase 5 – Advanced**
   - 1.2: Train-by-regime (optional)
   - 2.4: Platt scaling, reliability curve script

---

## New Settings Summary

Add to `backend/config/settings.py`:

```python
# Label refinement
EVENT_LABEL_SKIP_FIRST_MINUTES: int = 5
EVENT_MIN_GAP_FROM_PREV_EVENT: int = 15
EVENT_LABEL_MIN_MOVE_ATR_K: float = 0.5
EVENT_LABEL_HORIZON_MINUTES: int = 60

# Regime / session
EVENT_ENABLE_REGIME_FEATURES: bool = True
EVENT_ENABLE_EVENT_TYPE_FEATURES: bool = True
EVENT_TRAIN_BY_REGIME: bool = False

# Negative examples
EVENT_ADD_NEAR_MISS_SAMPLES: bool = False
EVENT_NEAR_MISS_MAX_PER_SESSION: int = 20
EVENT_DOWNSAMPLE_EASY_NEG_RATIO: float = 1.0

# Cost-aware
EVENT_LOSS_WEIGHT_BY_MAGNITUDE: bool = False
EVENT_VAL_PNL_PROXY: bool = False

# Regularization
EVENT_EARLY_STOP_MIN_DELTA: float = 0.0
EVENT_LABEL_SMOOTHING: float = 0.0

# Calibration
EVENT_CALIBRATION_METHOD: str = "temperature"
```

---

## Cache Invalidation

When changing label logic, horizon, or feature set:
- Bump `EVENT_FEATURE_VERSION` in `train_event_hour_models.py`
- Add cache meta checks in `_check_event_dataset_cache()` for new settings
- Set `EVENT_REBUILD_CACHE=True` for the first run after changes

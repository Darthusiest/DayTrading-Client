# Day Trading AI Agent Backend

A backend system for a day trading AI agent that learns from US regular trading hours (RTH, e.g. 9:30 AM – 4:00 PM Eastern) price chart screenshots and minute bar data for Nasdaq and S&P 500 futures. The agent predicts price movements, evaluates learning performance, and provides probability assessments.

## Features

- **Data Collection**: Polygon.io integration for market data and TradingView for chart screenshots
- **ML Model**: Hybrid CNN + Time Series model for price prediction from chart images
- **Learning Evaluation**: Comprehensive metrics tracking and learning progress monitoring
- **REST API**: FastAPI-based API for predictions, training, and notes management
- **Database**: PostgreSQL for storing snapshots, training data, predictions, and metrics

## Architecture

```
backend/
├── api/              # FastAPI routes and main application
├── services/         # Core business logic
│   ├── data_collection/    # TradingView integration, screenshot capture
│   ├── data_processing/    # Image preprocessing, feature extraction, labeling
│   ├── ml/                 # ML models, training, inference
│   └── evaluation/         # Metrics and learning tracking
├── database/         # Database models and connection
└── config/           # Configuration settings
```

## Setup

### Prerequisites

- Python 3.9+
- PostgreSQL database
- Chrome/Chromium (for screenshot capture)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DayTrade
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```env
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/daytrade
TRADINGVIEW_USERNAME=your_username
TRADINGVIEW_PASSWORD=your_password
DEBUG=True
```

5. Initialize the database:
```bash
python -c "from backend.database.db import init_db; init_db()"
```

## Usage

### Running the API Server

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Predictions
- `POST /api/v1/predict` - Submit a screenshot (ideally at or before market open, 9:30 ET) and optionally an expected price. Returns the model's **estimated price the market will hit** by market close (e.g. 4:00 PM ET), plus probability and learning metrics. Omit `expected_price` to get only the estimate.
- `GET /api/v1/predict/history` - Get prediction history

#### Training
- `POST /api/v1/train` - Trigger model training
- `GET /api/v1/train/status` - Get training status

#### Evaluation
- `GET /api/v1/evaluation/learning-status` - Get learning performance metrics
- `GET /api/v1/evaluation/learning-curve` - Get learning curve data
- `GET /api/v1/evaluation/best-model` - Get best model information

#### Data collection (scheduled and manual)
- **Scheduled**: When the API server runs, data collection is scheduled at **session open** (e.g. 9:30 AM ET) and **session close** (e.g. 4:00 PM ET). Configure `SESSION_START_TIME`, `SESSION_END_TIME`, `SESSION_TIMEZONE` in `.env`. Set `ENABLE_SCHEDULED_COLLECTION=false` to disable.
- **Session candle capture**: Optional long-running job from first bar after session start to session end (e.g. 9:31–16:00 ET) at 1m, 5m, 15m, 1h. Disabled by default (`ENABLE_SESSION_CANDLE_CAPTURE=false`). Run manually: `POST /api/v1/collection/run-session-candles` (optional `?session_date=YYYY-MM-DD`).
- `POST /api/v1/collection/run` - Run collection once now (optional `capture_screenshots=false` to fetch only Polygon price data).
- `POST /api/v1/collection/capture-now` - Log in to TradingView, take a screenshot of the current MNQ (or other symbol) chart, save to disk and database. Optional query: `?symbol=MES1!&interval=15`.
- `POST /api/v1/collection/process-training-data` - Build training samples from before/after pairs and from **session_candle** pairs (per-interval first bar vs session end). Session candle labels use `SessionMinuteBar` (populated after session close collection).
- `GET /api/v1/collection/schedule` - Scheduler status and next run times.

#### Live data (Polygon WebSocket)
- When enabled, a background WebSocket stream receives real-time minute bars for MNQ and MES. Set `ENABLE_POLYGON_WEBSOCKET=false` in `.env` to disable.
- `GET /api/v1/live/prices` - Latest minute-aggregate bar per symbol (OHLCV + timestamps) from the stream.

#### Notes
- `POST /api/v1/notes` - Create a note
- `GET /api/v1/notes` - Get all notes
- `GET /api/v1/notes/{note_id}` - Get specific note
- `PUT /api/v1/notes/{note_id}` - Update note
- `DELETE /api/v1/notes/{note_id}` - Delete note

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Screenshot to Price Estimate

Upload a screenshot of the market **at or before session open (e.g. 9:30 AM ET)** to get the model's estimate of the price the market will hit by **market close (e.g. 4:00 PM ET)**. Call `POST /api/v1/predict` with `file` and `symbol`; `expected_price` is optional. If you omit it, the response gives `model_predicted_price` (the estimate) and the model's confidence. If you provide an expected price, you also get the probability that level is hit.

## Data Collection

The system captures chart screenshots in two ways:

- **Before/after snapshots**: Single capture at **session open** (e.g. 9:30 AM ET) and **session close** (e.g. 4:00 PM ET) per symbol. Minute bars for the full session (9:30–16:00) are fetched and stored in `session_minute_bars` after the close snapshot.
- **Session candle capture (optional)**: From first bar after session start to session end (e.g. 9:31–16:00 ET), at each candle-close time, screenshots can be taken at 1m, 5m, 15m, 1h. Disabled by default (`ENABLE_SESSION_CANDLE_CAPTURE=false`). Files are named like `MNQ1!_session_2026-02-21_1m_0931.png`.

### Polygon.io Integration

The Polygon.io client (`backend/services/data_collection/tradingview_client.py`) provides:
- Real-time and historical price data fetching (OHLCV)
- Market status checking
- Session date management
- Support for futures contracts (Nasdaq E-mini, S&P 500 E-mini)

**Note**: Requires a Polygon.io API key. Set `POLYGON_API_KEY` in your `.env` file. The free tier is limited (5 calls/min, end-of-day data only; intraday/minute data requires a paid plan). See [docs/polygon_data.md](docs/polygon_data.md) for limits and example response data. If Polygon doesn’t offer futures API access for your plan, see [docs/futures_data_sources.md](docs/futures_data_sources.md) for alternative futures data providers and the data contract needed to plug one in.

### Screenshot Capture

The screenshot capture service (`backend/services/data_collection/screenshot_capture.py`) uses Selenium to:
- Log in to TradingView when `TRADINGVIEW_USERNAME` and `TRADINGVIEW_PASSWORD` are set
- Navigate to TradingView charts
- Capture screenshots at specified times
- Save images to the data directory

**Where screenshots are saved:**
- **Database**: Table **`snapshots`**. Each capture creates one row with: `id`, `symbol`, `snapshot_type` (`"before"`, `"after"`, `"manual"`, or `"session_candle"`), `timestamp`, `image_path`, `session_date`, `created_at`. For session_candle captures, `interval_minutes` (1, 5, 15, 60) and `bar_time` (candle-close time) are set. Optional current price from Polygon is stored in **`price_data`** linked by `snapshot_id`.
- **Filesystem**: Under **`data/raw/`**. Manual/before/after: `MNQ1!_manual_2025-02-19_143022.png`. Session candles: `MNQ1!_session_2026-02-21_1m_0631.png` (symbol, session, interval, bar time).

## ML Model

### Architecture

The `PricePredictor` model uses:
1. **CNN Encoder**: ResNet18/EfficientNet for visual feature extraction from chart screenshots
2. **MLP/LSTM**: For processing combined features and temporal patterns
3. **Regression Head**: Predicts price
4. **Classification Head**: Estimates probability of hitting expected price

### Training

To train the model:
1. Collect training data (before/after snapshots with price data)
2. Process and label the data
3. Trigger training via API: `POST /api/v1/train`

The training pipeline:
- Builds training samples from **before/after** snapshot pairs and from **session_candle** pairs (per-interval: first bar of session vs 8:00 bar). Session candle labels use **SessionMinuteBar** (minute bars 6:30–8:00, populated after the 8:00 collection).
- Preprocesses images, extracts features (including `interval_minutes` and `bar_time` for session samples)
- Trains with validation split
- Saves checkpoints and metrics

## Learning Evaluation

The system tracks:
- **Prediction Accuracy**: MAE, RMSE, direction accuracy
- **Calibration**: How well probability estimates match reality
- **Learning Progress**: Improvement over epochs
- **Pattern Recognition**: Ability to identify chart patterns

## Database Schema

- `snapshots`: Raw screenshot data. Columns include `snapshot_type`, `interval_minutes`, `bar_time` (for session_candle).
- `session_minute_bars`: Minute OHLCV bars for 6:30–8:00 session (used for session_candle training labels).
- `price_data`: OHLCV market data (linked to snapshots).
- `training_samples`: Processed training data with labels; `interval_minutes` set for session_candle-derived samples.
- `model_checkpoints`: Saved model versions
- `predictions`: User predictions and model outputs
- `learning_metrics`: Training/validation metrics
- `user_notes`: User notes and annotations

## Configuration

Key settings in `backend/config/settings.py`:
- `SYMBOLS`: Trading symbols (default: ["MNQ1!", "MES1!"] — MNQ & MES micro futures only)
- `BEFORE_SNAPSHOT_TIME`: Before snapshot time (default: "06:30")
- `AFTER_SNAPSHOT_TIME`: After snapshot time (default: "08:00")
- `BATCH_SIZE`: Training batch size (default: 64; set in `.env` if needed)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `NUM_EPOCHS`: Training epochs (default: 200)
- `NUM_LSTM_LAYERS`, `LSTM_HIDDEN_SIZE`: LSTM depth and width (defaults: 2, 128)
- `MLP_HIDDENS`: Comma-separated MLP hidden sizes (default: "256,128"); e.g. "256,256,128" for a deeper MLP
- `CNN_TRAINABLE_PARAM_GROUPS`: Number of CNN parameter tensors to train (default: 10); 0 = unfreeze all. See `.env.example`

## Development

### Project Structure

```
DayTrade/
├── backend/          # Backend application
├── data/             # Data storage
│   ├── raw/          # Raw screenshots
│   ├── processed/    # Processed images
│   └── models/       # Saved model checkpoints
├── tests/            # Test files
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

### Testing

```bash
pytest tests/
```

## Next Steps

1. **Model Improvements**: Experiment with different architectures and hyperparameters (e.g. multi-interval or sequence inputs from session candles)
2. **Frontend**: Build frontend interface for chat, screenshot upload, and notes
3. **Real-time Updates**: Add WebSocket support for real-time predictions
4. **Indicators**: Document or standardize chart indicators used in TradingView for training

## License

[Your License Here]

## Contributing

[Contributing Guidelines]

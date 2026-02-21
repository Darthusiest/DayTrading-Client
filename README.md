# Day Trading AI Agent Backend

A backend system for a day trading AI agent that learns from NY AM (6:30 AM - 8:00 AM PST) price chart screenshots of Nasdaq and S&P 500 futures. The agent predicts price movements, evaluates learning performance, and provides probability assessments.

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
- `POST /api/v1/predict` - Submit a screenshot (ideally before/at 6:30 AM PST) and optionally an expected price. Returns the model's **estimated price the market will hit** by 8:00 AM PST, plus probability and learning metrics. Omit `expected_price` to get only the estimate.
- `GET /api/v1/predict/history` - Get prediction history

#### Training
- `POST /api/v1/train` - Trigger model training
- `GET /api/v1/train/status` - Get training status

#### Evaluation
- `GET /api/v1/evaluation/learning-status` - Get learning performance metrics
- `GET /api/v1/evaluation/learning-curve` - Get learning curve data
- `GET /api/v1/evaluation/best-model` - Get best model information

#### Data collection (scheduled and manual)
- **Scheduled**: When the API server runs, data collection is scheduled for **6:30 AM** and **8:00 AM** (PST). Set `ENABLE_SCHEDULED_COLLECTION=false` in `.env` to disable.
- `POST /api/v1/collection/run` - Run collection once now (optional `capture_screenshots=false` to fetch only Polygon price data).
- `GET /api/v1/collection/schedule` - Scheduler status and next run times.

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

Upload a screenshot of the market **before or at 6:30 AM PST** to get the model's estimate of the price the market will hit by 8:00 AM PST. Call `POST /api/v1/predict` with `file` and `symbol`; `expected_price` is optional. If you omit it, the response gives `model_predicted_price` (the estimate) and the model's confidence. If you provide an expected price, you also get the probability that level is hit.

## Data Collection

The system is designed to capture chart screenshots at:
- **Before snapshot**: 6:30 AM PST
- **After snapshot**: 8:00 AM PST

### Polygon.io Integration

The Polygon.io client (`backend/services/data_collection/tradingview_client.py`) provides:
- Real-time and historical price data fetching (OHLCV)
- Market status checking
- Session date management
- Support for futures contracts (Nasdaq E-mini, S&P 500 E-mini)

**Note**: Requires a Polygon.io API key. Set `POLYGON_API_KEY` in your `.env` file.

### Screenshot Capture

The screenshot capture service (`backend/services/data_collection/screenshot_capture.py`) uses Selenium to:
- Navigate to TradingView charts
- Capture screenshots at specified times
- Save images to the data directory

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
- Loads paired before/after snapshots
- Preprocesses images
- Extracts features
- Trains with validation split
- Saves checkpoints and metrics

## Learning Evaluation

The system tracks:
- **Prediction Accuracy**: MAE, RMSE, direction accuracy
- **Calibration**: How well probability estimates match reality
- **Learning Progress**: Improvement over epochs
- **Pattern Recognition**: Ability to identify chart patterns

## Database Schema

- `snapshots`: Raw screenshot data
- `price_data`: OHLCV market data
- `training_samples`: Processed training data with labels
- `model_checkpoints`: Saved model versions
- `predictions`: User predictions and model outputs
- `learning_metrics`: Training/validation metrics
- `user_notes`: User notes and annotations

## Configuration

Key settings in `backend/config/settings.py`:
- `SYMBOLS`: Trading symbols (default: ["MNQ1!", "MES1!"] — MNQ & MES micro futures only)
- `BEFORE_SNAPSHOT_TIME`: Before snapshot time (default: "06:30")
- `AFTER_SNAPSHOT_TIME`: After snapshot time (default: "08:00")
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `NUM_EPOCHS`: Training epochs (default: 100)

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

1. **Data Collection Automation**: Set up scheduled tasks for automatic screenshot capture and data collection
2. **Polygon.io WebSocket**: Implement real-time streaming data via Polygon.io WebSocket API for live updates
3. **Model Improvements**: Experiment with different architectures and hyperparameters
4. **Frontend**: Build frontend interface for chat, screenshot upload, and notes
5. **Real-time Updates**: Add WebSocket support for real-time predictions

## License

[Your License Here]

## Contributing

[Contributing Guidelines]

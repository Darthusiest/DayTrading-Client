# Day Trading AI Agent Backend

A backend system for a day trading AI agent that learns from NY AM (6:30 AM - 8:00 AM PST) price chart screenshots of Nasdaq and S&P 500 futures. The agent predicts price movements, evaluates learning performance, and provides probability assessments.

## Features

- **Data Collection**: TradingView integration for capturing chart screenshots and market data
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
DATABASE_URL=postgresql://user:password@localhost:5432/daytrade
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
- `POST /api/v1/predict` - Submit screenshot and expected price, get prediction
- `GET /api/v1/predict/history` - Get prediction history

#### Training
- `POST /api/v1/train` - Trigger model training
- `GET /api/v1/train/status` - Get training status

#### Evaluation
- `GET /api/v1/evaluation/learning-status` - Get learning performance metrics
- `GET /api/v1/evaluation/learning-curve` - Get learning curve data
- `GET /api/v1/evaluation/best-model` - Get best model information

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

## Data Collection

The system is designed to capture chart screenshots at:
- **Before snapshot**: 6:30 AM PST
- **After snapshot**: 8:00 AM PST

### TradingView Integration

The TradingView client (`backend/services/data_collection/tradingview_client.py`) provides:
- Price data fetching (OHLCV)
- Market status checking
- Session date management

**Note**: Full TradingView API integration requires implementation based on TradingView's API documentation or WebSocket protocol.

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
- `SYMBOLS`: Trading symbols (default: ["NQ1!", "ES1!"])
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

1. **TradingView Integration**: Implement full TradingView API/WebSocket integration
2. **Data Collection Automation**: Set up scheduled tasks for automatic screenshot capture
3. **Model Improvements**: Experiment with different architectures and hyperparameters
4. **Frontend**: Build frontend interface for chat, screenshot upload, and notes
5. **Real-time Updates**: Add WebSocket support for real-time predictions

## License

[Your License Here]

## Contributing

[Contributing Guidelines]

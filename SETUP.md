# Setup Guide

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Set Environment Variables**
Create a `.env` file:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/daytrade
TRADINGVIEW_USERNAME=your_username
TRADINGVIEW_PASSWORD=your_password
DEBUG=True
```

3. **Initialize Database**
```bash
python scripts/init_db.py
```

4. **Run the API Server**
```bash
python run.py
# Or
uvicorn backend.api.main:app --reload
```

## Data Collection Workflow

1. **Collect Snapshots**
   - Run `scripts/collect_data.py` at 6:30 AM PST (before) and 8:00 AM PST (after)
   - Or set up a cron job/scheduler to run automatically

2. **Process Training Data**
   - After collecting snapshots, run `scripts/process_training_data.py`
   - This creates training samples with labels

3. **Train Model**
   - Once you have enough training samples (minimum 10), trigger training:
   ```bash
   curl -X POST http://localhost:8000/api/v1/train
   ```

## Testing the API

### Make a Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@path/to/chart_screenshot.png" \
  -F "expected_price=15000.0" \
  -F "symbol=NQ1!"
```

### Get Learning Status
```bash
curl http://localhost:8000/api/v1/evaluation/learning-status
```

### Create a Note
```bash
curl -X POST "http://localhost:8000/api/v1/notes" \
  -H "Content-Type: application/json" \
  -d '{"content": "Market looks bullish today", "symbol": "NQ1!"}'
```

## Important Notes

- **TradingView Integration**: The TradingView client is a placeholder. You'll need to implement actual API integration based on TradingView's documentation or use their WebSocket protocol.

- **Screenshot Capture**: Requires Chrome/Chromium and ChromeDriver. Install ChromeDriver:
  ```bash
  # macOS
  brew install chromedriver
  
  # Or download from https://chromedriver.chromium.org/
  ```

- **Model Training**: The model requires at least 10 training samples to start training. More data = better performance.

- **Database**: Make sure PostgreSQL is running and the database exists before initializing.

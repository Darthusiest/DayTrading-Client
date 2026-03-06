"""Configuration settings for the Day Trading AI Agent backend."""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Day Trading AI Agent"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Database (default SQLite so app runs without PostgreSQL; set DATABASE_URL for Postgres)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./data/daytrade.db"
    )
    
    # File Storage
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = DATA_DIR / "models"
    # Databento: raw batch downloads (e.g. glbx-mdp3-YYYYMMDD.ohlcv-1m.dbn.zst); ingest → SessionMinuteBar
    DATABENTO_RAW_DIR: Path = DATA_DIR / "databento" / "raw"
    # Map Databento raw_symbol substring to app symbol (e.g. MNQ -> MNQ1!, MES -> MES1!). Env: JSON object.
    DATABENTO_SYMBOL_MAP: dict[str, str] = {"MNQ": "MNQ1!", "MES": "MES1!"}
    
    # Market Data — focused on MNQ & MES only so the model gets really good at these
    # MNQ = Micro E-mini Nasdaq, MES = Micro E-mini S&P 500
    # Minute bars are ingested from Databento (see docs/databento_storage.md)
    SYMBOLS: list[str] = ["MNQ1!", "MES1!"]
    # Session: US RTH 9:30 AM – 4:00 PM Eastern (market open/close)
    SESSION_START_TIME: str = os.getenv("SESSION_START_TIME", "09:30")
    SESSION_END_TIME: str = os.getenv("SESSION_END_TIME", "16:00")
    SESSION_TIMEZONE: str = os.getenv("SESSION_TIMEZONE", "America/New_York")
    BEFORE_SNAPSHOT_TIME: str = os.getenv("BEFORE_SNAPSHOT_TIME", "09:30")  # Align with session open
    AFTER_SNAPSHOT_TIME: str = os.getenv("AFTER_SNAPSHOT_TIME", "16:00")   # Align with session close
    TIMEZONE: str = "America/Los_Angeles"
    # Session candle capture (9:31–16:00) is long-running; disable by default
    ENABLE_SESSION_CANDLE_CAPTURE: bool = os.getenv("ENABLE_SESSION_CANDLE_CAPTURE", "False").lower() == "true"

    # TradingView (optional — for chart screenshot capture; login gives full chart access)
    TRADINGVIEW_USERNAME: str = os.getenv("TRADINGVIEW_USERNAME", "")
    TRADINGVIEW_PASSWORD: str = os.getenv("TRADINGVIEW_PASSWORD", "")
    CHART_INTERVAL_MINUTES: int = int(os.getenv("CHART_INTERVAL_MINUTES", "15"))  # Chart timeframe (1, 5, 15, 60, etc.)
    TRADINGVIEW_LOGIN_WAIT_SECONDS: int = int(os.getenv("TRADINGVIEW_LOGIN_WAIT_SECONDS", "15"))  # Seconds to wait after login for redirect
    TRADINGVIEW_HEADLESS: bool = os.getenv("TRADINGVIEW_HEADLESS", "True").lower() == "true"  # False = show browser window (for debugging login)
    TRADINGVIEW_MANUAL_LOGIN: bool = os.getenv("TRADINGVIEW_MANUAL_LOGIN", "False").lower() == "true"  # True = wait for user to log in manually, then take screenshot

    # Scheduled data collection (APScheduler)
    ENABLE_SCHEDULED_COLLECTION: bool = os.getenv("ENABLE_SCHEDULED_COLLECTION", "True").lower() == "true"
    COLLECTION_CAPTURE_SCREENSHOTS: bool = os.getenv("COLLECTION_CAPTURE_SCREENSHOTS", "True").lower() == "true"
    COLLECTION_CHART_WAIT_SECONDS: int = int(os.getenv("COLLECTION_CHART_WAIT_SECONDS", "15"))
    
    # ML Model Configuration
    MODEL_NAME: str = "price_predictor"
    NUM_FEATURES: int = 18  # Time + price + session bars + chart patterns (match trainer _extract_feature_vector)
    NUM_LSTM_LAYERS: int = int(os.getenv("NUM_LSTM_LAYERS", "2"))
    LSTM_HIDDEN_SIZE: int = int(os.getenv("LSTM_HIDDEN_SIZE", "128"))
    # Comma-separated in .env (e.g. 256,256,128); parsed to list in MLP_HIDDENS
    MLP_HIDDENS_STR: str = Field(default="256,128", alias="MLP_HIDDENS")
    CNN_TRAINABLE_PARAM_GROUPS: int = int(os.getenv("CNN_TRAINABLE_PARAM_GROUPS", "10"))  # 0 = unfreeze all CNN
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))  # Larger batches for more stable gradients
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "1e-4"))
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "50"))  # Default epochs for full training runs
    # Quick mode: faster, lighter runs for experimentation
    QUICK_MODE: bool = os.getenv("QUICK_MODE", "False").lower() == "true"
    QUICK_NUM_EPOCHS: int = int(os.getenv("QUICK_NUM_EPOCHS", "10"))
    QUICK_BATCH_SIZE: int = int(os.getenv("QUICK_BATCH_SIZE", "64"))
    # Data loading workers for PyTorch DataLoader
    DATA_LOADER_WORKERS: int = int(os.getenv("DATA_LOADER_WORKERS", "2"))
    # Early stopping: stop when val loss stops improving
    EARLY_STOP_PATIENCE: int = int(os.getenv("EARLY_STOP_PATIENCE", "10"))
    EARLY_STOP_MIN_DELTA: float = float(os.getenv("EARLY_STOP_MIN_DELTA", "0.0"))
    # Next-minute bar model: dataset cache and early stopping
    BAR_CACHE_DATASET: bool = os.getenv("BAR_CACHE_DATASET", "True").lower() == "true"
    BAR_REBUILD_CACHE: bool = os.getenv("BAR_REBUILD_CACHE", "False").lower() == "true"
    BAR_EARLY_STOP_PATIENCE: int = int(os.getenv("BAR_EARLY_STOP_PATIENCE", "5"))
    BAR_EARLY_STOP_MIN_DELTA: float = float(os.getenv("BAR_EARLY_STOP_MIN_DELTA", "0.0"))
    # Early stop and best checkpoint. Higher-is-better: direction_5m_macro_f1, direction_5m_accuracy, breakout_10m_accuracy.
    # Lower-is-better (we negate): price_rmse, volatility_10m_rmse.
    BAR_EARLY_STOP_METRIC: str = os.getenv("BAR_EARLY_STOP_METRIC", "breakout_10m_accuracy")
    # Lookback window (bars). 15–30 recommended for next-minute; 120–240 for direction-focused runs.
    BAR_LOOKBACK: int = int(os.getenv("BAR_LOOKBACK", "30"))
    # dir5 (5m direction): threshold for up/down. Sideways if |ret_5m| < threshold. Smaller = fewer sideways (rarer).
    # 0.0003 = 0.03% (only very small 5m returns count as sideways).
    BAR_DIR5_THRESHOLD: float = float(os.getenv("BAR_DIR5_THRESHOLD", "0.0003"))
    # 5m direction head: 0 = single linear, >0 = hidden size for 2-layer MLP (e.g. 128).
    BAR_DIR5_HEAD_HIDDEN: int = int(os.getenv("BAR_DIR5_HEAD_HIDDEN", "128"))
    # Label smoothing for direction CrossEntropy (e.g. 0.1); can help 3-class accuracy.
    BAR_DIR5_LABEL_SMOOTHING: float = float(os.getenv("BAR_DIR5_LABEL_SMOOTHING", "0.1"))
    # Scale class weights for minority classes (down/up); 1.0 = inverse-freq only, >1 = push harder on minority.
    BAR_DIR5_MINORITY_WEIGHT_SCALE: float = float(os.getenv("BAR_DIR5_MINORITY_WEIGHT_SCALE", "1.5"))
    # Confidence threshold for direction: if max(prob) >= this and argmax in {down,up}, use that class; else predict sideways. 0 = raw argmax.
    BAR_DIR5_CONFIDENCE_THRESHOLD: float = float(os.getenv("BAR_DIR5_CONFIDENCE_THRESHOLD", "0.0"))
    # Focal loss for direction (recommended for imbalanced down/sideways/up): down-weights easy examples.
    BAR_DIR5_USE_FOCAL: bool = os.getenv("BAR_DIR5_USE_FOCAL", "True").lower() == "true"
    BAR_DIR5_FOCAL_GAMMA: float = float(os.getenv("BAR_DIR5_FOCAL_GAMMA", "2.0"))
    # Breakout: require move beyond recent range by > k * ATR to count as breakout.
    BAR_BREAKOUT_ATR_K: float = float(os.getenv("BAR_BREAKOUT_ATR_K", "1.0"))
    BAR_ATR_WINDOW: int = int(os.getenv("BAR_ATR_WINDOW", "14"))
    # Per-session normalization (inputs and optionally return/vol targets).
    BAR_NORMALIZE_INPUTS: bool = os.getenv("BAR_NORMALIZE_INPUTS", "True").lower() == "true"
    # When True, normalize each bar using only past bars (expanding window); avoids lookahead. Default False.
    BAR_NORMALIZE_INPUTS_EXPANDING: bool = os.getenv("BAR_NORMALIZE_INPUTS_EXPANDING", "False").lower() == "true"
    BAR_NORMALIZE_RETURN_TARGET: bool = os.getenv("BAR_NORMALIZE_RETURN_TARGET", "False").lower() == "true"
    BAR_NORMALIZE_VOL_TARGET: bool = os.getenv("BAR_NORMALIZE_VOL_TARGET", "False").lower() == "true"
    # Multi-task loss weights: breakout + volatility as primary; price/direction auxiliary.
    BAR_LOSS_WEIGHT_PRICE: float = float(os.getenv("BAR_LOSS_WEIGHT_PRICE", "0.3"))
    BAR_LOSS_WEIGHT_DIR5: float = float(os.getenv("BAR_LOSS_WEIGHT_DIR5", "0.3"))
    BAR_LOSS_WEIGHT_VOL: float = float(os.getenv("BAR_LOSS_WEIGHT_VOL", "2.0"))
    BAR_LOSS_WEIGHT_BREAKOUT: float = float(os.getenv("BAR_LOSS_WEIGHT_BREAKOUT", "2.0"))
    # Staged training: "all", "heads_only", or "direction_first" (train only direction head for BAR_DIR5_FIRST_EPOCHS then unfreeze all).
    BAR_TRAIN_PHASE: str = os.getenv("BAR_TRAIN_PHASE", "all")
    # When BAR_TRAIN_PHASE=direction_first, number of epochs to train only the direction head before unfreezing.
    BAR_DIR5_FIRST_EPOCHS: int = int(os.getenv("BAR_DIR5_FIRST_EPOCHS", "3"))
    # After training, tune confidence threshold on val for max macro-F1 and save in metrics (default False).
    BAR_DIR5_TUNE_THRESHOLD: bool = os.getenv("BAR_DIR5_TUNE_THRESHOLD", "False").lower() == "true"
    # When True, load existing next_minute_lstm.pt (if present) and continue training from it.
    BAR_RESUME_FROM_CHECKPOINT: bool = os.getenv("BAR_RESUME_FROM_CHECKPOINT", "False").lower() == "true"
    IMAGE_SIZE: tuple[int, int] = (224, 224)
    VALIDATION_SPLIT: float = 0.2  # Fraction for validation (time-based split)
    TEST_SPLIT: float = 0.1        # Fraction for test set (time-based split)
    # Optional seed for reproducible train-set shuffle (avoids memorization while keeping split time-based)
    _rs: str | None = os.getenv("RANDOM_SEED")
    RANDOM_SEED: int | None = int(_rs) if (_rs and _rs.strip()) else None

    @computed_field
    @property
    def MLP_HIDDENS(self) -> list[int]:
        """Parsed from MLP_HIDDENS_STR (env MLP_HIDDENS=256,256,128)."""
        parts = [x.strip() for x in self.MLP_HIDDENS_STR.split(",") if x.strip()]
        return [int(x) for x in parts] if parts else [256, 128]

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    class Config:
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")
        case_sensitive = True
        extra = "ignore"  # Ignore unknown env vars (e.g. legacy POLYGON_* after removal)


# Create data directories if they don't exist
settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATABENTO_RAW_DIR.mkdir(parents=True, exist_ok=True)

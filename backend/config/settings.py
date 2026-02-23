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
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "200"))  # More epochs for better convergence
    IMAGE_SIZE: tuple[int, int] = (224, 224)
    VALIDATION_SPLIT: float = 0.2  # Fraction for validation (time-based split)
    TEST_SPLIT: float = 0.1        # Fraction for test set (time-based split)

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

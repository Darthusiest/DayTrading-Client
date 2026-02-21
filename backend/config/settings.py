"""Configuration settings for the Day Trading AI Agent backend."""
import os
from pathlib import Path
from typing import Optional
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
    
    # Polygon.io Configuration
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
    
    # Market Data — focused on MNQ & MES only so the model gets really good at these
    # MNQ = Micro E-mini Nasdaq, MES = Micro E-mini S&P 500
    # Collecting only these two avoids dilution from other symbols
    SYMBOLS: list[str] = ["MNQ1!", "MES1!"]
    POLYGON_SYMBOL_MAP: dict[str, str] = {
        "MNQ1!": "C:MNQ1",  # Micro E-mini Nasdaq continuous
        "MES1!": "C:MES1",  # Micro E-mini S&P 500 continuous
    }
    BEFORE_SNAPSHOT_TIME: str = "06:30"  # PST
    AFTER_SNAPSHOT_TIME: str = "08:00"  # PST
    TIMEZONE: str = "America/Los_Angeles"

    # TradingView (optional — for chart screenshot capture; login gives full chart access)
    TRADINGVIEW_USERNAME: str = os.getenv("TRADINGVIEW_USERNAME", "")
    TRADINGVIEW_PASSWORD: str = os.getenv("TRADINGVIEW_PASSWORD", "")

    # Scheduled data collection (APScheduler)
    ENABLE_SCHEDULED_COLLECTION: bool = os.getenv("ENABLE_SCHEDULED_COLLECTION", "True").lower() == "true"
    COLLECTION_CAPTURE_SCREENSHOTS: bool = os.getenv("COLLECTION_CAPTURE_SCREENSHOTS", "True").lower() == "true"
    COLLECTION_CHART_WAIT_SECONDS: int = int(os.getenv("COLLECTION_CHART_WAIT_SECONDS", "15"))
    
    # ML Model Configuration
    MODEL_NAME: str = "price_predictor"
    NUM_FEATURES: int = 18  # Time + price + session bars + chart patterns (match trainer _extract_feature_vector)
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 100
    IMAGE_SIZE: tuple[int, int] = (224, 224)
    VALIDATION_SPLIT: float = 0.2  # Fraction for validation (time-based split)
    TEST_SPLIT: float = 0.1        # Fraction for test set (time-based split)
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create data directories if they don't exist
settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

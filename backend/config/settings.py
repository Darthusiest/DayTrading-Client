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
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost:5432/daytrade"
    )
    
    # File Storage
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = DATA_DIR / "models"
    
    # TradingView Configuration
    TRADINGVIEW_USERNAME: Optional[str] = os.getenv("TRADINGVIEW_USERNAME")
    TRADINGVIEW_PASSWORD: Optional[str] = os.getenv("TRADINGVIEW_PASSWORD")
    TRADINGVIEW_SESSION_ID: Optional[str] = os.getenv("TRADINGVIEW_SESSION_ID")
    
    # Market Data
    SYMBOLS: list[str] = ["NQ1!", "ES1!"]  # Nasdaq and S&P 500 futures
    BEFORE_SNAPSHOT_TIME: str = "06:30"  # PST
    AFTER_SNAPSHOT_TIME: str = "08:00"  # PST
    TIMEZONE: str = "America/Los_Angeles"
    
    # ML Model Configuration
    MODEL_NAME: str = "price_predictor"
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 100
    IMAGE_SIZE: tuple[int, int] = (224, 224)
    VALIDATION_SPLIT: float = 0.2
    
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
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

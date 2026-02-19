"""Database models for the Day Trading AI Agent."""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.database.db import Base


class Snapshot(Base):
    """Raw screenshot data before and after NY AM session."""
    __tablename__ = "snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    snapshot_type = Column(String(20), nullable=False)  # 'before' or 'after'
    timestamp = Column(DateTime, nullable=False, index=True)
    image_path = Column(String(500), nullable=False)
    session_date = Column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    price_data = relationship("PriceData", back_populates="snapshot", uselist=False)
    training_samples = relationship("TrainingSample", back_populates="snapshot")


class PriceData(Base):
    """Market data (OHLCV) for labeling."""
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True, index=True)
    snapshot_id = Column(Integer, ForeignKey("snapshots.id"), unique=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    snapshot = relationship("Snapshot", back_populates="price_data")


class TrainingSample(Base):
    """Processed training data with features and labels."""
    __tablename__ = "training_samples"
    
    id = Column(Integer, primary_key=True, index=True)
    snapshot_id = Column(Integer, ForeignKey("snapshots.id"))
    symbol = Column(String(20), nullable=False, index=True)
    session_date = Column(String(10), nullable=False, index=True)
    
    # Features (stored as JSON for flexibility)
    features = Column(JSON, nullable=True)
    processed_image_path = Column(String(500), nullable=True)
    
    # Labels
    price_change_absolute = Column(Float, nullable=True)
    price_change_percentage = Column(Float, nullable=True)
    direction = Column(String(10), nullable=True)  # 'up', 'down', 'sideways'
    target_hit = Column(Boolean, nullable=True)
    expected_price = Column(Float, nullable=True)
    actual_price = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    is_used_for_training = Column(Boolean, default=False, index=True)
    
    # Relationships
    snapshot = relationship("Snapshot", back_populates="training_samples")


class ModelCheckpoint(Base):
    """Saved model versions."""
    __tablename__ = "model_checkpoints"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    checkpoint_path = Column(String(500), nullable=False)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    train_accuracy = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)  # Additional metrics
    is_best = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """User predictions and model outputs."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    user_expected_price = Column(Float, nullable=False)
    model_predicted_price = Column(Float, nullable=False)
    probability_hit = Column(Float, nullable=False)  # Probability of expected price being hit
    screenshot_path = Column(String(500), nullable=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    
    # Learning metrics
    model_confidence = Column(Float, nullable=True)
    learning_score = Column(Float, nullable=True)  # How well model learned
    
    # Actual outcome (filled later)
    actual_price = Column(Float, nullable=True)
    was_hit = Column(Boolean, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class LearningMetric(Base):
    """Training/validation metrics over time."""
    __tablename__ = "learning_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    epoch = Column(Integer, nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # 'train_loss', 'val_loss', 'accuracy', etc.
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Additional context
    metadata = Column(JSON, nullable=True)


class UserNote(Base):
    """User notes and annotations."""
    __tablename__ = "user_notes"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=True)
    content = Column(Text, nullable=False)
    symbol = Column(String(20), nullable=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

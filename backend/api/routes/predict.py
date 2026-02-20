"""Prediction API endpoints."""
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from pathlib import Path
from datetime import datetime
from backend.database.db import get_db
from backend.database.models import Prediction
from backend.services.ml.inference.predictor import Predictor
from backend.services.data_processing.feature_extractor import FeatureExtractor
from backend.config.settings import settings

router = APIRouter(prefix="/predict", tags=["prediction"])
predictor = Predictor()
feature_extractor = FeatureExtractor()


@router.post("")
async def predict_price(
    file: UploadFile = File(...),
    expected_price: Optional[float] = Form(None),
    symbol: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Submit a screenshot (and optionally an expected price) to get the model's price estimate.

    The screenshot is intended to be the market **before or at 6:30 AM PST** (NY AM start)
    for best results. The returned **model_predicted_price** is the model's estimate of the
    price level the market will reach by end of NY AM (8:00 AM PST).

    Args:
        file: Chart screenshot image (ideally before/at 6:30 AM PST).
        expected_price: Optional. User's expected price; if provided, probability_hit
            is the probability that this level is hit. If omitted, probability_hit
            is the model's base confidence in its predicted level.
        symbol: Trading symbol (e.g., 'NQ1!', 'ES1!').

    Returns:
        prediction_id, symbol, user_expected_price (null if not provided),
        model_predicted_price (estimated price market will hit), probability_hit,
        model_confidence, learning_score, timestamp.
    """
    try:
        # Load model if not already loaded
        if not predictor.model_loaded:
            predictor.load_model(db=db)
        
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{symbol}_{timestamp}_{file.filename}"
        filepath = settings.RAW_DATA_DIR / filename
        
        # Save file
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract features
        features = feature_extractor.extract_features(
            filepath,
            datetime.now(),
            symbol
        )
        
        # Make prediction (expected_price=None supported; returns model estimate + confidence)
        result = predictor.predict(
            filepath,
            expected_price=expected_price,
            features=features
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # probability_hit: when expected_price given, P(hit that level); else model base confidence
        prob_hit = result["probability_hit"]
        if prob_hit is None:
            prob_hit = result.get("model_confidence") or 0.0

        # Save prediction to database (user_expected_price may be None)
        prediction = Prediction(
            symbol=symbol,
            user_expected_price=expected_price,
            model_predicted_price=result["predicted_price"],
            probability_hit=prob_hit,
            screenshot_path=str(filepath),
            model_confidence=result.get("model_confidence"),
            learning_score=result.get("learning_score")
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        
        return {
            "prediction_id": prediction.id,
            "symbol": symbol,
            "user_expected_price": expected_price,
            "model_predicted_price": result["predicted_price"],
            "probability_hit": prediction.probability_hit,
            "model_confidence": result.get("model_confidence"),
            "learning_score": result.get("learning_score"),
            "timestamp": prediction.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/history")
def get_prediction_history(
    symbol: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get prediction history."""
    query = db.query(Prediction)
    
    if symbol:
        query = query.filter(Prediction.symbol == symbol)
    
    predictions = query.order_by(Prediction.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": p.id,
            "symbol": p.symbol,
            "user_expected_price": p.user_expected_price,
            "model_predicted_price": p.model_predicted_price,
            "probability_hit": p.probability_hit,
            "model_confidence": p.model_confidence,
            "learning_score": p.learning_score,
            "actual_price": p.actual_price,
            "was_hit": p.was_hit,
            "timestamp": p.timestamp.isoformat()
        }
        for p in predictions
    ]

"""Training API endpoints."""
import logging
from datetime import datetime
from typing import Any

import torch
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader

from backend.config.settings import settings
from backend.database.db import SessionLocal, get_db
from backend.database.models import JobRun, TrainingSample, ModelCheckpoint
from backend.services.data_processing.image_preprocessor import ImagePreprocessor
from backend.services.ml.models.price_predictor import PricePredictor, price_predictor_kwargs_from_settings
from backend.services.ml.training.trainer import Trainer, PriceDataset
from backend.api.deps.security import require_api_key
from backend.services.ops.jobs import create_job, get_job, update_job

router = APIRouter(prefix="/train", tags=["training"])
logger = logging.getLogger(__name__)
MIN_TRAIN_SAMPLES = 2  # Lowered for early-stage testing; increase when you have more data


def train_model_task(job_id: str):
    """Background task for training the model."""
    db = SessionLocal()
    update_job(db, job_id, status="running", started_at=datetime.utcnow())
    try:
        # Get training samples (time-ordered)
        samples = db.query(TrainingSample).filter(
            TrainingSample.is_used_for_training == False,
            TrainingSample.actual_price.isnot(None)
        ).order_by(TrainingSample.session_date, TrainingSample.id).all()

        if len(samples) < MIN_TRAIN_SAMPLES:
            logger.warning(f"Insufficient training samples: {len(samples)}")
            update_job(db, job_id, status="skipped", finished_at=datetime.utcnow())
            return

        # Time-based split: train / validation / test
        n = len(samples)
        test_ratio = getattr(settings, "TEST_SPLIT", 0.1)
        val_ratio = settings.VALIDATION_SPLIT
        train_ratio = 1.0 - val_ratio - test_ratio
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        train_samples = samples[:n_train]
        val_samples = samples[n_train : n_train + n_val]
        test_samples = samples[n_train + n_val :] if n_test > 0 else []

        logger.info(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

        # Create datasets (each sample = one chart + its labels; split is time-based so no future leakage)
        image_preprocessor = ImagePreprocessor()
        train_dataset = PriceDataset(train_samples, image_preprocessor)
        val_dataset = PriceDataset(val_samples, image_preprocessor)
        test_dataset = PriceDataset(test_samples, image_preprocessor) if test_samples else None

        # Effective hyperparameters: allow a faster "quick mode" for experimentation
        batch_size = settings.QUICK_BATCH_SIZE if settings.QUICK_MODE else settings.BATCH_SIZE
        num_workers = settings.DATA_LOADER_WORKERS
        num_epochs = settings.QUICK_NUM_EPOCHS if settings.QUICK_MODE else settings.NUM_EPOCHS

        # Shuffle training set each epoch to avoid memorization; optional seed for reproducibility
        train_generator = (torch.Generator().manual_seed(settings.RANDOM_SEED) if settings.RANDOM_SEED is not None else None)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=train_generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = None
        if test_dataset is not None and len(test_samples) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        # Initialize model and trainer
        model = PricePredictor(**price_predictor_kwargs_from_settings())
        trainer = Trainer(model)

        # Train (plot_show=False in background to avoid requiring a display)
        save_dir = settings.MODELS_DIR / settings.MODEL_NAME
        history = trainer.train(
            train_loader,
            val_loader,
            num_epochs,
            db,
            save_dir,
            test_loader=test_loader,
            plot_show=False
        )

        # Mark samples as used
        for sample in samples:
            sample.is_used_for_training = True
        db.commit()

        logger.info(f"Training completed. Best val loss: {history['best_val_loss']}. Plot: {history.get('plot_path')}")
        update_job(
            db,
            job_id,
            status="completed",
            finished_at=datetime.utcnow(),
            details={"best_val_loss": history.get("best_val_loss"), "plot_path": history.get("plot_path")},
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        db.rollback()
        update_job(db, job_id, status="failed", error=str(e), finished_at=datetime.utcnow())
    finally:
        db.close()


@router.post("")
def trigger_training(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: None = Depends(require_api_key),
):
    """Trigger model training on available data."""
    # Check if there's enough data
    sample_count = db.query(TrainingSample).filter(
        TrainingSample.is_used_for_training == False,
        TrainingSample.actual_price.isnot(None)
    ).count()
    
    if sample_count < MIN_TRAIN_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient training data: {sample_count} samples (minimum {MIN_TRAIN_SAMPLES} required)"
        )
    
    # Start training in background
    job = create_job(
        db,
        job_type="training",
        status="pending",
        details={"training_samples": sample_count},
    )
    job_id = job.job_id
    background_tasks.add_task(train_model_task, job_id)
    
    return {
        "status": "training_started",
        "message": "Model training started in background",
        "training_samples": sample_count,
        "job_id": job_id,
    }


@router.get("/status")
def get_training_status(db: Session = Depends(get_db)):
    """Get training status and available data."""
    try:
        total_samples = db.query(TrainingSample).count()
        unused_samples = db.query(TrainingSample).filter(
            TrainingSample.is_used_for_training == False,
            TrainingSample.actual_price.isnot(None)
        ).count()

        latest_checkpoint = db.query(ModelCheckpoint).order_by(
            ModelCheckpoint.created_at.desc()
        ).first()

        status = {
            "total_samples": total_samples,
            "unused_samples": unused_samples,
            "can_train": unused_samples >= MIN_TRAIN_SAMPLES,
            "min_train_samples": MIN_TRAIN_SAMPLES,
        }

        if latest_checkpoint:
            created_at = latest_checkpoint.created_at
            status["latest_model"] = {
                "version": latest_checkpoint.version,
                "epoch": latest_checkpoint.epoch,
                "val_loss": latest_checkpoint.val_loss,
                "val_accuracy": latest_checkpoint.val_accuracy,
                "created_at": created_at.isoformat() if created_at else None
            }

        return status
    except Exception as e:
        logger.exception("get_training_status failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
def get_training_job(job_id: str, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
    return {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "status": job.status,
        "error": job.error,
        "details": job.details or {},
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }

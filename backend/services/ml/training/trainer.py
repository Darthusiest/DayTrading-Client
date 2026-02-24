"""Training pipeline for the price prediction model."""
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from sqlalchemy.orm import Session
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server; use default for scripts
import matplotlib.pyplot as plt
from backend.services.ml.models.price_predictor import PricePredictor
from backend.database.models import TrainingSample, ModelCheckpoint, LearningMetric
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class PriceDataset(Dataset):
    """Dataset for price prediction training."""
    
    def __init__(
        self,
        samples: List[TrainingSample],
        image_preprocessor,
        device: str = "cpu"
    ):
        self.samples = samples
        self.image_preprocessor = image_preprocessor
        self.device = device
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Image: bar-only samples have processed_image_path (e.g. placeholder); else use snapshot
        image_path = (
            Path(sample.processed_image_path)
            if sample.processed_image_path
            else (Path(sample.snapshot.image_path) if sample.snapshot else None)
        )
        if image_path is None or not image_path.is_file():
            image_path = Path(settings.PROCESSED_DATA_DIR) / "placeholder_chart.png"
        image = self.image_preprocessor.preprocess(image_path)
        
        if image is None:
            raise ValueError(f"Failed to preprocess image: {image_path}")
        
        # Convert to tensor and change from HWC to CHW
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        # Get labels
        actual_price = sample.actual_price if sample.actual_price else 0.0
        expected_price = sample.expected_price if sample.expected_price else actual_price
        change_abs = sample.price_change_absolute if sample.price_change_absolute is not None else None
        change_pct = sample.price_change_percentage if sample.price_change_percentage is not None else None
        # Before price for difference/ratio targets (actual = before + change)
        if change_abs is not None and actual_price != 0:
            before_price = float(actual_price - change_abs)
        else:
            before_price = float(actual_price) if actual_price != 0 else 1.0
        if before_price <= 0:
            before_price = 1.0
        target_change = float(change_abs) if change_abs is not None else 0.0
        target_ratio = float(actual_price / before_price) if before_price > 0 else 1.0

        # Features (if available)
        features = sample.features or {}
        feature_vector = self._extract_feature_vector(features)

        return {
            "image": image_tensor,
            "features": torch.tensor(feature_vector, dtype=torch.float32),
            "actual_price": torch.tensor(actual_price, dtype=torch.float32),
            "expected_price": torch.tensor(expected_price, dtype=torch.float32),
            "target_hit": torch.tensor(1.0 if sample.target_hit else 0.0, dtype=torch.float32),
            "before_price": torch.tensor(before_price, dtype=torch.float32),
            "target_change": torch.tensor(target_change, dtype=torch.float32),
            "target_ratio": torch.tensor(target_ratio, dtype=torch.float32),
        }
    
    def _extract_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Extract numeric features from features dict (includes SessionMinuteBar summary)."""
        feature_list = []
        
        # Time features
        feature_list.append(features.get("hour", 0) / 24.0)
        feature_list.append(features.get("minute", 0) / 60.0)
        feature_list.append(features.get("day_of_week", 0) / 6.0)
        
        # Price features
        feature_list.append(features.get("price_change_pct", 0.0))
        feature_list.append(features.get("price_range_pct", 0.0))
        
        # Session bar features (session start–end, e.g. 9:30–16:00 ET)
        feature_list.append(features.get("session_return_pct", 0.0) / 100.0)
        feature_list.append(features.get("session_range_pct", 0.0) / 100.0)
        feature_list.append(min(1.0, features.get("session_volatility", 0.0)))
        feature_list.append(min(1.0, features.get("session_num_bars", 0) / 100.0))
        
        # Chart pattern features
        trend = features.get("trend_direction", "unknown")
        trend_val = 0.0 if trend == "down" else (0.5 if trend == "sideways" else 1.0)
        feature_list.append(trend_val)
        feature_list.append(features.get("volatility_estimate", 0.0))
        feature_list.append(1.0 if features.get("has_support_level") else 0.0)
        feature_list.append(1.0 if features.get("has_resistance_level") else 0.0)
        
        target_size = settings.NUM_FEATURES
        while len(feature_list) < target_size:
            feature_list.append(0.0)
        return feature_list[:target_size]


class Trainer:
    """Trainer for the price prediction model."""
    
    def __init__(
        self,
        model: PricePredictor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=settings.LEARNING_RATE
        )
        self.criterion_price = nn.MSELoss()
        self.criterion_prob = nn.BCELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_price_loss = 0.0
        total_prob_loss = 0.0
        total_change_loss = 0.0
        total_ratio_loss = 0.0
        correct_direction = 0
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            images = batch["image"].to(self.device)
            features = batch["features"].to(self.device)
            actual_prices = batch["actual_price"].to(self.device)
            target_hits = batch["target_hit"].to(self.device)
            before_prices = batch["before_price"].to(self.device)
            target_changes = batch["target_change"].to(self.device)
            target_ratios = batch["target_ratio"].to(self.device)

            # Forward pass
            outputs = self.model(images, features)
            predicted_prices = outputs["predicted_price"]
            probabilities = outputs["probability"]

            # Price loss (level)
            price_loss = self.criterion_price(predicted_prices, actual_prices)
            prob_loss = self.criterion_prob(probabilities, target_hits)

            # Difference loss: predicted change vs target change (actual - before)
            eps = 1e-6
            valid = before_prices > eps
            if valid.any():
                pred_change = predicted_prices - before_prices
                change_loss = self.criterion_price(pred_change[valid], target_changes[valid])
            else:
                change_loss = torch.tensor(0.0, device=self.device)

            # Ratio loss: predicted price/before vs actual/before
            if valid.any():
                pred_ratio = (predicted_prices / before_prices.clamp(min=eps))[valid]
                ratio_loss = self.criterion_price(pred_ratio, target_ratios[valid])
            else:
                ratio_loss = torch.tensor(0.0, device=self.device)

            # Combined loss (weighted): level + probability + difference + ratio
            loss = price_loss + 0.5 * prob_loss + 0.3 * change_loss + 0.3 * ratio_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_price_loss += price_loss.item()
            total_prob_loss += prob_loss.item()
            total_change_loss += change_loss.item()
            total_ratio_loss += ratio_loss.item()

            # Train direction accuracy (same definition as validate)
            batch_mean = actual_prices.mean()
            actual_up = (actual_prices > batch_mean).float()
            pred_up = (predicted_prices > actual_prices).float()
            correct_direction += (pred_up == actual_up).sum().item()
            total_samples += len(actual_prices)

        n = len(dataloader)
        avg_loss = total_loss / n
        avg_price_loss = total_price_loss / n
        avg_prob_loss = total_prob_loss / n
        avg_change_loss = total_change_loss / n
        avg_ratio_loss = total_ratio_loss / n
        direction_accuracy = correct_direction / total_samples if total_samples > 0 else 0.0

        return {
            "loss": avg_loss,
            "price_loss": avg_price_loss,
            "prob_loss": avg_prob_loss,
            "change_loss": avg_change_loss,
            "ratio_loss": avg_ratio_loss,
            "direction_accuracy": direction_accuracy,
        }
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validate or test the model."""
        self.model.eval()
        total_loss = 0.0
        total_price_loss = 0.0
        total_prob_loss = 0.0
        total_change_loss = 0.0
        total_ratio_loss = 0.0
        correct_direction = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                features = batch["features"].to(self.device)
                actual_prices = batch["actual_price"].to(self.device)
                target_hits = batch["target_hit"].to(self.device)
                before_prices = batch["before_price"].to(self.device)
                target_changes = batch["target_change"].to(self.device)
                target_ratios = batch["target_ratio"].to(self.device)

                outputs = self.model(images, features)
                predicted_prices = outputs["predicted_price"]
                probabilities = outputs["probability"]

                price_loss = self.criterion_price(predicted_prices, actual_prices)
                prob_loss = self.criterion_prob(probabilities, target_hits)
                eps = 1e-6
                valid = before_prices > eps
                if valid.any():
                    pred_change = predicted_prices - before_prices
                    change_loss = self.criterion_price(pred_change[valid], target_changes[valid])
                    pred_ratio = (predicted_prices / before_prices.clamp(min=eps))[valid]
                    ratio_loss = self.criterion_price(pred_ratio, target_ratios[valid])
                else:
                    change_loss = torch.tensor(0.0, device=self.device)
                    ratio_loss = torch.tensor(0.0, device=self.device)
                loss = price_loss + 0.5 * prob_loss + 0.3 * change_loss + 0.3 * ratio_loss

                total_loss += loss.item()
                total_price_loss += price_loss.item()
                total_prob_loss += prob_loss.item()
                total_change_loss += change_loss.item()
                total_ratio_loss += ratio_loss.item()

                batch_mean = actual_prices.mean()
                actual_up = (actual_prices > batch_mean).float()
                pred_up = (predicted_prices > actual_prices).float()
                correct_direction += (pred_up == actual_up).sum().item()
                total_samples += len(actual_prices)

        n = len(dataloader)
        avg_loss = total_loss / n if n else 0.0
        avg_price_loss = total_price_loss / n if n else 0.0
        avg_prob_loss = total_prob_loss / n if n else 0.0
        avg_change_loss = total_change_loss / n if n else 0.0
        avg_ratio_loss = total_ratio_loss / n if n else 0.0
        direction_accuracy = correct_direction / total_samples if total_samples > 0 else 0.0

        return {
            "loss": avg_loss,
            "price_loss": avg_price_loss,
            "prob_loss": avg_prob_loss,
            "change_loss": avg_change_loss,
            "ratio_loss": avg_ratio_loss,
            "direction_accuracy": direction_accuracy,
        }
    
    def plot_validation_curves(
        self,
        history: List[Dict[str, Any]],
        save_path: Path,
        show: bool = True
    ) -> None:
        """Plot validation loss and accuracy and optionally show the figure."""
        if not history:
            return
        epochs = [h["epoch"] + 1 for h in history]
        train_loss = [h["train"]["loss"] for h in history]
        val_loss = [h["val"]["loss"] for h in history]
        val_accuracy = [h["val"].get("direction_accuracy", 0.0) for h in history]
        train_accuracy = [h["train"].get("direction_accuracy", 0.0) for h in history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax1.plot(epochs, train_loss, label="Train loss", color="C0")
        ax1.plot(epochs, val_loss, label="Validation loss", color="C1")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.set_title("Training and validation loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_accuracy, label="Train accuracy", color="C0")
        ax2.plot(epochs, val_accuracy, label="Validation accuracy", color="C1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")
        ax2.set_title("Training and validation accuracy (direction)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved validation curves to {save_path}")
        if show:
            try:
                plt.show()
            except Exception as e:
                logger.debug(f"Could not show plot (e.g. no display): {e}")
        plt.close(fig)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        db: Session,
        save_dir: Path,
        test_loader: Optional[DataLoader] = None,
        plot_show: bool = True
    ) -> Dict[str, Any]:
        """Full training loop with optional test set and validation curves plot.

        Adds simple progress reporting and per-epoch / total timing so you can
        see how long training takes and how far along it is.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        training_history: List[Dict[str, Any]] = []
        overall_start = time.perf_counter()

        patience = settings.EARLY_STOP_PATIENCE
        min_delta = settings.EARLY_STOP_MIN_DELTA

        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            logger.info("Epoch %s/%s", epoch + 1, num_epochs)

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)
            current_val_loss = val_metrics["loss"]

            # Update learning rate
            self.scheduler.step(current_val_loss)

            epoch_seconds = time.perf_counter() - epoch_start
            progress_pct = ((epoch + 1) / max(1, num_epochs)) * 100.0

            # Log metrics + progress/timing
            logger.info(
                "Epoch %s/%s (%.1f%%) finished in %.2fs | "
                "Train Loss=%.4f Val Loss=%.4f Val Acc=%.4f",
                epoch + 1,
                num_epochs,
                progress_pct,
                epoch_seconds,
                train_metrics["loss"],
                current_val_loss,
                val_metrics.get("direction_accuracy", 0.0),
            )

            # Early stopping: track improvement in validation loss
            improved = current_val_loss < (best_val_loss - min_delta)
            if improved:
                best_val_loss = current_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No val loss improvement for %s epoch(s) (current=%.4f best=%.4f)",
                    epochs_without_improvement,
                    current_val_loss,
                    best_val_loss,
                )

            # Save metrics to database
            self._save_metrics(db, epoch, train_metrics, val_metrics)

            # Save checkpoint (mark as best only when val loss improves)
            is_best = improved
            checkpoint_path = self._save_checkpoint(
                db,
                epoch,
                train_metrics,
                val_metrics,
                save_dir,
                is_best
            )

            training_history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "duration_seconds": epoch_seconds,
                "checkpoint_path": str(checkpoint_path),
            })

            # Early stopping break condition
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered: no val loss improvement for %s epochs "
                    "(patience=%s). Stopping at epoch %s/%s.",
                    epochs_without_improvement,
                    patience,
                    epoch + 1,
                    num_epochs,
                )
                break

        # Final test set evaluation (optional)
        test_metrics = None
        if test_loader is not None:
            test_metrics = self.validate(test_loader)
            logger.info(
                f"Test Loss: {test_metrics['loss']:.4f}, "
                f"Test Accuracy: {test_metrics['direction_accuracy']:.4f}"
            )

        # Plot validation loss and accuracy and save / show
        plot_path = save_dir / "validation_curves.png"
        self.plot_validation_curves(training_history, plot_path, show=plot_show)

        total_seconds = time.perf_counter() - overall_start
        effective_epochs = len(training_history)
        logger.info(
            "Training complete in %.2fs (%.2f min) over %s epochs",
            total_seconds,
            total_seconds / 60.0,
            effective_epochs,
        )

        return {
            "history": training_history,
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
            "plot_path": str(plot_path),
            "total_duration_seconds": total_seconds,
        }
    
    def _save_metrics(
        self,
        db: Session,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Save metrics to database."""
        model_version = f"{settings.MODEL_NAME}_v1"
        
        for metric_type, value in train_metrics.items():
            metric = LearningMetric(
                model_version=model_version,
                epoch=epoch,
                metric_type=f"train_{metric_type}",
                metric_value=value
            )
            db.add(metric)
        
        for metric_type, value in val_metrics.items():
            metric = LearningMetric(
                model_version=model_version,
                epoch=epoch,
                metric_type=f"val_{metric_type}",
                metric_value=value
            )
            db.add(metric)
        
        db.commit()
    
    def _save_checkpoint(
        self,
        db: Session,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        save_dir: Path,
        is_best: bool
    ) -> Path:
        """Save model checkpoint."""
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }, checkpoint_path)
        
        # Save to database
        checkpoint = ModelCheckpoint(
            model_name=settings.MODEL_NAME,
            version=f"v1_epoch_{epoch}",
            checkpoint_path=str(checkpoint_path),
            epoch=epoch,
            train_loss=train_metrics.get("loss"),
            val_loss=val_metrics.get("loss"),
            train_accuracy=train_metrics.get("direction_accuracy", 0.0),
            val_accuracy=val_metrics.get("direction_accuracy", 0.0),
            metrics={
                "train": train_metrics,
                "val": val_metrics
            },
            is_best=is_best
        )
        db.add(checkpoint)
        db.commit()
        
        return checkpoint_path

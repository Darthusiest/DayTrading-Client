"""Run training with train/val/test split and show validation loss/accuracy plot in a window."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.db import SessionLocal
from backend.database.models import TrainingSample, ModelCheckpoint
from backend.services.ml.training.trainer import Trainer, PriceDataset
from backend.services.ml.models.price_predictor import PricePredictor, price_predictor_kwargs_from_settings
from backend.services.data_processing.image_preprocessor import ImagePreprocessor
from backend.config.settings import settings
from torch.utils.data import DataLoader


def main():
    db = SessionLocal()
    try:
        samples = db.query(TrainingSample).filter(
            TrainingSample.is_used_for_training == False,
            TrainingSample.actual_price.isnot(None)
        ).order_by(TrainingSample.session_date, TrainingSample.id).all()

        if len(samples) < 10:
            print(f"Insufficient training samples: {len(samples)} (need at least 10)")
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

        print(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

        image_preprocessor = ImagePreprocessor()
        train_dataset = PriceDataset(train_samples, image_preprocessor)
        val_dataset = PriceDataset(val_samples, image_preprocessor)
        test_dataset = PriceDataset(test_samples, image_preprocessor) if test_samples else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=settings.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=settings.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        test_loader = None
        if test_samples:
            test_loader = DataLoader(
                test_dataset,
                batch_size=settings.BATCH_SIZE,
                shuffle=False,
                num_workers=0
            )

        model = PricePredictor(**price_predictor_kwargs_from_settings())
        trainer = Trainer(model)
        save_dir = settings.MODELS_DIR / settings.MODEL_NAME

        # Train with plot saved only (Agg backend used by trainer)
        history = trainer.train(
            train_loader,
            val_loader,
            settings.NUM_EPOCHS,
            db,
            save_dir,
            test_loader=test_loader,
            plot_show=False
        )

        for sample in samples:
            sample.is_used_for_training = True
        db.commit()

        plot_path = history.get("plot_path")
        if plot_path and Path(plot_path).exists():
            # Show the saved plot in a matplotlib window (pops up)
            try:
                import matplotlib
                for backend in ("TkAgg", "Qt5Agg", "MacOSX", "GTK3Agg", "WXAgg"):
                    try:
                        matplotlib.use(backend)
                        break
                    except Exception:
                        continue
                import matplotlib.pyplot as plt
                img = plt.imread(plot_path)
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.axis("off")
                plt.title("Validation loss and accuracy")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not show plot window: {e}. Opening file with default viewer.")
                import subprocess
                import platform
                path = Path(plot_path)
                if platform.system() == "Darwin":
                    subprocess.run(["open", str(path)])
                elif platform.system() == "Windows":
                    subprocess.run(["start", "", str(path)], shell=True)
                else:
                    subprocess.run(["xdg-open", str(path)])
        else:
            print(f"Plot saved to: {plot_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

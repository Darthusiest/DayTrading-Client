"""Run EOD model training with train/val/test split and show validation loss/accuracy plot in a window.

Optionally initialize the PricePredictor LSTM weights from the bar-only next-minute model:

  python scripts/train_next_minute_model.py   # train bar model first
  python scripts/train_with_plot.py --init-from-bar-model
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.db import SessionLocal  # noqa: E402
from backend.database.models import TrainingSample  # noqa: E402
from backend.services.ml.training.trainer import Trainer, PriceDataset  # noqa: E402
from backend.services.ml.models.price_predictor import (  # noqa: E402
    PricePredictor,
    price_predictor_kwargs_from_settings,
)
from backend.services.data_processing.image_preprocessor import ImagePreprocessor  # noqa: E402
from backend.services.ml.bar_next_minute import (  # noqa: E402
    NextMinuteBarLSTM,
    NextMinuteModelConfig,
)
from backend.config.settings import settings  # noqa: E402


MIN_TRAIN_SAMPLES = 2  # Keep in sync with API training route for early testing


def _maybe_init_from_bar_model(model: PricePredictor) -> None:
    """
    Initialize the PricePredictor LSTM weights from the bar-only next-minute model,
    when a checkpoint exists. Copies recurrent weights exactly and input weights
    where dimensions allow (partial copy for larger input size).
    """
    ckpt_path = settings.MODELS_DIR / "next_minute_lstm.pt"
    if not ckpt_path.is_file():
        print(f"[init-from-bar-model] No next-minute checkpoint found at {ckpt_path}, skipping.")
        return

    # Recreate bar model architecture (input_size here only affects weight_ih shape)
    # We use F=5 features (OHLCV) to match train_next_minute_model.py.
    config = NextMinuteModelConfig(input_size=5)
    bar_model = NextMinuteBarLSTM(config)
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        bar_model.load_state_dict(state)
    except Exception as exc:
        print(f"[init-from-bar-model] Failed to load bar model from {ckpt_path}: {exc}")
        return

    src_state = bar_model.lstm.state_dict()
    dst_state = model.lstm.state_dict()

    updated = 0
    for name, param in dst_state.items():
        src_param = src_state.get(name)
        if src_param is None:
            continue
        if src_param.shape == param.shape:
            param.copy_(src_param)
            updated += 1
        elif "weight_ih" in name and src_param.shape[0] == param.shape[0]:
            # Partial copy for input weights when PricePredictor has larger input_size.
            cols = min(src_param.shape[1], param.shape[1])
            param[:, :cols].copy_(src_param[:, :cols])
            updated += 1

    model.lstm.load_state_dict(dst_state)
    print(f"[init-from-bar-model] Copied {updated} LSTM parameter tensors from bar model.")


def main():
    parser = argparse.ArgumentParser(description="Train EOD PricePredictor and show validation plot.")
    parser.add_argument(
        "--init-from-bar-model",
        action="store_true",
        help="Initialize PricePredictor LSTM from next-minute bar model weights before training.",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        samples = (
            db.query(TrainingSample)
            .filter(
                TrainingSample.is_used_for_training == False,
                TrainingSample.actual_price.isnot(None),
            )
            .order_by(TrainingSample.session_date, TrainingSample.id)
            .all()
        )

        if len(samples) < MIN_TRAIN_SAMPLES:
            print(f"Insufficient training samples: {len(samples)} (need at least {MIN_TRAIN_SAMPLES})")
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

        # Effective hyperparameters (support QUICK_MODE for fast experiments)
        batch_size = settings.QUICK_BATCH_SIZE if settings.QUICK_MODE else settings.BATCH_SIZE
        num_workers = settings.DATA_LOADER_WORKERS
        num_epochs = settings.QUICK_NUM_EPOCHS if settings.QUICK_MODE else settings.NUM_EPOCHS

        from torch.utils.data import DataLoader  # local import to avoid shadowing

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = None
        if test_samples:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        model = PricePredictor(**price_predictor_kwargs_from_settings())
        if args.init_from_bar_model:
            _maybe_init_from_bar_model(model)

        trainer = Trainer(model)
        save_dir = settings.MODELS_DIR / settings.MODEL_NAME

        # Train with plot saved only (Agg backend used by trainer)
        history = trainer.train(
            train_loader,
            val_loader,
            num_epochs,
            db,
            save_dir,
            test_loader=test_loader,
            plot_show=False,
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


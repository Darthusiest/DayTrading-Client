#!/usr/bin/env python3
"""
Run training (if enough data), wait for it to complete, then run a price prediction
using a chart image. Usage:

  python scripts/train_then_predict.py path/to/chart.png [--symbol MNQ1!] [--base-url http://localhost:8000]

Requires the API server to be running: uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import requests

DEFAULT_BASE = "http://localhost:8000"
API = "/api/v1"


def _resolve_image_path(path: Path) -> Optional[Path]:
    """Resolve image path; handle macOS screenshot names with narrow no-break space (U+202F)."""
    if path.is_file():
        return path
    # Try relative to cwd if path has no directory
    if not path.is_absolute() and len(path.parts) == 1:
        cwd_file = Path.cwd() / path
        if cwd_file.is_file():
            return cwd_file
        # Filename might have U+202F instead of space (e.g. "5.33.34 PM" from macOS)
        parent = Path.cwd()
        for f in parent.glob("Screenshot*.png"):
            if f.name.replace("\u202f", " ").strip() == path.name.replace("\u202f", " ").strip():
                return f
        # Fallback: single Screenshot*.png in cwd
        matches = list(parent.glob("Screenshot*.png"))
        if len(matches) == 1:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Train model then predict with a chart image.")
    parser.add_argument("image", type=Path, help="Path to chart screenshot (e.g. PNG)")
    parser.add_argument("--symbol", default="MNQ1!", help="Symbol (default: MNQ1!)")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help=f"API base URL (default: {DEFAULT_BASE})")
    parser.add_argument("--skip-train", action="store_true", help="Skip training; only run prediction (use existing model)")
    parser.add_argument("--train-timeout", type=int, default=7200, help="Max seconds to wait for training (default: 2h)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between status polls (default: 30)")
    args = parser.parse_args()

    resolved = _resolve_image_path(args.image)
    if resolved is None:
        print(f"Error: image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    args.image = resolved

    base = args.base_url.rstrip("/")

    if not args.skip_train:
        # 1) Check status and trigger training if we can
        try:
            r = requests.get(f"{base}{API}/train/status", timeout=10)
            r.raise_for_status()
            status = r.json()
        except requests.RequestException as e:
            print(f"Error: could not reach API at {base}: {e}", file=sys.stderr)
            print("Make sure the server is running: uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000", file=sys.stderr)
            sys.exit(1)

        if not status.get("can_train"):
            min_needed = status.get("min_train_samples", 10)
            print(
                f"Not enough data to train: need at least {min_needed} samples with actual_price. "
                f"unused_samples={status.get('unused_samples', 0)}, total={status.get('total_samples', 0)}.",
                file=sys.stderr,
            )
            print("Ingest Databento (POST /api/v1/collection/ingest-databento), then POST /api/v1/collection/process-training-data to build samples from bars.", file=sys.stderr)
            # Still try prediction in case they have an existing model
            if not status.get("latest_model"):
                sys.exit(1)
            print("Proceeding to prediction with existing model.", file=sys.stderr)
        else:
            # Start training
            try:
                r = requests.post(f"{base}{API}/train", timeout=10)
                r.raise_for_status()
                body = r.json()
                print(f"Training started: {body.get('training_samples')} samples.")
            except requests.RequestException as e:
                print(f"Error starting training: {e}", file=sys.stderr)
                sys.exit(1)

            # Wait until training completes (samples marked used => unused_samples == 0)
            print(f"Waiting for training to complete (polling every {args.poll_interval}s, max {args.train_timeout}s)...")
            start = time.time()
            while time.time() - start < args.train_timeout:
                time.sleep(args.poll_interval)
                try:
                    r = requests.get(f"{base}{API}/train/status", timeout=10)
                    r.raise_for_status()
                    status = r.json()
                except requests.RequestException:
                    continue
                if status.get("unused_samples", 1) == 0:
                    print("Training finished.")
                    break
                print(f"  ... unused_samples={status.get('unused_samples')} (training in progress)")
            else:
                print("Training did not finish within timeout. Proceeding to prediction with current best model.", file=sys.stderr)

    # 2) Predict with the chart image
    print(f"Running prediction for {args.image} (symbol={args.symbol})...")
    try:
        with open(args.image, "rb") as f:
            files = {"file": (args.image.name, f, "image/png")}
            data = {"symbol": args.symbol}
            r = requests.post(f"{base}{API}/predict", files=files, data=data, timeout=60)
        r.raise_for_status()
        out = r.json()
    except requests.RequestException as e:
        print(f"Prediction request failed: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None and e.response.text:
            print(e.response.text[:500], file=sys.stderr)
        sys.exit(1)

    print("\n--- Prediction result ---")
    print(f"  prediction_id:    {out.get('prediction_id')}")
    print(f"  symbol:           {out.get('symbol')}")
    print(f"  model_predicted_price: {out.get('model_predicted_price')}")
    print(f"  probability_hit:  {out.get('probability_hit')}")
    print(f"  model_confidence: {out.get('model_confidence')}")
    print(f"  learning_score:   {out.get('learning_score')}")
    if out.get("user_expected_price") is not None:
        print(f"  user_expected_price: {out.get('user_expected_price')}")
    print(f"  timestamp:        {out.get('timestamp')}")


if __name__ == "__main__":
    main()

"""Data collection services."""
from backend.services.data_collection.screenshot_capture import ScreenshotCapture
from backend.services.data_collection.collector import run_collection

__all__ = [
    "ScreenshotCapture",
    "run_collection",
]

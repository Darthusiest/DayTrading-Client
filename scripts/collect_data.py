"""Script to collect before/after snapshots."""
import sys
from pathlib import Path
from datetime import datetime
import pytz
from sqlalchemy.orm import Session

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.db import SessionLocal
from backend.database.models import Snapshot, PriceData
from backend.services.data_collection.tradingview_client import PolygonClient
from backend.services.data_collection.screenshot_capture import ScreenshotCapture
from backend.config.settings import settings

def collect_snapshots():
    """Collect before/after snapshots for all symbols."""
    db: Session = SessionLocal()
    timezone = pytz.timezone(settings.TIMEZONE)
    session_date = datetime.now(timezone).strftime("%Y-%m-%d")
    
    polygon_client = PolygonClient()
    screenshot_capture = ScreenshotCapture()
    
    try:
        for symbol in settings.SYMBOLS:
            print(f"Collecting snapshots for {symbol}...")
            
            # Determine snapshot type based on current time
            now = datetime.now(timezone)
            current_time = now.time()
            before_time = datetime.strptime(settings.BEFORE_SNAPSHOT_TIME, "%H:%M").time()
            after_time = datetime.strptime(settings.AFTER_SNAPSHOT_TIME, "%H:%M").time()
            
            snapshot_type = None
            if current_time <= before_time:
                snapshot_type = "before"
            elif current_time >= after_time:
                snapshot_type = "after"
            else:
                print(f"Current time {current_time} is between snapshots, skipping...")
                continue
            
            # Capture screenshot
            image_path = screenshot_capture.capture_chart_screenshot(
                symbol=symbol,
                snapshot_type=snapshot_type,
                session_date=session_date
            )
            
            if image_path:
                # Save snapshot to database
                snapshot = Snapshot(
                    symbol=symbol,
                    snapshot_type=snapshot_type,
                    timestamp=now,
                    image_path=str(image_path),
                    session_date=session_date
                )
                db.add(snapshot)
                db.flush()
                
                # Fetch price data
                price_data_dict = polygon_client.get_price_data(symbol, now)
                if price_data_dict:
                    price_data = PriceData(
                        snapshot_id=snapshot.id,
                        symbol=symbol,
                        timestamp=now,
                        open_price=price_data_dict.get("open", 0.0),
                        high_price=price_data_dict.get("high", 0.0),
                        low_price=price_data_dict.get("low", 0.0),
                        close_price=price_data_dict.get("close", 0.0),
                        volume=price_data_dict.get("volume", 0)
                    )
                    db.add(price_data)
                
                db.commit()
                print(f"Saved {snapshot_type} snapshot for {symbol}")
            else:
                print(f"Failed to capture screenshot for {symbol}")
    
    except Exception as e:
        print(f"Error collecting snapshots: {e}")
        db.rollback()
    finally:
        screenshot_capture.close()
        db.close()

if __name__ == "__main__":
    collect_snapshots()

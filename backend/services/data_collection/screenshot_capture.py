"""Screenshot capture service for TradingView charts."""
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import pytz
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import io
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class ScreenshotCapture:
    """Service for capturing TradingView chart screenshots."""
    
    def __init__(self):
        self.timezone = pytz.timezone(settings.TIMEZONE)
        self.driver: Optional[webdriver.Chrome] = None
    
    def _init_driver(self):
        """Initialize Selenium WebDriver."""
        if self.driver is None:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
            except Exception as e:
                logger.error(f"Failed to initialize Chrome driver: {e}")
                raise
    
    def capture_chart_screenshot(
        self,
        symbol: str,
        snapshot_type: str,
        session_date: Optional[str] = None
    ) -> Optional[Path]:
        """
        Capture a screenshot of TradingView chart.
        
        Args:
            symbol: Trading symbol (MNQ1! or MES1!)
            snapshot_type: 'before' or 'after'
            session_date: Session date in YYYY-MM-DD format
        
        Returns:
            Path to saved screenshot or None if failed
        """
        try:
            self._init_driver()
            
            if session_date is None:
                session_date = datetime.now(self.timezone).strftime("%Y-%m-%d")
            
            # Construct TradingView URL
            # Note: This assumes TradingView chart URL structure
            # You may need to adjust based on actual TradingView URL format
            chart_url = f"https://www.tradingview.com/chart/?symbol={symbol}"
            
            logger.info(f"Navigating to {chart_url}")
            self.driver.get(chart_url)
            
            wait_seconds = getattr(settings, "COLLECTION_CHART_WAIT_SECONDS", 15)
            wait = WebDriverWait(self.driver, max(30, wait_seconds))
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # Wait for chart area: try canvas first (TradingView uses canvas for chart)
            try:
                WebDriverWait(self.driver, min(wait_seconds, 20)).until(
                    EC.presence_of_element_located((By.TAG_NAME, "canvas"))
                )
            except Exception:
                pass  # Proceed; chart may use different structure or need more time
            time.sleep(min(wait_seconds, 10))  # Allow chart to fully render

            # Take screenshot
            screenshot_bytes = self.driver.get_screenshot_as_png()
            
            # Save screenshot
            filename = f"{symbol}_{snapshot_type}_{session_date}_{datetime.now().strftime('%H%M%S')}.png"
            filepath = settings.RAW_DATA_DIR / filename
            
            # Save image
            image = Image.open(io.BytesIO(screenshot_bytes))
            image.save(filepath)
            
            logger.info(f"Screenshot saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error capturing screenshot for {symbol}: {e}")
            return None
    
    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()

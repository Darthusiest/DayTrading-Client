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
    
    SIGNIN_URL = "https://www.tradingview.com/accounts/signin/"
    
    def __init__(self):
        self.timezone = pytz.timezone(settings.TIMEZONE)
        self.driver: Optional[webdriver.Chrome] = None
        self._logged_in = False
    
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
    
    def _ensure_logged_in(self) -> bool:
        """Log in to TradingView if credentials are set and not already logged in. Returns True if session is ready (logged in or no credentials)."""
        if not settings.TRADINGVIEW_USERNAME or not settings.TRADINGVIEW_PASSWORD:
            return True
        if self._logged_in:
            return True
        wait_after_submit = getattr(settings, "TRADINGVIEW_LOGIN_WAIT_SECONDS", 10)
        try:
            logger.info("Logging in to TradingView…")
            self.driver.get(self.SIGNIN_URL)
            time.sleep(3)  # Let the page and any modals/iframes render
            wait = WebDriverWait(self.driver, 25)

            def find_email_and_password(driver):
                """Find email and password inputs in current context (default content or iframe)."""
                email_selectors = [
                    "input[name='username']",
                    "input[type='email']",
                    "input[name='id_username']",
                    "input[placeholder*='mail' i]",
                    "input[placeholder*='username' i]",
                    "input[autocomplete='username']",
                ]
                pass_selectors = [
                    "input[name='password']",
                    "input[type='password']",
                    "input[name='id_password']",
                    "input[placeholder*='assword' i]",
                    "input[autocomplete='current-password']",
                ]
                email_el = None
                for sel in email_selectors:
                    try:
                        el = driver.find_element(By.CSS_SELECTOR, sel)
                        if el and el.is_displayed():
                            email_el = el
                            break
                    except Exception:
                        continue
                pass_el = None
                for sel in pass_selectors:
                    try:
                        el = driver.find_element(By.CSS_SELECTOR, sel)
                        if el and el.is_displayed():
                            pass_el = el
                            break
                    except Exception:
                        continue
                return email_el, pass_el

            # Try default content first, then switch to iframe if form is inside one
            email_el, pass_el = find_email_and_password(self.driver)
            iframe_used = None
            if (not email_el or not pass_el) and self.driver.find_elements(By.TAG_NAME, "iframe"):
                for idx, frame in enumerate(self.driver.find_elements(By.TAG_NAME, "iframe")):
                    try:
                        self.driver.switch_to.frame(frame)
                        time.sleep(1)
                        email_el, pass_el = find_email_and_password(self.driver)
                        if email_el and pass_el:
                            iframe_used = idx
                            break
                    except Exception:
                        pass
                    finally:
                        self.driver.switch_to.default_content()
                if iframe_used is not None:
                    self.driver.switch_to.frame(self.driver.find_elements(By.TAG_NAME, "iframe")[iframe_used])

            if not email_el:
                raise ValueError("Could not find TradingView username/email input (tried main page and iframes)")
            if not pass_el:
                raise ValueError("Could not find TradingView password input (tried main page and iframes)")

            email_el.clear()
            email_el.send_keys(settings.TRADINGVIEW_USERNAME)
            pass_el.clear()
            pass_el.send_keys(settings.TRADINGVIEW_PASSWORD)
            time.sleep(0.5)
            submit_selectors = [
                "button[type='submit']",
                "input[type='submit']",
                "[data-name='signin-button']",
                ".signin-button",
                "form button",
                "form [type='submit']",
            ]
            submit = None
            for sel in submit_selectors:
                try:
                    submit = self.driver.find_element(By.CSS_SELECTOR, sel)
                    if submit and submit.is_displayed():
                        break
                except Exception:
                    continue
            if not submit:
                try:
                    submit = self.driver.find_element(By.XPATH, "//button[contains(translate(., 'SIGN IN', 'sign in'), 'sign in')]")
                except Exception:
                    pass
            if not submit:
                raise ValueError("Could not find TradingView sign-in submit button")
            submit.click()
            if iframe_used is not None:
                self.driver.switch_to.default_content()
            # Wait for redirect away from sign-in; verify we're actually logged in
            logged_in_verified = False
            for _ in range(wait_after_submit):
                time.sleep(1)
                current = (self.driver.current_url or "").lower()
                if "signin" not in current or "chart" in current:
                    logged_in_verified = True
                    break
                try:
                    if self.driver.find_element(By.CSS_SELECTOR, "[data-name='header-user-menu-button'], .tv-header__user-menu, [data-name='user-menu-button']"):
                        logged_in_verified = True
                        break
                except Exception:
                    pass
            if not logged_in_verified:
                current = (self.driver.current_url or "").lower()
                logger.warning(
                    "TradingView login may have failed: still on sign-in page after %ss (URL: %s). Screenshot will be unauthenticated.",
                    wait_after_submit,
                    current[:80],
                )
                self._logged_in = False
                return False
            self._logged_in = True
            logger.info("TradingView login completed.")
            return True
        except Exception as e:
            logger.warning("TradingView login failed (screenshot will be unauthenticated): %s", e)
            return False
    
    def capture_chart_screenshot(
        self,
        symbol: str,
        snapshot_type: str,
        session_date: Optional[str] = None,
        interval_minutes: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Capture a screenshot of TradingView chart.
        
        Args:
            symbol: Trading symbol (MNQ1! or MES1!)
            snapshot_type: 'before', 'after', or 'manual'
            session_date: Session date in YYYY-MM-DD format
            interval_minutes: Chart timeframe in minutes (1, 5, 15, 60, 240, 1440). If None, uses settings.CHART_INTERVAL_MINUTES.
        
        Returns:
            Path to saved screenshot or None if failed
        """
        try:
            self._init_driver()
            self._ensure_logged_in()
            
            if session_date is None:
                session_date = datetime.now(self.timezone).strftime("%Y-%m-%d")
            
            # Construct TradingView URL with interval (timeframe in minutes: 1, 5, 15, 60, 240, 1440)
            interval = interval_minutes if interval_minutes is not None else getattr(settings, "CHART_INTERVAL_MINUTES", 15)
            chart_url = f"https://www.tradingview.com/chart/?symbol={symbol}&interval={interval}"
            
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
        self._logged_in = False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()

"""Image preprocessing for chart screenshots."""
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess chart screenshots for ML model."""
    
    def __init__(self):
        self.target_size = settings.IMAGE_SIZE
    
    def preprocess(
        self,
        image_path: Path,
        extract_chart_region: bool = True,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Preprocess a chart screenshot.
        
        Args:
            image_path: Path to input image
            extract_chart_region: Whether to extract chart region (remove UI elements)
            normalize: Whether to normalize pixel values
        
        Returns:
            Preprocessed image as numpy array or None if failed
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Extract chart region if requested
            if extract_chart_region:
                image = self._extract_chart_region(image)
            
            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)
            
            # Normalize pixel values to [0, 1]
            if normalize:
                img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def _extract_chart_region(self, image: Image.Image) -> Image.Image:
        """
        Extract chart region from screenshot, removing UI elements.
        
        This is a simplified version - you may need to adjust based on
        actual TradingView chart layout.
        """
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Try to detect chart region using edge detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely the chart)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract region with some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_cv.shape[1] - x, w + 2 * padding)
            h = min(img_cv.shape[0] - y, h + 2 * padding)
            
            cropped = img_cv[y:y+h, x:x+w]
            
            # Convert back to PIL
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cropped_rgb)
        
        # If extraction fails, return original image
        return image
    
    def augment_image(
        self,
        image: np.ndarray,
        rotation_range: float = 5.0,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """
        Apply data augmentation to image.
        
        Args:
            image: Image as numpy array (normalized to [0, 1])
            rotation_range: Maximum rotation angle in degrees
            brightness_range: Tuple of (min, max) brightness multipliers
            contrast_range: Tuple of (min, max) contrast multipliers
        
        Returns:
            Augmented image
        """
        # Convert back to PIL for augmentation
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        # Random rotation
        if rotation_range > 0:
            import random
            angle = random.uniform(-rotation_range, rotation_range)
            img_pil = img_pil.rotate(angle, fillcolor=(255, 255, 255))
        
        # Random brightness
        if brightness_range[0] != 1.0 or brightness_range[1] != 1.0:
            import random
            enhancer = ImageEnhance.Brightness(img_pil)
            factor = random.uniform(brightness_range[0], brightness_range[1])
            img_pil = enhancer.enhance(factor)
        
        # Random contrast
        if contrast_range[0] != 1.0 or contrast_range[1] != 1.0:
            import random
            enhancer = ImageEnhance.Contrast(img_pil)
            factor = random.uniform(contrast_range[0], contrast_range[1])
            img_pil = enhancer.enhance(factor)
        
        # Convert back to numpy array
        img_array = np.array(img_pil, dtype=np.float32) / 255.0
        
        return img_array
    
    def save_processed_image(
        self,
        image: np.ndarray,
        output_path: Path
    ) -> bool:
        """Save processed image to disk."""
        try:
            # Convert from normalized [0, 1] to [0, 255]
            img_uint8 = (image * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)
            img_pil.save(output_path)
            return True
        except Exception as e:
            logger.error(f"Error saving processed image to {output_path}: {e}")
            return False

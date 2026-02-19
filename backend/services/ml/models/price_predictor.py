"""ML model architecture for price prediction."""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PricePredictor(nn.Module):
    """
    Hybrid CNN + Time Series model for price prediction from chart screenshots.
    
    Architecture:
    1. CNN encoder (ResNet/EfficientNet) for visual feature extraction
    2. LSTM/Transformer layers for temporal patterns
    3. Regression head for price prediction
    4. Classification head for probability estimation
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        cnn_backbone: str = "resnet18",
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        num_features: int = 0  # Additional non-image features
    ):
        super(PricePredictor, self).__init__()
        
        self.image_size = image_size
        self.cnn_backbone = cnn_backbone
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_features = num_features
        
        # CNN Encoder
        if cnn_backbone == "resnet18":
            cnn = models.resnet18(pretrained=True)
            # Remove final fully connected layer
            self.cnn_features = nn.Sequential(*list(cnn.children())[:-1])
            cnn_output_size = 512
        elif cnn_backbone == "efficientnet_b0":
            cnn = models.efficientnet_b0(pretrained=True)
            self.cnn_features = nn.Sequential(*list(cnn.children())[:-1])
            cnn_output_size = 1280
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
        
        # Freeze early layers (optional - can be fine-tuned)
        for param in list(self.cnn_features.parameters())[:-10]:
            param.requires_grad = False
        
        # Flatten CNN output
        self.cnn_flatten = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature combination
        combined_input_size = cnn_output_size + num_features
        
        # LSTM for temporal patterns (if using sequence data)
        # For single image prediction, we can use a simpler approach
        self.lstm = nn.LSTM(
            input_size=combined_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Alternative: Use transformer or simple MLP if not using sequences
        self.mlp = nn.Sequential(
            nn.Linear(combined_input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Regression head for price prediction
        self.price_head = nn.Sequential(
            nn.Linear(lstm_hidden_size + 128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single price output
        )
        
        # Classification head for probability estimation
        self.probability_head = nn.Sequential(
            nn.Linear(lstm_hidden_size + 128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor = None,
        use_lstm: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Batch of images [B, C, H, W]
            features: Optional additional features [B, num_features]
            use_lstm: Whether to use LSTM (requires sequence data)
        
        Returns:
            Dictionary with 'predicted_price' and 'probability'
        """
        batch_size = images.size(0)
        
        # Extract CNN features
        cnn_out = self.cnn_features(images)  # [B, C, H, W]
        cnn_out = self.cnn_flatten(cnn_out)  # [B, C, 1, 1]
        cnn_out = cnn_out.view(batch_size, -1)  # [B, C]
        
        # Combine with additional features if provided
        if features is not None:
            combined = torch.cat([cnn_out, features], dim=1)
        else:
            combined = cnn_out
        
        # Process through MLP
        mlp_out = self.mlp(combined)  # [B, 128]
        
        # Use LSTM if sequence data available
        if use_lstm and features is not None:
            # Reshape for LSTM: [B, seq_len, features]
            # For now, treat single sample as sequence of length 1
            lstm_input = combined.unsqueeze(1)  # [B, 1, features]
            lstm_out, _ = self.lstm(lstm_input)  # [B, 1, hidden_size]
            lstm_out = lstm_out.squeeze(1)  # [B, hidden_size]
            
            # Combine LSTM and MLP outputs
            combined_features = torch.cat([lstm_out, mlp_out], dim=1)
        else:
            combined_features = torch.cat([cnn_out, mlp_out], dim=1)
        
        # Predict price
        predicted_price = self.price_head(combined_features)
        
        # Predict probability
        probability = self.probability_head(combined_features)
        
        return {
            "predicted_price": predicted_price.squeeze(-1),
            "probability": probability.squeeze(-1)
        }
    
    def predict_price(
        self,
        images: torch.Tensor,
        features: torch.Tensor = None,
        expected_price: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        Predict price and probability of hitting expected price.
        
        Args:
            images: Batch of images
            features: Optional additional features
            expected_price: Optional expected price to calculate probability for
        
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, features)
            
            predicted_price = outputs["predicted_price"]
            base_probability = outputs["probability"]
            
            # If expected price provided, calculate probability of hitting it
            if expected_price is not None:
                # Calculate probability based on distance from predicted price
                price_diff = torch.abs(predicted_price - expected_price)
                # Normalize difference (assuming price range)
                # This is a simplified approach - can be improved
                normalized_diff = price_diff / (predicted_price + 1e-6)
                hit_probability = base_probability * (1.0 - torch.clamp(normalized_diff, 0, 1))
            else:
                hit_probability = base_probability
            
            return {
                "predicted_price": predicted_price.cpu().numpy(),
                "probability": hit_probability.cpu().numpy(),
                "base_confidence": base_probability.cpu().numpy()
            }

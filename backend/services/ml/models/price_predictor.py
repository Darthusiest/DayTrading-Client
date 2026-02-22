"""ML model architecture for price prediction."""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class PricePredictor(nn.Module):
    """
    Hybrid CNN + Time Series model for price prediction from chart screenshots.

    Architecture (configurable via settings):
    1. CNN encoder: ResNet18 or EfficientNet-B0; trainable param groups configurable (0 = all trainable).
    2. MLP: hidden sizes from mlp_hiddens (e.g. [256, 256, 128]).
    3. LSTM: num_lstm_layers and lstm_hidden_size configurable.
    4. Regression head and classification head: input = lstm_hidden_size + mlp_output_size.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        cnn_backbone: str = "resnet18",
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        num_features: int = 0,  # Additional non-image features
        mlp_hiddens: Optional[List[int]] = None,
        cnn_trainable_param_groups: int = 10,
    ):
        super(PricePredictor, self).__init__()
        if mlp_hiddens is None:
            mlp_hiddens = [256, 128]

        self.image_size = image_size
        self.cnn_backbone = cnn_backbone
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_features = num_features

        # CNN Encoder
        if cnn_backbone == "resnet18":
            cnn = models.resnet18(pretrained=True)
            self.cnn_features = nn.Sequential(*list(cnn.children())[:-1])
            cnn_output_size = 512
        elif cnn_backbone == "efficientnet_b0":
            cnn = models.efficientnet_b0(pretrained=True)
            self.cnn_features = nn.Sequential(*list(cnn.children())[:-1])
            cnn_output_size = 1280
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")

        # Freeze CNN: 0 = train all; else train only last N param tensors
        cnn_params = list(self.cnn_features.parameters())
        if cnn_trainable_param_groups > 0 and len(cnn_params) > cnn_trainable_param_groups:
            for param in cnn_params[:-cnn_trainable_param_groups]:
                param.requires_grad = False

        # Flatten CNN output
        self.cnn_flatten = nn.AdaptiveAvgPool2d((1, 1))
        combined_input_size = cnn_output_size + num_features

        # LSTM
        self.lstm = nn.LSTM(
            input_size=combined_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # MLP from list of hidden sizes
        mlp_layers: List[nn.Module] = []
        prev_size = combined_input_size
        for h in mlp_hiddens:
            mlp_layers.extend([
                nn.Linear(prev_size, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp_output_size = mlp_hiddens[-1]
        head_input_size = lstm_hidden_size + self.mlp_output_size

        # Regression head for price prediction
        self.price_head = nn.Sequential(
            nn.Linear(head_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Classification head for probability estimation
        self.probability_head = nn.Sequential(
            nn.Linear(head_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
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


def price_predictor_kwargs_from_settings() -> Dict[str, Any]:
    """Build PricePredictor constructor kwargs from backend settings (for training and inference)."""
    from backend.config.settings import settings
    return {
        "num_features": settings.NUM_FEATURES,
        "num_lstm_layers": settings.NUM_LSTM_LAYERS,
        "lstm_hidden_size": settings.LSTM_HIDDEN_SIZE,
        "mlp_hiddens": settings.MLP_HIDDENS,
        "cnn_trainable_param_groups": settings.CNN_TRAINABLE_PARAM_GROUPS,
    }

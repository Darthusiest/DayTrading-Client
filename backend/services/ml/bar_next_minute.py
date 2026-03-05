"""Next-minute bar dataset and LSTM model for 1m next-candle prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from backend.config.settings import settings


class NextMinuteBarDataset(Dataset):
    """
    Simple tensor-backed dataset for next-minute prediction.

    Expects prebuilt tensors:
      - sequences: [N, T, F] float32 (T = lookback window length, F = features per bar)
      - targets:   [N] float32 (next-bar target value, e.g. close price or return)
    """

    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor):
        if sequences.ndim != 3:
            raise ValueError(f"sequences must be [N, T, F], got shape={sequences.shape}")
        if targets.ndim != 1:
            raise ValueError(f"targets must be [N], got shape={targets.shape}")
        if sequences.size(0) != targets.size(0):
            raise ValueError(
                f"Batch size mismatch: sequences N={sequences.size(0)} targets N={targets.size(0)}"
            )
        self.sequences = sequences
        self.targets = targets

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "sequence": self.sequences[idx],
            "target": self.targets[idx],
        }


@dataclass
class NextMinuteModelConfig:
    """Config for the next-minute LSTM model."""

    input_size: int  # number of per-bar features (e.g. OHLCV -> 5)
    hidden_size: int = settings.LSTM_HIDDEN_SIZE
    num_layers: int = settings.NUM_LSTM_LAYERS
    dropout: float = 0.1


class NextMinuteBarLSTM(nn.Module):
    """
    LSTM model for 1m next-candle prediction.

    Given a window of T bars with F features each, predicts a scalar target
    (typically the next bar's close price or return).
    """

    def __init__(self, config: NextMinuteModelConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F] sequence of bars

        Returns:
            predictions: [B] scalar target per sequence
        """
        out, _ = self.lstm(x)  # [B, T, H]
        last = out[:, -1, :]   # [B, H]
        pred = self.head(last)  # [B, 1]
        return pred.squeeze(-1)


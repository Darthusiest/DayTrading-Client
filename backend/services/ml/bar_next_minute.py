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
    Tensor-backed dataset for next-minute prediction with multi-horizon targets.

    Expects prebuilt tensors:
      - sequences:         [N, T, F] float32 (T = lookback window length, F = features per bar)
      - targets_price:     [N]       float32 (next-bar close price)
      - targets_dir5:      [N]       int64   (direction next 5m: 0=down,1=sideways,2=up)
      - targets_vol10:     [N]       float32 (volatility next 10m)
      - targets_breakout:  [N]       float32 (0/1 breakout label for next 10m)
    """

    def __init__(
        self,
        sequences: torch.Tensor,
        targets_price: torch.Tensor,
        targets_dir5: torch.Tensor,
        targets_vol10: torch.Tensor,
        targets_breakout: torch.Tensor,
    ):
        if sequences.ndim != 3:
            raise ValueError(f"sequences must be [N, T, F], got shape={sequences.shape}")
        n = sequences.size(0)
        for name, t in [
            ("targets_price", targets_price),
            ("targets_dir5", targets_dir5),
            ("targets_vol10", targets_vol10),
            ("targets_breakout", targets_breakout),
        ]:
            if t.ndim != 1:
                raise ValueError(f"{name} must be [N], got shape={t.shape}")
            if t.size(0) != n:
                raise ValueError(f"{name} batch size mismatch: sequences N={n} {name} N={t.size(0)}")

        self.sequences = sequences
        self.targets_price = targets_price
        self.targets_dir5 = targets_dir5
        self.targets_vol10 = targets_vol10
        self.targets_breakout = targets_breakout

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "sequence": self.sequences[idx],
            "target_price": self.targets_price[idx],
            "target_dir5": self.targets_dir5[idx],
            "target_vol10": self.targets_vol10[idx],
            "target_breakout": self.targets_breakout[idx],
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

        # Shared trunk head (hidden representation at last time step)
        self.trunk = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # 1) Next-bar price regression
        self.price_head = nn.Linear(config.hidden_size, 1)

        # 2) Direction next 5m (3-way classification: 0=down,1=sideways,2=up)
        self.dir5_head = nn.Linear(config.hidden_size, 3)

        # 3) Volatility next 10m (regression)
        self.vol10_head = nn.Linear(config.hidden_size, 1)

        # 4) Breakout probability next 10m (binary classification)
        self.breakout_head = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, F] sequence of bars

        Returns:
            Dict of predictions:
              - price: [B]        next-bar close price
              - dir5_logits: [B,3] logits for direction next 5m
              - vol10: [B]        volatility next 10m
              - breakout: [B]     probability of breakout next 10m
        """
        out, _ = self.lstm(x)  # [B, T, H]
        last = out[:, -1, :]   # [B, H]
        h = self.trunk(last)   # [B, H]

        price = self.price_head(h).squeeze(-1)
        dir5_logits = self.dir5_head(h)
        vol10 = self.vol10_head(h).squeeze(-1)
        breakout = self.breakout_head(h).squeeze(-1)

        return {
            "price": price,
            "dir5_logits": dir5_logits,
            "vol10": vol10,
            "breakout": breakout,
        }


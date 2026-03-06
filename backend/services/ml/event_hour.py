"""Event-driven 1-hour direction dataset and LSTM classifier.

This module supports training models that only make predictions at event times
(e.g., PDH/PDL sweep, ORB, impulse candle). Targets are binary labels such as
continuation or reversal over the next 60 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset

from backend.config.settings import settings


class EventHourDataset(Dataset):
    """
    Tensor-backed dataset for event-driven 1-hour direction tasks.

    Expects:
      - sequences: [N, T, F] float32
      - targets:   [N]       float32 (0/1)
      - event_type:[N]       int64   (optional, for analysis)
      - event_dir: [N]       int64   (+1/-1 encoded as 1/0 or stored separately)
      - session_id:[N]       int64   (optional, for splits)
      - symbol_id:[N]        int64   (optional, for analysis; e.g. 0=MNQ, 1=MES)
    """

    def __init__(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        event_type: torch.Tensor | None = None,
        event_dir: torch.Tensor | None = None,
        session_id: torch.Tensor | None = None,
        symbol_id: torch.Tensor | None = None,
    ) -> None:
        if sequences.ndim != 3:
            raise ValueError(f"sequences must be [N, T, F], got shape={sequences.shape}")
        if targets.ndim != 1:
            raise ValueError(f"targets must be [N], got shape={targets.shape}")
        n = sequences.size(0)
        if targets.size(0) != n:
            raise ValueError(f"targets batch size mismatch: sequences N={n} targets N={targets.size(0)}")
        for name, t in [
            ("event_type", event_type),
            ("event_dir", event_dir),
            ("session_id", session_id),
            ("symbol_id", symbol_id),
        ]:
            if t is None:
                continue
            if t.ndim != 1:
                raise ValueError(f"{name} must be [N], got shape={t.shape}")
            if t.size(0) != n:
                raise ValueError(f"{name} batch size mismatch: sequences N={n} {name} N={t.size(0)}")

        self.sequences = sequences
        self.targets = targets
        self.event_type = event_type
        self.event_dir = event_dir
        self.session_id = session_id
        self.symbol_id = symbol_id

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out = {
            "sequence": self.sequences[idx],
            "target": self.targets[idx],
        }
        if self.event_type is not None:
            out["event_type"] = self.event_type[idx]
        if self.event_dir is not None:
            out["event_dir"] = self.event_dir[idx]
        if self.session_id is not None:
            out["session_id"] = self.session_id[idx]
        if self.symbol_id is not None:
            out["symbol_id"] = self.symbol_id[idx]
        return out


@dataclass
class EventHourModelConfig:
    input_size: int
    hidden_size: int = settings.LSTM_HIDDEN_SIZE
    num_layers: int = settings.NUM_LSTM_LAYERS
    dropout: float = float(getattr(settings, "EVENT_DROPOUT", 0.1))


class EventHourLSTM(nn.Module):
    """Binary LSTM classifier (logits) for event-driven 1-hour targets."""

    def __init__(self, config: EventHourModelConfig) -> None:
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.trunk = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # [B, T, H]
        last = out[:, -1, :]  # [B, H]
        h = self.trunk(last)
        logits = self.head(h).squeeze(-1)  # [B]
        return logits


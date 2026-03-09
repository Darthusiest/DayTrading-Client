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
        sample_weight: torch.Tensor | None = None,
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
            ("sample_weight", sample_weight),
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
        self.sample_weight = sample_weight

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
        if self.sample_weight is not None:
            out["sample_weight"] = self.sample_weight[idx]
        return out


@dataclass
class EventHourModelConfig:
    input_size: int
    hidden_size: int = int(getattr(settings, "EVENT_LSTM_HIDDEN_SIZE", settings.LSTM_HIDDEN_SIZE))
    num_layers: int = int(getattr(settings, "EVENT_NUM_LSTM_LAYERS", settings.NUM_LSTM_LAYERS))
    dropout: float = float(getattr(settings, "EVENT_DROPOUT", 0.1))
    num_event_types: int = 11  # 0 padding + 1-10 event types
    event_embed_dim: int = 16
    use_attention: bool = True


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
        self.event_embed = nn.Embedding(config.num_event_types, config.event_embed_dim, padding_idx=0)
        trunk_in = config.hidden_size + config.event_embed_dim
        self.attn = nn.Linear(config.hidden_size, 1) if config.use_attention else None
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, x: torch.Tensor, event_type: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(x)  # [B, T, H]
        if self.attn is not None:
            scores = self.attn(out).squeeze(-1)  # [B, T]
            weights = torch.softmax(scores, dim=1)
            context = (out * weights.unsqueeze(-1)).sum(1)  # [B, H]
        else:
            context = out[:, -1, :]  # [B, H]
        if event_type is not None:
            et = event_type.clamp(0, self.config.num_event_types - 1)
            emb = self.event_embed(et)  # [B, E]
            context = torch.cat([context, emb], dim=1)  # [B, H+E]
        else:
            emb = torch.zeros(x.size(0), self.config.event_embed_dim, device=x.device, dtype=x.dtype)
            context = torch.cat([context, emb], dim=1)
        h = self.trunk(context)
        logits = self.head(h).squeeze(-1)  # [B]
        return logits


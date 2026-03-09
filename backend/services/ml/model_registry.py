"""Simple model and artifact registry with process-local caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import torch

from backend.config.settings import settings

_MODEL_CACHE: Dict[str, Any] = {}
_JSON_CACHE: Dict[str, Dict[str, Any]] = {}
_OBJECT_CACHE: Dict[str, Any] = {}


def _cache_key(name: str, suffix: str = "") -> str:
    return f"{name}:{suffix}" if suffix else name


def get_or_create_object(key: str, builder: Callable[[], Any]) -> Any:
    """Cache arbitrary Python objects (for route-level singletons)."""
    if key not in _OBJECT_CACHE:
        _OBJECT_CACHE[key] = builder()
    return _OBJECT_CACHE[key]


def load_json_artifact(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load JSON artifact once and cache it by full path."""
    key = str(path.resolve())
    if key in _JSON_CACHE:
        return _JSON_CACHE[key]
    if not path.is_file():
        out = default or {}
        _JSON_CACHE[key] = out
        return out
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception:
        data = default or {}
    _JSON_CACHE[key] = data
    return data


def load_torch_model(
    model_name: str,
    model_builder: Callable[[], torch.nn.Module],
    version_key: str = "",
    strict: bool = True,
) -> torch.nn.Module:
    """Load a model checkpoint from MODELS_DIR and cache the model instance."""
    key = _cache_key(model_name, version_key)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    ckpt_path = settings.MODELS_DIR / f"{model_name}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

    model = model_builder()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=strict)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


"""Security dependencies for API routes."""

from __future__ import annotations

from fastapi import Header, HTTPException

from backend.config.settings import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require API key when API_KEY is configured."""
    if not settings.REQUIRE_API_KEY:
        return
    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


#!/usr/bin/env python3
"""
Run event-hour backtest with the current go-forward threshold set.

Defaults:
  - min-confidence=0.53
  - min-action-margin=0.02
  - min-continuation-prob=0.50
  - min-reversal-prob=0.51

Any extra CLI args are forwarded to scripts/backtest_event_hour.py so you can
still pass options like --prefix, --start-date, --end-date, etc.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


DEFAULT_ARGS = [
    "--min-confidence",
    "0.53",
    "--min-action-margin",
    "0.02",
    "--min-continuation-prob",
    "0.50",
    "--min-reversal-prob",
    "0.51",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    # Use module execution so repo-root imports like `backend.*` resolve.
    cmd = [sys.executable, "-m", "scripts.backtest_event_hour", *DEFAULT_ARGS, *sys.argv[1:]]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())

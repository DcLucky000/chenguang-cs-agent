"""Local demo paths. Runs stay on this machine; there is no shared lock service."""

from __future__ import annotations

import os
import threading
from pathlib import Path

WRITE_LOCK = threading.Lock()

ROOT = Path(__file__).resolve().parent.parent


def runs_dir() -> Path:
    raw = os.environ.get("CHENGUANG_RUNS_DIR", "").strip()
    if raw:
        return Path(raw)
    return ROOT / "runs"

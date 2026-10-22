"""Temp runs directory so tests do not share ticket counters."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def isolated_runs():
    tmp = tempfile.TemporaryDirectory()
    old = os.environ.get("CHENGUANG_RUNS_DIR")
    os.environ["CHENGUANG_RUNS_DIR"] = tmp.name
    try:
        yield Path(tmp.name)
    finally:
        if old is None:
            os.environ.pop("CHENGUANG_RUNS_DIR", None)
        else:
            os.environ["CHENGUANG_RUNS_DIR"] = old
        tmp.cleanup()

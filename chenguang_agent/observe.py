"""One JSON object per turn, appended to runs/observability.jsonl."""

from __future__ import annotations

import json

from chenguang_agent.paths import WRITE_LOCK, runs_dir


def append_observability(record: dict) -> bool:
    path = runs_dir() / "observability.jsonl"
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    try:
        with WRITE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:
        return False
    return True

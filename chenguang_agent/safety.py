"""Demo input length cap and per-session rate limit. In-process memory only."""

from __future__ import annotations

import os
import threading
import time

DEFAULT_MAX_INPUT_CHARS = 1000
DEFAULT_RATE_LIMIT = 30
DEFAULT_RATE_WINDOW_S = 60.0

INPUT_TOO_LONG = "input_too_long"
RATE_LIMITED = "rate_limited"


def max_input_chars() -> int:
    raw = os.environ.get("CHENGUANG_MAX_INPUT_CHARS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MAX_INPUT_CHARS


def rate_limit_per_minute() -> int:
    raw = os.environ.get("CHENGUANG_RATE_LIMIT", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_RATE_LIMIT


class RateLimiter:
    """Count requests per key inside this process. Not shared across machines."""

    def __init__(self, limit: int | None = None, window_s: float = DEFAULT_RATE_WINDOW_S) -> None:
        self.limit = rate_limit_per_minute() if limit is None else limit
        self.window_s = window_s
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if not key:
            return True
        now = time.monotonic()
        with self._lock:
            hits = [stamp for stamp in self._hits.get(key, []) if now - stamp < self.window_s]
            if len(hits) >= self.limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def rejection(user_text: str, *, rate_key: str | None, limiter: RateLimiter) -> tuple[str, str] | None:
    """Reject over-long text before it consumes a rate-limit slot."""
    limit = max_input_chars()
    if len(user_text) > limit:
        return (
            INPUT_TOO_LONG,
            f"这句话超过了 {limit} 字上限，我没有检索，也没有调用工具。请缩短后再问。",
        )
    if rate_key and not limiter.allow(rate_key):
        return (
            RATE_LIMITED,
            f"这个会话一分钟内的请求已经达到 {limiter.limit} 次，我先不处理这一轮。请稍后再试。",
        )
    return None

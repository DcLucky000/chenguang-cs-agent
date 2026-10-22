"""Small string helpers shared by the router, tools, and guardrails."""

from __future__ import annotations

import re


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def matched_keywords(text: str, keywords: list[str]) -> list[str]:
    """Longer keywords win. A hit that sits inside an already matched keyword is dropped."""
    hits: list[str] = []
    for keyword in sorted({k for k in keywords if k}, key=len, reverse=True):
        if keyword in text and not any(keyword in previous for previous in hits):
            hits.append(keyword)
    return hits

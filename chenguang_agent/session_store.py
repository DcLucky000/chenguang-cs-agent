"""Append-only session JSONL. Same turn_id is not rewritten."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from chenguang_agent.paths import WRITE_LOCK, runs_dir

_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def session_file(session_id: str) -> Path:
    if _SAFE_ID.fullmatch(session_id):
        name = session_id
    else:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8]
        folded = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id).strip("._")[:40] or "session"
        name = f"{folded}-{digest}"
    return runs_dir() / "sessions" / f"{name}.jsonl"


class SessionStore:
    def append(self, session_id: str, record: dict) -> bool:
        """Append one turn. Return False when this turn_id is already on disk."""
        if not session_id:
            return False
        turn_id = str(record.get("turn_id") or "")
        path = session_file(session_id)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with WRITE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            if turn_id and path.is_file():
                for existing in _read_unlocked(path):
                    if str(existing.get("turn_id") or "") == turn_id:
                        return False
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return True

    def read(self, session_id: str) -> list[dict]:
        path = session_file(session_id)
        if not path.is_file():
            return []
        with WRITE_LOCK:
            return _read_unlocked(path)

    def memory(self, session_id: str, limit: int = 6) -> list[dict]:
        """Compact prior turns for the planner. The current turn is not included."""

        if not session_id:
            return []
        rows = self.read(session_id)[-limit:]
        memory: list[dict] = []
        for row in rows:
            skus: list[str] = []
            tools: list[str] = []
            for call in row.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "")
                if name:
                    tools.append(name)
                if name != "lookup_product":
                    continue
                for match in (call.get("output") or {}).get("matches") or []:
                    if isinstance(match, dict) and match.get("sku_id"):
                        sku = str(match["sku_id"])
                        if sku not in skus:
                            skus.append(sku)
            reply = str(row.get("reply") or "")
            memory.append(
                {
                    "user_text": row.get("user_text") or "",
                    "intent": row.get("intent") or "",
                    "reply": reply[:180],
                    "tools": tools,
                    "skus": skus,
                }
            )
        return memory


def _read_unlocked(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows

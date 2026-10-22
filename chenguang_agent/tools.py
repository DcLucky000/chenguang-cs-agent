"""The only three tools the agent is allowed to call."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone

from chenguang_agent.kb import KnowledgeBase
from chenguang_agent.paths import WRITE_LOCK, runs_dir
from chenguang_agent.schemas import TOOL_SCHEMAS, tool_error, validate_tool_arguments
from chenguang_agent.textutil import matched_keywords, squash

# Longer aliases first inside each SKU. Matching is substring, not medical inference.
PRODUCT_ALIASES: list[tuple[str, list[str]]] = [
    ("SKU01", ["氨基酸洁面", "洁面乳", "洁面"]),
    ("SKU02", ["保湿修护精华水", "精华水", "保湿水"]),
    ("SKU03", ["烟酰胺亮肤", "亮肤精华", "烟酰胺精华", "烟酰胺"]),
    ("SKU04", ["神经酰胺面霜", "神经酰胺", "面霜"]),
    ("SKU05", ["防晒乳", "防晒"]),
    ("SKU06", ["泥膜", "面膜"]),
    ("SKU07", ["眼部精华", "眼霜", "眼部"]),
    ("SKU08", ["旅行装", "三件套"]),
    ("SKU09", ["卸妆膏", "卸妆"]),
    ("SKU10", ["身体乳"]),
]

DEFAULT_TOOL_TIMEOUT_S = 2.0


class Toolbelt:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.tickets: list[dict] = []
        self._ctx = threading.local()

    def bind_context(self, **kwargs: str) -> None:
        """Request metadata for the ticket log. Not part of the tool schema."""
        self._ctx.data = {key: str(value or "") for key, value in kwargs.items()}

    def matches_product(self, query: str) -> bool:
        return bool(self._product_ids(query))

    def call(self, name: str, arguments: dict, *, timeout_s: float = DEFAULT_TOOL_TIMEOUT_S) -> dict:
        """Validate, run, and always return a dict. Callers do not see tracebacks."""
        schema = TOOL_SCHEMAS.get(name)
        if schema is None:
            return tool_error(name, "unknown_tool", "未知工具")
        problem = validate_tool_arguments(schema, arguments)
        if problem:
            return tool_error(name, "invalid_arguments", problem)
        fn = getattr(self, name)
        captured = dict(getattr(self._ctx, "data", {}) or {})
        box: dict = {}

        def target() -> None:
            self._ctx.data = captured
            try:
                box["result"] = fn(**arguments)
            except Exception as exc:  # noqa: BLE001 — demo boundary: structured error, no traceback
                box["error"] = exc

        if timeout_s <= 0:
            target()
        else:
            worker = threading.Thread(target=target, daemon=True)
            worker.start()
            worker.join(timeout_s)
            if worker.is_alive():
                return tool_error(name, "timeout", f"超过 {timeout_s:g} 秒仍未返回")
        if "error" in box:
            return tool_error(name, "tool_failure", type(box["error"]).__name__)
        result = box.get("result")
        if not isinstance(result, dict):
            return tool_error(name, "tool_failure", "工具没有返回对象")
        return result

    def lookup_product(self, query: str) -> dict:
        if not isinstance(query, str) or not query.strip():
            return tool_error("lookup_product", "invalid_arguments", "query 必须是非空字符串")
        ids = self._product_ids(query)
        by_id = {row["sku_id"]: row for row in self.kb.products}
        matches = [_product_payload(by_id[sku]) for sku in ids if sku in by_id]
        return {"found": bool(matches), "query": query, "matches": matches}

    def lookup_faq(self, question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return tool_error("lookup_faq", "invalid_arguments", "question 必须是非空字符串")
        compact = squash(question)
        ranked: list[tuple[int, list[str], dict[str, str]]] = []
        for row in self.kb.faqs:
            terms = self._terms_for(row)
            hits = matched_keywords(compact, terms)
            score = _score(
                hits,
                restrictive=row["是否允许Bot直接答"] == "否",
                efficacy=any(hit in self.kb.efficacy_keywords for hit in hits),
            )
            if score:
                ranked.append((score, hits, row))
        ranked.sort(key=lambda item: (-item[0], item[2]["q_id"]))
        if not ranked:
            return _empty_faq(question)
        score, hits, row = ranked[0]
        alternatives = [
            f"{other['q_id']} {other['标准问']}（{other_score}分）"
            for other_score, _, other in ranked[1:3]
        ]
        return {
            "found": True,
            "q_id": row["q_id"],
            "question": row["标准问"],
            "answer": row["标准答"],
            "intent": row["意图"],
            "bot_may_answer": row["是否允许Bot直接答"] == "是",
            "matched": hits,
            "score": score,
            "alternatives": alternatives,
        }

    def escalate_to_human(self, reason: str) -> dict:
        if not isinstance(reason, str) or not reason.strip():
            return tool_error("escalate_to_human", "invalid_arguments", "reason 必须是非空字符串")
        ctx = dict(getattr(self._ctx, "data", {}) or {})
        seq = _next_ticket_seq()
        ticket_id = f"CG-{seq:04d}"
        reason = reason.strip()
        ticket = {
            "escalated": True,
            "ticket_id": ticket_id,
            "reason": reason,
            "script": self.kb.escalation_script,
            "queue": "人工客服",
        }
        record = {
            "ticket_id": ticket_id,
            "reason": reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "summary": _summary(ctx.get("user_text", "")),
            "session_id": ctx.get("session_id", ""),
            "turn_id": ctx.get("turn_id", ""),
            "status": "open",
            "queue": "人工客服",
        }
        ticket["persisted"] = _append_ticket(record)
        self.tickets.append(ticket)
        return ticket

    def _terms_for(self, row: dict[str, str]) -> list[str]:
        blob = row["标准问"] + row["相似问"]
        terms = [term for term in self.kb.vocabulary if term in blob]
        terms.extend(self.kb.faq_aliases.get(row["q_id"], []))
        return terms

    def _product_ids(self, query: str) -> list[str]:
        compact = squash(query)
        found: list[str] = []
        for match in re.finditer(r"SKU\s*0*(\d{1,2})", compact, flags=re.IGNORECASE):
            found.append(f"SKU{int(match.group(1)):02d}")
        folded = compact.lower()
        for sku, aliases in PRODUCT_ALIASES:
            if any(alias.lower() in folded for alias in aliases):
                found.append(sku)
        unique: list[str] = []
        for sku in found:
            if sku not in unique:
                unique.append(sku)
        return unique


def _empty_faq(_question: str) -> dict:
    return {
        "found": False,
        "q_id": "",
        "question": "",
        "answer": "",
        "intent": "",
        "bot_may_answer": False,
        "matched": [],
        "score": 0,
        "alternatives": [],
    }


def _score(hits: list[str], *, restrictive: bool, efficacy: bool) -> int:
    if not hits:
        return 0
    score = sum(min(len(hit), 8) for hit in hits)
    if restrictive:
        # A matched "do not answer directly" row beats a longer but permissive row.
        score += 3
    if efficacy:
        # Efficacy-policy words should beat a longer usage FAQ (e.g. 烟酰胺刺激 vs 美白承诺).
        score += 3
    return score


def _product_payload(row: dict[str, str]) -> dict[str, str]:
    return {
        "sku_id": row["sku_id"],
        "name": row["名称"],
        "spec": row["规格"],
        "price": row["价格"],
        "selling_points": row["卖点"],
        "audience": row["适用人群"],
        "caution": row["禁忌注意"],
        "faq_points": row["常见问答要点"],
        "stock": row["库存状态"],
    }


def _summary(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _next_ticket_seq() -> int:
    path = runs_dir() / "ticket_seq.txt"
    with WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        current = 0
        if path.is_file():
            raw = path.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                current = int(raw)
        current += 1
        path.write_text(str(current), encoding="utf-8")
        return current


def _append_ticket(record: dict) -> bool:
    path = runs_dir() / "tickets.jsonl"
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    try:
        with WRITE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:
        return False
    return True

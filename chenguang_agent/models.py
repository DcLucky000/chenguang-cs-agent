"""Shared turn, trace, and observation types for the three roles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from chenguang_agent.guardrails import InputScreen


@dataclass
class ToolCall:
    name: str
    arguments: dict
    output: dict
    latency_ms: float = 0.0


@dataclass
class TraceEvent:
    role: str
    action: str
    detail: str


@dataclass
class Observations:
    """What the specialist has gathered for the current turn."""

    user_text: str
    intent: str
    signals: list[str]
    screen: InputScreen
    faq: dict | None = None
    products: dict | None = None
    escalation: dict | None = None
    tracking_only: bool = False
    clarify_only: bool = False
    lookup_product: bool = False
    calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Turn:
    user_text: str
    intent: str
    signals: list[str]
    reply: str
    escalated: bool
    tool_calls: list[ToolCall] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    output_violations: list[str] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)
    used_llm: bool = False
    trace: list[TraceEvent] = field(default_factory=list)
    send_backs: int = 0
    turn_id: str = ""
    session_id: str = ""
    kb_version: str = ""
    llm_fallback: str = ""
    planner: str = ""
    created_at: str = ""

    @property
    def escalate_reason(self) -> str:
        for call in self.tool_calls:
            if call.name == "escalate_to_human" and call.output.get("reason"):
                return str(call.output["reason"])
        return ""

    @property
    def ticket_id(self) -> str:
        for call in self.tool_calls:
            if call.name == "escalate_to_human" and call.output.get("ticket_id"):
                return str(call.output["ticket_id"])
        return ""

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "user_text": self.user_text,
            "intent": self.intent,
            "signals": self.signals,
            "reply": self.reply,
            "escalated": self.escalated,
            "escalate_reason": self.escalate_reason,
            "ticket_id": self.ticket_id,
            "escalate": {
                "escalated": self.escalated,
                "reason": self.escalate_reason,
                "ticket_id": self.ticket_id,
            },
            "notes": self.notes,
            "output_violations": self.output_violations,
            "flags": self.flags,
            "used_llm": self.used_llm,
            "llm_fallback": self.llm_fallback,
            "planner": self.planner,
            "kb_version": self.kb_version,
            "send_backs": self.send_backs,
            "trace": [
                {"role": event.role, "action": event.action, "detail": event.detail} for event in self.trace
            ],
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "output": call.output,
                    "latency_ms": round(call.latency_ms, 3),
                }
                for call in self.tool_calls
            ],
            "tools": [
                {"name": call.name, "latency_ms": round(call.latency_ms, 3)} for call in self.tool_calls
            ],
        }

    def observation(self) -> dict:
        return {
            "ts": self.created_at or datetime.now(timezone.utc).isoformat(),
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "intent": self.intent,
            "send_backs": self.send_backs,
            "tools": [
                {"name": call.name, "latency_ms": round(call.latency_ms, 3)} for call in self.tool_calls
            ],
            "escalate_reason": self.escalate_reason,
            "kb_version": self.kb_version,
            "escalated": self.escalated,
        }

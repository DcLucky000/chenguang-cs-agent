"""Customer-service entrypoint. An agent loop asks a planner for each next step."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from chenguang_agent.coordinator import Coordinator
from chenguang_agent.guardrails import Guardrails
from chenguang_agent.kb import KnowledgeBase, load_kb
from chenguang_agent.llm import DEFAULT_MODEL_ID, DeepSeekClient, LLMConfig, load_llm_config
from chenguang_agent.loop import AgentLoop
from chenguang_agent.models import ToolCall, TraceEvent, Turn
from chenguang_agent.observe import append_observability
from chenguang_agent.planner import LLMPlanner, OfflinePlanner
from chenguang_agent.router import Router
from chenguang_agent.safety import INPUT_TOO_LONG, RATE_LIMITED, RateLimiter, max_input_chars, rejection
from chenguang_agent.session_store import SessionStore
from chenguang_agent.tools import Toolbelt

__all__ = ["CustomerServiceAgent", "ToolCall", "TraceEvent", "Turn"]

_NO_KEY = "未配置 DEEPSEEK_API_KEY，本轮由离线策略逐步决定"


class CustomerServiceAgent:
    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        *,
        use_llm: bool = False,
        planner: str | None = None,
        llm_config: LLMConfig | None = None,
        llm_client=None,
        limiter: RateLimiter | None = None,
        max_iterations: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.kb = kb or load_kb()
        self.router = Router(
            complaint_keywords=self.kb.complaint_keywords,
            pregnancy_keywords=self.kb.pregnancy_keywords,
            acute_keywords=self.kb.acute_keywords,
            efficacy_keywords=self.kb.efficacy_keywords,
            tracking_keywords=self.kb.tracking_keywords,
        )
        self.guardrails = Guardrails(self.kb)
        self.tools = Toolbelt(self.kb)
        self.use_llm = use_llm
        self.llm_config = llm_config
        self.llm_client = llm_client
        self.limiter = limiter or RateLimiter()
        self.max_iterations = max_iterations
        self.timeout_s = timeout_s
        if planner is not None:
            self.planner_mode = planner
        elif use_llm:
            self.planner_mode = "llm"
        else:
            self.planner_mode = "auto"
        self.coordinator = Coordinator(
            router=self.router,
            guardrails=self.guardrails,
            tools=self.tools,
            use_llm=use_llm,
            llm_config=llm_config,
        )

    def run(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        observe: bool = False,
        turn_id: str | None = None,
        rate_key: str | None = None,
    ) -> Turn:
        turn_id = turn_id or ("t-" + uuid.uuid4().hex[:12])
        session_value = session_id or ""
        bucket = rate_key if rate_key is not None else (session_value or None)
        blocked = rejection(user_text, rate_key=bucket, limiter=self.limiter)
        if blocked is not None:
            kind, reply = blocked
            note = "超过输入长度上限，拒绝后不调用工具" if kind == INPUT_TOO_LONG else "会话限流，拒绝后不调用工具"
            turn = Turn(
                user_text=_stored_user_text(user_text),
                intent="未识别",
                signals=[],
                reply=reply,
                escalated=False,
                notes=[f"协调 · 拒绝：{note}"],
                flags={INPUT_TOO_LONG: kind == INPUT_TOO_LONG, RATE_LIMITED: kind == RATE_LIMITED},
                trace=[TraceEvent("协调", "拒绝", note)],
                turn_id=turn_id,
                session_id=session_value,
                kb_version=self.kb.kb_version,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._persist(turn, session_value, observe)
            return turn

        self.tools.bind_context(user_text=user_text, session_id=session_value, turn_id=turn_id)
        memory = SessionStore().memory(session_value) if session_value else []
        planner, fallback, preface, initial_fallback = self._planners()
        turn = AgentLoop(
            planner=planner,
            fallback=fallback,
            tools=self.tools,
            guardrails=self.guardrails,
            checker=self.coordinator.checker,
            max_iterations_limit=self.max_iterations,
            timeout_s=self.timeout_s,
        ).run(user_text, memory, llm_fallback=initial_fallback)
        if preface:
            event = TraceEvent("agent", "回落", preface)
            turn.trace.insert(0, event)
            turn.notes.insert(0, f"{event.role} · {event.action}：{event.detail}")
        turn.planner = self._planner_label(turn)
        turn.turn_id = turn_id
        turn.session_id = session_value
        turn.kb_version = self.kb.kb_version
        turn.created_at = datetime.now(timezone.utc).isoformat()
        self._persist(turn, session_value, observe)
        return turn

    def _planners(self):
        offline = OfflinePlanner(self.router, self.guardrails, self.tools)
        mode = self.planner_mode
        config = self.llm_config or load_llm_config()
        if mode == "auto":
            if config is None and self.llm_client is None:
                return offline, offline, "", ""
            mode = "llm"
        if mode == "offline":
            return offline, offline, "", ""
        if self.llm_client is None and config is None:
            return offline, offline, _NO_KEY, "no_api_key"
        client = self.llm_client or DeepSeekClient(config)
        return LLMPlanner(client, self.router, self.guardrails, self.tools), offline, "", ""

    def _planner_label(self, turn: Turn) -> str:
        if not turn.used_llm:
            return "offline"
        if self.llm_config is not None and self.llm_config.model:
            return self.llm_config.model
        loaded = load_llm_config()
        if loaded is not None and loaded.model:
            return loaded.model
        return DEFAULT_MODEL_ID

    def _persist(self, turn: Turn, session_id: str, observe: bool) -> None:
        if session_id:
            SessionStore().append(session_id, turn.to_dict())
        if observe:
            append_observability(turn.observation())


def _stored_user_text(user_text: str) -> str:
    limit = max_input_chars()
    if len(user_text) <= limit:
        return user_text
    return user_text[:limit] + "…（已截断存储）"

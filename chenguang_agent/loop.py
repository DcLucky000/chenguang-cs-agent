"""Think, act, observe, think again.

This loop never chooses a tool name. A planner returns the next action; the
loop runs it, writes the observation back, and asks the planner again.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from chenguang_agent.checker import Checker
from chenguang_agent.guardrails import SAFE_FALLBACK, Guardrails, InputScreen
from chenguang_agent.llm import PlannerError
from chenguang_agent.models import Observations, ToolCall, TraceEvent, Turn
from chenguang_agent.render import render_reply
from chenguang_agent.schemas import tool_error

DEFAULT_MAX_ITERATIONS = 8
DEFAULT_TURN_TIMEOUT_S = 25.0


def max_iterations() -> int:
    raw = os.environ.get("CHENGUANG_MAX_ITERATIONS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MAX_ITERATIONS


def turn_timeout_s() -> float:
    raw = os.environ.get("CHENGUANG_TURN_TIMEOUT_S", "").strip()
    if not raw:
        return DEFAULT_TURN_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TURN_TIMEOUT_S
    if value < 0:
        return DEFAULT_TURN_TIMEOUT_S
    return value


@dataclass
class LoopState:
    user_text: str
    memory: list[dict]
    messages: list[dict] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    rejections: list[list[str]] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    intent: str = ""
    signals: list[str] = field(default_factory=list)
    screen: InputScreen = field(default_factory=InputScreen)
    faq: dict | None = None
    products: dict | None = None
    escalation: dict | None = None
    tracking_only: bool = False
    clarify_only: bool = False
    lookup_product: bool = False
    product_query: str = ""
    memory_sku: str = ""
    planned: bool = False
    draft: str = ""
    approved: bool = False
    used_llm: bool = False
    llm_fallback: str = ""
    send_backs: int = 0
    iterations: int = 0
    stop: str = ""
    llm_reject_count: int = 0


class AgentLoop:
    def __init__(
        self,
        *,
        planner,
        fallback,
        tools,
        guardrails: Guardrails,
        checker: Checker,
        max_iterations_limit: int | None = None,
        timeout_s: float | None = None,
        clock=time.monotonic,
    ) -> None:
        self.planner = planner
        self.fallback = fallback
        self.tools = tools
        self.guardrails = guardrails
        self.checker = checker
        self.max_iterations_limit = max_iterations() if max_iterations_limit is None else max_iterations_limit
        self.timeout_s = turn_timeout_s() if timeout_s is None else timeout_s
        self.clock = clock

    def run(self, user_text: str, memory: list[dict] | None = None, *, llm_fallback: str = "") -> Turn:
        state = LoopState(user_text=user_text, memory=list(memory or []), llm_fallback=llm_fallback)
        started = self.clock()
        planner = self.planner
        while True:
            if state.iterations >= self.max_iterations_limit:
                state.stop = "max_iterations"
                state.trace.append(TraceEvent("agent", "停止", f"达到最大迭代 {self.max_iterations_limit}"))
                break
            remaining = self.timeout_s - (self.clock() - started)
            if remaining <= 0:
                state.stop = "timeout"
                state.trace.append(TraceEvent("agent", "停止", f"超过本轮时限 {self.timeout_s:g} 秒"))
                break
            try:
                decision = planner.decide(state, timeout_s=remaining)
            except PlannerError as exc:
                state.trace.append(TraceEvent("agent", "失败", f"模型调用失败（{exc}），交给离线策略继续决定下一步"))
                if planner is self.fallback or state.llm_fallback == "call_failed":
                    state.stop = "planner_failure"
                    state.llm_fallback = state.llm_fallback or "call_failed"
                    break
                state.llm_fallback = "call_failed"
                planner = self.fallback
                state.trace.append(TraceEvent("agent", "回环", "模型不可用，改由离线策略按已有观察决定下一步"))
                continue

            state.iterations += 1
            state.trace.append(TraceEvent("agent", "思考", decision.thought or "决定下一步"))
            if decision.raw_message is not None:
                state.messages.append(decision.raw_message)

            if decision.kind == "tools":
                self._act(state, decision)
                continue

            if decision.intent:
                state.intent = decision.intent
            review = self.checker.review(decision.reply, _observations(state))
            if review.approved:
                state.approved = True
                state.draft = decision.reply
                state.used_llm = decision.source == "llm"
                state.trace.append(TraceEvent("检查", "通过", "这稿符合护栏，可以发给顾客"))
                break

            state.send_backs += 1
            state.rejections.append(list(review.reasons))
            detail = "；".join(review.reasons)
            if decision.source == "llm":
                detail = "模型回复不通过：" + detail
            state.trace.append(TraceEvent("检查", "打回", detail))
            if decision.source == "llm":
                state.llm_reject_count += 1
                hard = any("输出护栏" in reason for reason in review.reasons)
                if hard or state.llm_reject_count >= 2:
                    state.llm_fallback = "checker_rejected"
                    planner = self.fallback
                    state.trace.append(TraceEvent("agent", "回环", "丢弃未通过检查的模型回复，改由离线策略按观察重写"))
                    continue
                state.messages.append(
                    {
                        "role": "user",
                        "content": "检查打回，请重新决定下一步。可以再调用工具，或改写回复。不要新增事实。原因：" + "；".join(review.reasons),
                    }
                )
                state.trace.append(TraceEvent("agent", "回环", "把检查意见交回模型，等它重新决定"))
                continue
            if _stuck(state.rejections):
                state.stop = "checker_stuck"
                state.draft = decision.reply
                state.trace.append(TraceEvent("agent", "停止", "同一稿连续被打回，停止再改"))
                break
            state.trace.append(TraceEvent("agent", "回环", "检查未通过，按意见再决定下一步"))

        if not state.approved:
            state.draft = _best_effort(state)
        reply, violations = self.guardrails.screen_output(state.draft or SAFE_FALLBACK, escalated=state.escalation is not None)
        if violations:
            state.trace.append(TraceEvent("检查", "拦截", "输出护栏改写了最终稿：" + "、".join(violations)))
        flags = {
            "complaint": state.screen.complaint,
            "pregnancy": state.screen.pregnancy,
            "acute": state.screen.acute,
            "efficacy": state.screen.efficacy,
            "tracking": state.screen.tracking,
        }
        if state.stop == "max_iterations":
            flags["max_iterations"] = True
        if state.stop == "timeout":
            flags["timed_out"] = True
        if state.stop == "planner_failure":
            flags["planner_failed"] = True
        notes = [f"{event.role} · {event.action}：{event.detail}" for event in state.trace]
        return Turn(
            user_text=state.user_text,
            intent=state.intent or "未识别",
            signals=list(state.signals),
            reply=reply,
            escalated=state.escalation is not None,
            tool_calls=list(state.calls),
            notes=notes,
            output_violations=violations,
            flags=flags,
            used_llm=state.used_llm,
            trace=list(state.trace),
            send_backs=state.send_backs,
            llm_fallback=state.llm_fallback,
        )

    def _act(self, state: LoopState, decision) -> None:
        calls = list(decision.tool_calls)
        for index, call in enumerate(calls):
            arguments = call.arguments if isinstance(call.arguments, dict) else None
            decomposed = _decomposition_note(state, call.name, arguments or {})
            if decomposed:
                state.trace.append(TraceEvent("agent", "拆解", decomposed))
            state.trace.append(
                TraceEvent("agent", "行动", f"调用 {call.name or '未知工具'} {_dump(arguments or {})}")
            )
            started = time.perf_counter()
            if arguments is None:
                output = tool_error(call.name or "unknown", "invalid_arguments", "入参必须是对象")
            else:
                try:
                    output = self.tools.call(call.name, arguments)
                except Exception as exc:  # noqa: BLE001 — loop boundary, same as the toolbelt
                    output = tool_error(call.name, "tool_failure", type(exc).__name__)
            latency_ms = (time.perf_counter() - started) * 1000
            if not isinstance(output, dict):
                output = tool_error(call.name, "tool_failure", "工具没有返回对象")
            state.calls.append(
                ToolCall(
                    name=call.name,
                    arguments=dict(arguments or {}),
                    output=output,
                    latency_ms=latency_ms,
                )
            )
            output = state.calls[-1].output
            _absorb(state, call.name, output)
            detail = _observe_detail(call.name, output)
            state.trace.append(TraceEvent("agent", "观察", detail))
            state.observations.append({"name": call.name, "output": _public_output(output)})
            if decision.raw_message is not None:
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or f"call_{state.iterations}_{index}",
                        "content": json.dumps(_public_output(output), ensure_ascii=False),
                    }
                )


def _observations(state: LoopState) -> Observations:
    return Observations(
        user_text=state.user_text,
        intent=state.intent or "未识别",
        signals=list(state.signals),
        screen=state.screen,
        faq=state.faq,
        products=state.products,
        escalation=state.escalation,
        tracking_only=state.tracking_only,
        clarify_only=state.clarify_only,
        lookup_product=state.lookup_product,
        calls=list(state.calls),
    )


def _absorb(state: LoopState, name: str, output: dict) -> None:
    if output.get("ok") is False:
        if name == "lookup_faq" and state.faq is None:
            state.faq = {"found": False}
        return
    if name == "lookup_faq":
        state.faq = output
    elif name == "lookup_product":
        state.products = output
    elif name == "escalate_to_human":
        state.escalation = output


def _best_effort(state: LoopState) -> str:
    if state.draft and state.stop == "checker_stuck":
        return state.draft
    if not state.planned and not state.calls:
        return SAFE_FALLBACK
    try:
        return render_reply(
            user_text=state.user_text,
            intent=state.intent or "未识别",
            screen=state.screen,
            faq=state.faq,
            products=state.products,
            escalation=state.escalation,
        )
    except Exception:  # noqa: BLE001
        return SAFE_FALLBACK


def _stuck(rejections: list[list[str]]) -> bool:
    if len(rejections) < 2:
        return False
    return rejections[-1] == rejections[-2]


def _public_output(output: dict) -> dict:
    """Hide selling points from the model. The stored tool trace keeps them."""

    matches = output.get("matches")
    if not isinstance(matches, list):
        return output
    cloned = dict(output)
    cloned["matches"] = [
        {key: value for key, value in item.items() if key != "selling_points"} if isinstance(item, dict) else item
        for item in matches
    ]
    return cloned


def _observe_detail(name: str, output: dict) -> str:
    if output.get("ok") is False:
        error = output.get("error") or {}
        code = error.get("code") or "tool_failure"
        return f"{name} 返回结构化错误 {code}，不把异常抛到 CLI"
    if name == "lookup_faq":
        if not output.get("found"):
            return "lookup_faq 没有命中"
        permission = "允许直接答" if output.get("bot_may_answer") else "不允许直接答"
        return f"lookup_faq 命中 {output.get('q_id')}（{permission}）"
    if name == "lookup_product":
        if not output.get("found"):
            return "lookup_product 没有命中"
        ids = "、".join(str(item.get("sku_id", "")) for item in output.get("matches", []) if isinstance(item, dict))
        return f"lookup_product 命中 {ids}"
    if name == "escalate_to_human":
        return f"escalate_to_human 工单 {output.get('ticket_id', '')}"
    return f"{name} 已返回"


def _dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(", ", ": "))


def _decomposition_note(state: LoopState, name: str, arguments: dict) -> str:
    """Describe a model split. The loop does not choose the argument, it only records it."""

    question = arguments.get("question") if name == "lookup_faq" else None
    query = arguments.get("query") if name == "lookup_product" else None
    if isinstance(question, str) and question.strip() and question.strip() != state.user_text:
        return f"这一步只处理子问题「{question.strip()}」，不是把整句原话一次做完"
    if not isinstance(query, str) or not query.strip():
        return ""
    if query.strip() == state.user_text or query.strip() in state.user_text:
        return ""
    if not state.observations:
        return ""
    previous = json.dumps(state.observations[-1].get("output", {}), ensure_ascii=False)
    token = query.strip()
    if token in previous and token not in state.user_text:
        return f"下一步参数「{token}」来自上一步观察，顾客原话里没有这串字"
    return ""

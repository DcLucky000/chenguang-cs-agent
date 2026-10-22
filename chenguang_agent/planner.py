"""Planners decide the next step. The loop does not.

``OfflinePlanner`` is a deterministic stand-in used when there is no API key,
so the gold set stays reproducible. It still returns one action at a time and
reads the latest observation before the next action.

``LLMPlanner`` asks DeepSeek. Whatever tool call the model returns is the next
action. Two different model transcripts for the same utterance can take two
different tool paths.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from chenguang_agent.guardrails import Guardrails
from chenguang_agent.kb import INTENTS
from chenguang_agent.llm import LLMResult, PlannedToolCall, PlannerError, tool_specs
from chenguang_agent.render import render_reply
from chenguang_agent.router import Router
from chenguang_agent.tools import Toolbelt

_ANAPHORA = ("那款", "刚才", "上一款", "那个呢", "它多少", "这个呢")
_INTENT_LINE = re.compile(r"^意图[:：]\s*(售前咨询|物流查询|售后政策|投诉情绪|未识别)\s*")


@dataclass
class Decision:
    kind: str  # "tools" or "final"
    thought: str
    tool_calls: list[PlannedToolCall] = field(default_factory=list)
    reply: str = ""
    intent: str = ""
    source: str = "offline"
    raw_message: dict | None = None


def memory_sku_for(memory: list[dict], user_text: str) -> str:
    if not any(token in user_text for token in _ANAPHORA):
        return ""
    for item in reversed(memory):
        skus = item.get("skus") or []
        if skus:
            return str(skus[-1])
    return ""


def prepare_case(state, router: Router, guardrails: Guardrails, tools: Toolbelt) -> None:
    """Label the turn for the checker and the offline policy. Does not call tools."""

    if state.planned:
        return
    intent, signals = router.classify(state.user_text)
    screen = guardrails.inspect(state.user_text)
    if screen.complaint and intent != "投诉情绪":
        intent = "投诉情绪"
    tracking_only = bool(screen.tracking and not (screen.complaint or screen.pregnancy or screen.acute))
    clarify_only = False
    if tracking_only and intent == "未识别":
        intent = "物流查询"
    elif intent == "未识别" and not (screen.complaint or screen.pregnancy or screen.acute):
        clarify_only = True

    memory_sku = memory_sku_for(state.memory, state.user_text)
    product_query = state.user_text
    lookup_product = (
        not tracking_only
        and not clarify_only
        and not screen.complaint
        and not screen.pregnancy
        and tools.matches_product(state.user_text)
    )
    if memory_sku and not screen.complaint and not screen.pregnancy and not screen.acute and not tracking_only:
        clarify_only = False
        if intent == "未识别":
            intent = "售前咨询"
        if not lookup_product:
            lookup_product = True
            product_query = memory_sku

    state.intent = intent
    state.signals = signals
    state.screen = screen
    state.tracking_only = tracking_only
    state.clarify_only = clarify_only
    state.lookup_product = lookup_product
    state.product_query = product_query
    state.memory_sku = memory_sku
    state.planned = True


class OfflinePlanner:
    """One next action from the observations so far. Not a pre-built tool list."""

    def __init__(self, router: Router, guardrails: Guardrails, tools: Toolbelt) -> None:
        self.router = router
        self.guardrails = guardrails
        self.tools = tools

    def decide(self, state, *, timeout_s: float) -> Decision:
        del timeout_s
        prepare_case(state, self.router, self.guardrails, self.tools)
        if self._should_call(state, "lookup_faq", self._need_faq(state)):
            return Decision(
                kind="tools",
                thought="还没有可用的 FAQ 观察。这一步只查 FAQ，查完再决定要不要查商品或转人工。",
                tool_calls=[PlannedToolCall(id="", name="lookup_faq", arguments={"question": state.user_text})],
            )
        product_query = self._product_query(state)
        if product_query and self._should_call(state, "lookup_product", True):
            seen = "还没有 FAQ 观察，但这一步不需要它。"
            if state.faq and state.faq.get("found"):
                seen = f"看到 FAQ {state.faq.get('q_id')} 之后，"
            elif any(call.name == "lookup_faq" for call in state.calls):
                seen = "FAQ 没有命中之后，"
            return Decision(
                kind="tools",
                thought=seen + "原话或会话里还有未核对的商品，下一步只查目录。",
                tool_calls=[PlannedToolCall(id="", name="lookup_product", arguments={"query": product_query})],
            )
        reason = None if state.tracking_only or not self._faq_settled(state) else self.guardrails.escalation_reason(state.screen, state.faq)
        if reason and self._should_call(state, "escalate_to_human", True):
            return Decision(
                kind="tools",
                thought=f"观察表明必须转人工（{reason}）。先建工单，再写给顾客的话。",
                tool_calls=[PlannedToolCall(id="", name="escalate_to_human", arguments={"reason": reason})],
            )
        draft = render_reply(
            user_text=state.user_text,
            intent=state.intent,
            screen=state.screen,
            faq=state.faq,
            products=state.products,
            escalation=state.escalation,
        )
        return Decision(kind="final", thought=_final_thought(state), reply=draft, intent=state.intent)

    def _need_faq(self, state) -> bool:
        return not (state.tracking_only or state.clarify_only)

    def _faq_settled(self, state) -> bool:
        """Escalation looks at the FAQ row, so it waits until that observation exists."""

        if not self._need_faq(state):
            return True
        return any(call.name == "lookup_faq" for call in state.calls)

    def _product_query(self, state) -> str:
        """Decided only once the FAQ observation is in, not as a step list up front."""

        if not self._faq_settled(state):
            return ""
        if state.tracking_only or state.clarify_only or state.screen.complaint or state.screen.pregnancy:
            return ""
        if state.memory_sku and not self.tools.matches_product(state.user_text):
            return state.memory_sku
        if self.tools.matches_product(state.user_text):
            return state.user_text
        return ""

    def _should_call(self, state, name: str, needed: bool) -> bool:
        if not needed:
            return False
        return not any(call.name == name for call in state.calls)


class LLMPlanner:
    """Ask the model what to do next. Tool choice comes only from the response."""

    def __init__(self, client, router: Router, guardrails: Guardrails, tools: Toolbelt) -> None:
        self.client = client
        self.router = router
        self.guardrails = guardrails
        self.tools = tools

    def decide(self, state, *, timeout_s: float) -> Decision:
        prepare_case(state, self.router, self.guardrails, self.tools)
        if not state.messages:
            state.messages.extend(_initial_messages(state))
        result = self.client.complete(state.messages, tool_specs(), max(0.1, timeout_s))
        if not isinstance(result, LLMResult):
            raise PlannerError("bad_client_result")
        if not result.tool_calls and not result.content.strip():
            raise PlannerError("empty_response")
        if result.tool_calls:
            # One tool per think. A model that lists several steps must look at the
            # observation before the loop will run the next one.
            call = result.tool_calls[0]
            thought = result.thought.strip() or "模型决定先做这一步，看到观察后再决定要不要继续。"
            if len(result.tool_calls) > 1:
                names = "、".join(item.name or "?" for item in result.tool_calls)
                thought = f"{thought} 模型本轮还提到了 {names}，循环只执行第一个，其余等观察后再判断。"
            return Decision(
                kind="tools",
                thought=thought,
                tool_calls=[call],
                source="llm",
                raw_message=_single_tool_message(result.raw_message, call.id),
            )
        intent, reply = _split_intent(result.content)
        thought = result.thought.strip() or "模型决定结束工具调用，直接回复顾客。"
        return Decision(
            kind="final",
            thought=thought,
            reply=reply,
            intent=intent if intent in INTENTS else "",
            source="llm",
            raw_message=result.raw_message,
        )


def _final_thought(state) -> str:
    if state.rejections:
        prefix = "检查打回后，根据已有观察重写。"
    elif _last_failed(state):
        prefix = "工具返回失败，不再重复同一次调用，按已有结果回复。"
    elif state.tracking_only:
        prefix = "这是包裹轨迹。没有轨迹工具，不检索。"
    elif state.clarify_only:
        prefix = "没有命中业务范围，不检索。"
    else:
        prefix = "这一步的观察够用来回复了。"
    if state.faq and state.faq.get("found"):
        return prefix + f"已经看到 {state.faq.get('q_id')}。"
    if state.calls:
        return prefix + "已经看到工具结果。"
    return prefix + "没有调用工具。"


def _last_failed(state) -> bool:
    if not state.calls:
        return False
    return state.calls[-1].output.get("ok") is False


def _single_tool_message(raw: dict | None, call_id: str) -> dict | None:
    if not isinstance(raw, dict):
        return raw
    calls = raw.get("tool_calls")
    if not isinstance(calls, list) or len(calls) <= 1:
        return raw
    kept = [item for item in calls if isinstance(item, dict) and str(item.get("id") or "") == call_id]
    cloned = dict(raw)
    cloned["tool_calls"] = kept or calls[:1]
    return cloned


def _split_intent(content: str) -> tuple[str, str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    match = _INTENT_LINE.match(text)
    if not match:
        return "", text
    reply = text[match.end() :].lstrip("\n").strip()
    return match.group(1), reply


def _initial_messages(state) -> list[dict]:
    memory = state.memory[-6:]
    hint = {
        "规则给的意图标签": state.intent,
        "只是标签不是流程": "你可以忽略这个标签，自己决定先查 FAQ、查商品、转人工，还是直接回复。",
        "会话记忆": memory,
        "指代": state.memory_sku or "",
    }
    system = (
        "你是澄光肌研的客服 Agent，不是固定脚本。"
        "每一步只决定一件事：调用一个工具，或在信息够了之后回复顾客。"
        "看到工具结果后你会再次被调用，必须根据观察重新决定，不要假设后面的步骤已经排好。"
        "一个问题里如果有多件事，拆开：每次只调用一个工具，用子问题或从上一步观察里读到的商品名/SKU 作为参数。"
        "不要在同一次回复里把后续工具都排完。观察回来之后再决定下一步，或者结束。"
        "如果观察文本里出现了顾客原话里没有的 SKU 或商品，而你还没查过它，下一步应查这个商品。"
        "会话记忆里的商品可以用来理解「那款」「刚才」。"
        "禁止编造物流轨迹和运单号，禁止承诺美白祛痘等医疗功效，禁止答应退款或补差价金额，"
        "禁止对孕哺说能用或不能用。投诉、孕哺、已经刺痛红肿、FAQ 不允许直接答时要 escalate_to_human。"
        "回复用中文，对顾客说话，不要出现内部词 Bot，不要复述卖点。"
        "需要改意图时，回复第一行写「意图：售前咨询」这类标签（售前咨询、物流查询、售后政策、投诉情绪、未识别），下一行起才是给顾客的话。"
        "不调用工具就表示这一步结束并回复顾客。"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps({"顾客原话": state.user_text, "提示": hint}, ensure_ascii=False),
        },
    ]

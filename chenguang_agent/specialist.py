"""Specialist: look up FAQ and products, then draft. Revision follows checker reasons."""

from __future__ import annotations

import time

from chenguang_agent.checker import Review
from chenguang_agent.guardrails import Guardrails
from chenguang_agent.models import Observations, ToolCall, TraceEvent
from chenguang_agent.render import render_reply
from chenguang_agent.tools import Toolbelt


class Specialist:
    def __init__(self, tools: Toolbelt, guardrails: Guardrails) -> None:
        self.tools = tools
        self.guardrails = guardrails

    def gather(self, obs: Observations, trace: list[TraceEvent]) -> None:
        if obs.tracking_only or obs.clarify_only:
            trace.append(TraceEvent("执行", "检索", "按调度不调用工具"))
            return

        faq = self._call(obs, "lookup_faq", {"question": obs.user_text})
        if faq.get("ok") is False:
            obs.faq = {"found": False}
            trace.append(TraceEvent("执行", "失败", _failure_note("lookup_faq", faq)))
        else:
            obs.faq = faq
            trace.append(TraceEvent("执行", "检索", _faq_note(faq)))
            if faq.get("alternatives"):
                trace.append(TraceEvent("执行", "检索", "相近条目：" + "；".join(faq["alternatives"])))

        if obs.lookup_product:
            products = self._call(obs, "lookup_product", {"query": obs.user_text})
            if products.get("ok") is False:
                obs.products = None
                trace.append(TraceEvent("执行", "失败", _failure_note("lookup_product", products)))
            else:
                obs.products = products
                names = "、".join(item["sku_id"] for item in products.get("matches", []))
                trace.append(TraceEvent("执行", "检索", f"lookup_product 命中 {names}"))

    def propose(self, obs: Observations, trace: list[TraceEvent]) -> str:
        """First draft is assembled from raw tool text. Compliance is the checker's job."""
        if obs.tracking_only:
            draft = "我先按常见情况帮您看，包裹正在运输中，您再等一下就能到。"
            trace.append(TraceEvent("执行", "起草", "没有轨迹数据，第 1 稿按经验写成了在途"))
            return draft
        if obs.clarify_only:
            topic = obs.user_text.strip()
            draft = f"关于「{topic}」，我们可以随便聊聊别的。"
            trace.append(TraceEvent("执行", "起草", "第 1 稿把未识别问题当成了闲聊"))
            return draft

        parts: list[str] = []
        notes: list[str] = []
        if obs.faq and obs.faq.get("found"):
            parts.append(obs.faq["answer"])
            notes.append("FAQ 原文")
        matches = []
        if obs.products and obs.products.get("found"):
            matches = list(obs.products["matches"][:2])
        if matches:
            for item in matches:
                points = item["selling_points"].replace("|", "、")
                parts.append(f"{item['name']}的卖点是{points}。")
            notes.append("商品卖点")
        if not parts:
            trace.append(TraceEvent("执行", "起草", "没有检索结果，第 1 稿按经验写了一句"))
            return "我先按经验给您一个说法。"
        detail = "第 1 稿只拼接了 " + "和".join(notes) + "，还没做合规改写"
        trace.append(TraceEvent("执行", "起草", detail))
        return "".join(parts)

    def revise(self, obs: Observations, review: Review, trace: list[TraceEvent]) -> str:
        done: list[str] = []
        if review.requires_escalation and obs.escalation is None:
            reason = self.guardrails.escalation_reason(obs.screen, obs.faq)
            if reason:
                escalation = self._call(obs, "escalate_to_human", {"reason": reason})
                if escalation.get("ok") is False:
                    trace.append(TraceEvent("执行", "失败", _failure_note("escalate_to_human", escalation)))
                else:
                    obs.escalation = escalation
                    trace.append(TraceEvent("执行", "升级", f"escalate_to_human，原因：{reason}"))
                    done.append("补上转人工")
        if any("卖点" in reason for reason in review.reasons):
            done.append("去掉卖点，只留注意事项和问答要点")
        if any("口吻" in reason or "Bot" in reason for reason in review.reasons):
            done.append("改成对顾客说话")
        if any("物流" in reason or "编造" in reason for reason in review.reasons):
            done.append("改为明确不编造物流")
        if any("闲聊" in reason or "澄清" in reason for reason in review.reasons):
            done.append("改为澄清可咨询范围")
        if any("商品名" in reason or "药品" in reason or "药效" in reason for reason in review.reasons):
            done.append("补上功效边界")
        if any("话术" in reason for reason in review.reasons) and "补上转人工" not in done:
            done.append("补上转人工话术")
        if not done:
            done.append("按知识库允许说出的内容重写")
        draft = render_reply(
            user_text=obs.user_text,
            intent=obs.intent,
            screen=obs.screen,
            faq=obs.faq,
            products=obs.products,
            escalation=obs.escalation,
        )
        trace.append(TraceEvent("执行", "改稿", "；".join(done)))
        return draft

    def _call(self, obs: Observations, name: str, arguments: dict) -> dict:
        started = time.perf_counter()
        output = self.tools.call(name, arguments)
        latency_ms = (time.perf_counter() - started) * 1000
        obs.calls.append(
            ToolCall(name=name, arguments=dict(arguments), output=output, latency_ms=latency_ms)
        )
        return output


def _failure_note(name: str, output: dict) -> str:
    error = output.get("error") or {}
    code = error.get("code") or "tool_failure"
    return f"{name} 返回结构化错误 {code}，不把异常抛到 CLI"


def _faq_note(faq: dict) -> str:
    if not faq.get("found"):
        return "lookup_faq 没有命中"
    permission = "允许直接答" if faq["bot_may_answer"] else "不允许直接答"
    matched = "、".join(faq["matched"])
    return f"lookup_faq 命中 {faq['q_id']} {faq['question']}（{permission}，词：{matched}）"

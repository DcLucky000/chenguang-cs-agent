"""Input and output guardrails. These outrank FAQ permissions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from chenguang_agent.kb import KnowledgeBase
from chenguang_agent.textutil import matched_keywords, squash

# Phrases that must never survive into a customer-facing reply.
BANNED_PHRASES = (
    "保证美白",
    "一定美白",
    "一定能美白",
    "医疗级",
    "包治",
    "可以治愈",
    "具有药效",
    "保证祛痘",
    "一定祛痘",
    "可以去掉眼袋",
    "保证去眼袋",
    "能祛眼袋",
    "孕妇可以用",
    "哺乳期可以用",
    "孕期可以用",
    "备孕可以用",
    "保证不过敏",
    "一定不会过敏",
    "绝对不会过敏",
    "完全不会过敏",
    "同意退全款",
    "马上退全款",
    "已为您退款",
    "已退全款",
    "同意补差价",
    "已补差价",
    "已签收",
    "已揽收",
    "派送中",
    "运输中",
    "运单号",
    "预计明天送达",
    "明天就能到",
)

_WAYBILL = re.compile(r"[A-Za-z]{2}\d{8,}")


def _join_script(body: str, script: str) -> str:
    body = body.strip()
    if body and body[-1] not in "。！？":
        body += "。"
    return body + script


SAFE_FALLBACK = "我不能承诺医疗功效，不能编造物流进度，也不能直接同意退款或补偿。"


@dataclass
class InputScreen:
    complaint_hits: list[str] = field(default_factory=list)
    pregnancy_hits: list[str] = field(default_factory=list)
    acute_hits: list[str] = field(default_factory=list)
    efficacy_hits: list[str] = field(default_factory=list)
    tracking_hits: list[str] = field(default_factory=list)

    @property
    def complaint(self) -> bool:
        return bool(self.complaint_hits)

    @property
    def pregnancy(self) -> bool:
        return bool(self.pregnancy_hits)

    @property
    def acute(self) -> bool:
        return bool(self.acute_hits)

    @property
    def efficacy(self) -> bool:
        return bool(self.efficacy_hits)

    @property
    def tracking(self) -> bool:
        return bool(self.tracking_hits)


class Guardrails:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    def inspect(self, text: str) -> InputScreen:
        compact = squash(text)
        return InputScreen(
            complaint_hits=matched_keywords(compact, self.kb.complaint_keywords),
            pregnancy_hits=matched_keywords(compact, self.kb.pregnancy_keywords),
            acute_hits=matched_keywords(compact, self.kb.acute_keywords),
            efficacy_hits=matched_keywords(compact, self.kb.efficacy_keywords),
            tracking_hits=matched_keywords(compact, self.kb.tracking_keywords),
        )

    def escalation_reason(self, screen: InputScreen, faq: dict | None) -> str | None:
        if screen.complaint:
            return "命中投诉或退全款威胁，强制转人工"
        if screen.pregnancy:
            return "孕哺期咨询，不提供能否使用的绝对建议"
        if screen.acute:
            return "已出现刺激或红肿等不适，先提示停用就医并转人工"
        if faq and faq.get("found") and not faq.get("bot_may_answer"):
            return f"知识库 {faq['q_id']} 不允许 Bot 直接答复"
        return None

    def screen_output(self, reply: str, *, escalated: bool) -> tuple[str, list[str]]:
        violations = [phrase for phrase in BANNED_PHRASES if phrase in reply]
        if _WAYBILL.search(reply):
            violations.append("疑似运单号")
        script = self.kb.escalation_script
        if violations:
            cleaned = SAFE_FALLBACK + (script if escalated else "")
            return cleaned, violations
        if escalated and script not in reply:
            return _join_script(reply, script), []
        return reply, []

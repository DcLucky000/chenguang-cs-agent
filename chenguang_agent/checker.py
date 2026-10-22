"""Checker: review a draft and reject it with reasons. It does not rewrite."""

from __future__ import annotations

import re
from dataclasses import dataclass

from chenguang_agent.guardrails import Guardrails
from chenguang_agent.models import Observations
from chenguang_agent.render import CLARIFY, LOGISTICS_CODA

_PREGNANCY_BANS = (
    "孕妇可以用",
    "哺乳期可以用",
    "孕期可以用",
    "备孕可以用",
    "可以用烟酰胺",
    "建议使用",
    "隔天",
)

_TRACKING_BANS = ("已签收", "运单号", "派送中", "已揽收", "运输中")

_DRUG_TOKENS = ("祛痘", "药效", "药用", "药品")


@dataclass
class Review:
    approved: bool
    reasons: list[str]
    requires_escalation: bool


class Checker:
    def __init__(self, guardrails: Guardrails) -> None:
        self.guardrails = guardrails

    def review(self, draft: str, obs: Observations) -> Review:
        reasons: list[str] = []
        _, violations = self.guardrails.screen_output(draft, escalated=False)
        if violations:
            reasons.append("输出护栏命中：" + "、".join(violations))

        leaked = _leaked_selling_points(draft, obs)
        if leaked:
            shown = "、".join(leaked[:4])
            suffix = f"等{len(leaked)}处" if len(leaked) > 4 else ""
            reasons.append(f"卖点不能写进回复：{shown}{suffix}")

        if "Bot" in draft:
            reasons.append("草稿还是内部口吻（Bot），要改成对顾客说话")

        must_reason = self.guardrails.escalation_reason(obs.screen, obs.faq)
        script = self.guardrails.kb.escalation_script
        if must_reason and obs.escalation is None:
            reasons.append(_escalation_gap(obs))
        if must_reason and script not in draft:
            reasons.append("转人工回复缺少固定话术")
        if not must_reason and script and script in draft:
            reasons.append("这个问题不该使用转人工话术")

        if obs.screen.pregnancy:
            if "医嘱" not in draft:
                reasons.append("孕哺问题必须提示遵医嘱")
            if re.search(r"SKU\s*\d", draft, flags=re.IGNORECASE):
                reasons.append("孕哺问题不能推荐具体商品")
            for needle in _PREGNANCY_BANS:
                if needle in draft:
                    reasons.append(f"孕哺回复不能出现「{needle}」")

        if obs.screen.acute:
            for needle in ("停用", "冲洗", "就医"):
                if needle not in draft:
                    reasons.append(f"急性不适回复必须包含「{needle}」")

        if obs.tracking_only:
            if "不会编造" not in draft:
                reasons.append("没有物流轨迹工具，必须明确拒绝编造")
            for needle in _TRACKING_BANS:
                if needle in draft:
                    reasons.append(f"不能编造物流状态「{needle}」")

        if obs.intent == "物流查询" and not must_reason and not obs.tracking_only:
            if LOGISTICS_CODA not in draft:
                reasons.append("物流答复必须声明不会凭空补充物流状态")

        products = _matches(obs)
        drug_question = obs.screen.efficacy and any(token in obs.user_text for token in _DRUG_TOKENS)
        if drug_question and "不是祛痘药" not in draft:
            reasons.append("涉及祛痘药或药效时，必须澄清这不是药品")

        open_efficacy = (
            obs.screen.efficacy
            and not products
            and not must_reason
            and not obs.screen.acute
            and not obs.screen.pregnancy
            and not obs.screen.complaint
        )
        if open_efficacy and "具体商品名" not in draft:
            reasons.append("没有点名商品的功效问题，要请顾客补充具体商品名")

        if obs.clarify_only and draft != CLARIFY:
            reasons.append("未识别问题只能澄清可咨询的范围，不能当闲聊回答")

        for item in products:
            audience = item.get("audience", "")
            if "不承诺" in audience and item["sku_id"] in draft and "不承诺" not in draft:
                reasons.append(f"{item['sku_id']} 的适用说明要求保留「不承诺」")

        return Review(
            approved=not reasons,
            reasons=reasons,
            requires_escalation=bool(must_reason) and obs.escalation is None,
        )


def _matches(obs: Observations) -> list[dict]:
    if not obs.products or not obs.products.get("found"):
        return []
    return list(obs.products.get("matches") or [])


def _leaked_selling_points(draft: str, obs: Observations) -> list[str]:
    leaked: list[str] = []
    for item in _matches(obs):
        for point in item.get("selling_points", "").split("|"):
            point = point.strip()
            if point and point in draft and point not in leaked:
                leaked.append(point)
    return leaked


def _escalation_gap(obs: Observations) -> str:
    faq = obs.faq or {}
    if obs.screen.complaint:
        return "投诉或退全款威胁必须转人工，不能直接承诺退款"
    if obs.screen.pregnancy:
        return "孕哺问题不提供能否使用的绝对建议，必须转人工"
    if obs.screen.acute and faq.get("bot_may_answer"):
        return f"知识库 {faq.get('q_id', '')} 允许直接答，但急性不适仍必须转人工"
    if faq.get("found") and not faq.get("bot_may_answer"):
        return f"知识库 {faq.get('q_id', '')} 不允许直接答复，必须转人工"
    return "必须转人工，但还没有调用 escalate_to_human"

"""Rule router. Safety lexicons come from guardrails.md; commercial words stay here."""

from __future__ import annotations

from chenguang_agent.textutil import matched_keywords, squash

# Higher-risk intents win when keyword counts tie.
_TIE_BREAK = ("售后政策", "物流查询", "售前咨询")

_PRESALE = [
    "敏感肌",
    "烟酰胺",
    "卸妆",
    "防晒",
    "眼霜",
    "眼部",
    "洁面",
    "成分",
    "正品",
    "防伪",
    "假货",
    "专柜",
    "真伪",
    "包邮",
    "运费",
    "套装",
    "酒精",
    "香精",
    "临期",
    "保质期",
    "生产日期",
    "屏障",
    "适合",
    "面膜",
    "泥膜",
    "面霜",
    "精华水",
    "身体乳",
    "旅行装",
    "怎么用",
    "护肤顺序",
]

_AFTERSALE = [
    "退货",
    "换货",
    "退换",
    "无理由",
    "发票",
    "开票",
    "增值税",
    "差价",
    "补差价",
    "保价",
    "买贵",
    "退款",
    "售后",
]

_LOGISTICS = [
    "收货地址",
    "改一下收货",
    "改收货地址",
    "改地址",
    "改收货",
    "发货",
    "快递",
    "物流",
    "中通",
    "圆通",
    "顺丰",
    "包裹",
    "几天发",
]


class Router:
    def __init__(
        self,
        complaint_keywords: list[str],
        pregnancy_keywords: list[str],
        acute_keywords: list[str],
        efficacy_keywords: list[str],
        tracking_keywords: list[str],
    ) -> None:
        self.complaint_keywords = complaint_keywords
        self.buckets = {
            "售前咨询": _PRESALE + pregnancy_keywords + efficacy_keywords,
            "售后政策": _AFTERSALE + acute_keywords,
            "物流查询": _LOGISTICS + tracking_keywords,
        }

    def classify(self, text: str) -> tuple[str, list[str]]:
        compact = squash(text)
        complaints = matched_keywords(compact, self.complaint_keywords)
        if complaints:
            return "投诉情绪", complaints

        scored = {
            intent: matched_keywords(compact, keywords) for intent, keywords in self.buckets.items()
        }
        best_count = max((len(hits) for hits in scored.values()), default=0)
        if best_count == 0:
            return "未识别", []

        tied = [intent for intent, hits in scored.items() if len(hits) == best_count]
        winner = min(tied, key=_TIE_BREAK.index)
        return winner, scored[winner]

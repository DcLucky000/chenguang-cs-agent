"""Compose the customer reply from tool observations only."""

from __future__ import annotations

from chenguang_agent.guardrails import InputScreen

CLARIFY = (
    "这个问题超出了我能处理的范围。我是澄光肌研的客服助手，可以帮您看商品、发货和售后。"
    "您可以问拍下多久发货、满多少包邮，或某款适不适合敏感肌。"
)

TRACKING_REFUSAL = (
    "我查不到具体物流轨迹，也不会编造包裹到了哪一步。"
    "您可以在订单页查看物流。若要拦截或核对异常，需要人工处理，您告诉我后我再转接。"
)

_ACUTE_DEFAULT = (
    "请立即停用并用清水冲洗。如持续不适请及时就医。请保留订单与产品批次信息，由人工客服协助售后评估。"
)

_NOT_A_DRUG = "澄光的商品是化妆品，不是祛痘药，也不能替代药品或医嘱。"

LOGISTICS_CODA = "我看不到具体包裹进度，也不会凭空补充物流状态。"


def customerize(answer: str) -> str:
    text = answer.replace("Bot不承诺一定不过敏", "我不能向您保证一定不过敏")
    return text.replace("Bot", "我")


def render_reply(
    *,
    user_text: str,
    intent: str,
    screen: InputScreen,
    faq: dict | None,
    products: dict | None,
    escalation: dict | None,
) -> str:
    script = (escalation or {}).get("script") or ""
    if screen.complaint:
        return _complaint(user_text, faq, script)
    if screen.pregnancy:
        return _pregnancy(faq, script)
    if screen.acute:
        return _acute(faq, products, script)
    if escalation:
        return _policy_handoff(faq, script)
    if screen.tracking:
        return TRACKING_REFUSAL
    if intent == "未识别":
        return CLARIFY
    if screen.efficacy:
        return _efficacy(user_text, faq, products)
    return _normal(intent, faq, products)


def describe_products(products: dict | None) -> str:
    if not products or not products.get("found"):
        return ""
    parts: list[str] = []
    for item in products["matches"][:2]:
        caution = item["caution"].replace("|", "；")
        points = item["faq_points"].replace("|", "；")
        sentence = (
            f"{item['name']}（{item['sku_id']}，{item['spec']}，{item['price']}元，{item['stock']}）。"
            f"注意事项：{caution}。说明要点：{points}。"
        )
        if "不承诺" in item.get("audience", ""):
            sentence += f"适用说明：{item['audience']}。"
        parts.append(sentence)
    return "".join(parts)


def _join_script(body: str, script: str) -> str:
    body = body.strip()
    if not script or script in body:
        return body
    if body and body[-1] not in "。！？":
        body += "。"
    return body + script


def _complaint(user_text: str, faq: dict | None, script: str) -> str:
    if faq and faq.get("q_id") in {"F15", "F16"}:
        body = customerize(faq["answer"])
    elif any(token in user_text for token in ("退全款", "退款")):
        body = "理解您的心情。退款或补偿需要人工按规则审核，我不能直接答应退全款。"
    else:
        body = "非常抱歉给您带来不好的体验。这类问题由人工专员处理，我立即为您转接。"
    return _join_script(body, script)


def _pregnancy(faq: dict | None, script: str) -> str:
    if faq and faq.get("q_id") == "F08":
        body = customerize(faq["answer"])
    else:
        body = "孕哺期请遵医嘱。我不能给出能用或不能用的绝对建议，请咨询医生。"
    return _join_script(body, script)


def _acute(faq: dict | None, products: dict | None, script: str) -> str:
    if faq and faq.get("q_id") == "F05":
        body = customerize(faq["answer"])
    else:
        body = _ACUTE_DEFAULT
    return _join_script(body + describe_products(products), script)


def _policy_handoff(faq: dict | None, script: str) -> str:
    if faq and faq.get("found"):
        body = customerize(faq["answer"])
    else:
        body = "这件事我不能直接处理。"
    return _join_script(body, script)


def _efficacy(user_text: str, faq: dict | None, products: dict | None) -> str:
    parts: list[str] = []
    if faq and faq.get("found"):
        parts.append(customerize(faq["answer"]))
    else:
        parts.append("我不承诺医疗或绝对化功效，效果因人而异。")
    if any(token in user_text for token in ("祛痘", "药效", "药用", "药品")):
        parts.append(_NOT_A_DRUG)
    described = describe_products(products)
    if described:
        parts.append(described)
    elif not (faq and faq.get("found")):
        parts.append("您告诉我具体商品名，我按该商品说明里的注意事项回答，不会另外承诺效果。")
    else:
        parts.append("您告诉我具体商品名，我可以按该商品说明里的注意事项回答，不会另外承诺效果。")
    return "".join(parts)


def _normal(intent: str, faq: dict | None, products: dict | None) -> str:
    parts: list[str] = []
    if faq and faq.get("found"):
        parts.append(customerize(faq["answer"]))
    described = describe_products(products)
    if described:
        parts.append(described)
    if intent == "物流查询":
        parts.append(LOGISTICS_CODA)
    if not parts:
        parts.append("我没有查到可以直接回答的条目，也不会凭空补充。您可以补充商品名，或换个说法问发货、售后和政策。")
    return "".join(parts)

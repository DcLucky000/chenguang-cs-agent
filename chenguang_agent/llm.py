"""DeepSeek tool-calling client.

The product model is DeepSeek-V4.1-Flash-1017 (user name: DeepSeek v4.1Flash1017).
DeepSeek's API serves that snapshot as ``deepseek-flash``. The legacy ids
``deepseek-v4-flash`` and ``deepseek-v4-flash-vision-exp`` route to the same
weights; ``deepseek-chat`` was retired. Override with ``DEEPSEEK_MODEL`` only
to pin another id.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# Official API id for DeepSeek-V4.1-Flash-1017.
DEFAULT_MODEL_ID = "deepseek-flash"
MODEL_SNAPSHOT = "DeepSeek-V4.1-Flash-1017"
DEFAULT_BASE_URL = "https://api.deepseek.com"


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str


@dataclass
class PlannedToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    """One model turn. ``thought`` is the decision trace; tools or content are the action."""

    thought: str
    content: str
    tool_calls: list[PlannedToolCall] = field(default_factory=list)
    raw_message: dict = field(default_factory=dict)


class PlannerError(RuntimeError):
    """The model call failed. The loop records it and can hand off to the offline planner."""


def load_llm_config() -> LLMConfig | None:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None
    base = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    return LLMConfig(api_key=key, base_url=base, model=model)


class DeepSeekClient:
    """OpenAI-compatible chat completions against the DeepSeek API."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete(self, messages: list[dict], tools: list[dict], timeout: float) -> LLMResult:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            # Tool calls are the decision. Thinking mode is off so the action is the tool call.
            "thinking": {"type": "disabled"},
        }
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                detail = ""
            raise PlannerError(f"http_{exc.code} {detail}".strip()) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PlannerError(type(exc).__name__) from exc
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PlannerError("empty_response") from exc
        if not isinstance(message, dict):
            raise PlannerError("empty_response")
        return parse_message(message)


def parse_message(message: dict) -> LLMResult:
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = str(content)
    reasoning = message.get("reasoning_content") or ""
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    tool_calls: list[PlannedToolCall] = []
    raw_calls = message.get("tool_calls") or []
    if isinstance(raw_calls, list):
        for index, call in enumerate(raw_calls):
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(function.get("name") or call.get("name") or "").strip()
            arguments = _arguments(function.get("arguments", call.get("arguments")))
            call_id = str(call.get("id") or f"call_{index}_{name or 'tool'}")
            tool_calls.append(PlannedToolCall(id=call_id, name=name, arguments=arguments))
    thought = reasoning.strip() or (content.strip() if tool_calls else "")
    raw = {"role": "assistant", "content": content}
    if tool_calls:
        raw["tool_calls"] = raw_calls
    if reasoning:
        raw["reasoning_content"] = reasoning
    return LLMResult(thought=thought, content=content.strip(), tool_calls=tool_calls, raw_message=raw)


def _arguments(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def tool_specs() -> list[dict]:
    """OpenAI function tools. The model picks which one to call, or none to finish."""

    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_faq",
                "description": "在澄光肌研 FAQ 里检索与顾客这句话最接近的标准答。政策、售后、发货、功效边界不确定时调用。不要臆造知识库里没有的规则。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "用来检索的顾客问题，可以是原话或你拆出来的一个子问题。"},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_product",
                "description": "按商品名或 SKU 查价格、库存、注意事项和问答要点。卖点字段只供内部判断，不能写进给顾客的回复。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "商品名、别名或 SKU。"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "escalate_to_human",
                "description": "创建人工工单。投诉威胁、孕哺、已经出现的刺痛红肿、以及 FAQ 标明不能直接答的问题必须调用。不要自己答应退款金额。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "升级原因，一句中文。"},
                    },
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        },
    ]

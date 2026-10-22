"""JSON Schema (subset) for the three tools. No third-party validator."""

from __future__ import annotations

# type / properties / required / additionalProperties / minLength / maxLength.
TOOL_SCHEMAS: dict[str, dict] = {
    "lookup_product": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "lookup_faq": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
    "escalate_to_human": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
}


def validate_tool_arguments(schema: dict, arguments: object) -> str | None:
    """Return an error message, or None when the arguments match the schema."""
    if not isinstance(arguments, dict):
        return "入参必须是对象"
    properties: dict = schema.get("properties") or {}
    required: list = schema.get("required") or []
    if schema.get("additionalProperties") is False:
        extra = [key for key in arguments if key not in properties]
        if extra:
            return "出现了未声明的字段：" + "、".join(sorted(extra))
    for key in required:
        if key not in arguments:
            return f"缺少字段 {key}"
    for key, spec in properties.items():
        if key not in arguments:
            continue
        value = arguments[key]
        if spec.get("type") == "string":
            if not isinstance(value, str):
                return f"{key} 必须是字符串"
            if spec.get("minLength") and len(value.strip()) < int(spec["minLength"]):
                return f"{key} 不能为空"
            if "maxLength" in spec and len(value) > int(spec["maxLength"]):
                return f"{key} 超过 {spec['maxLength']} 字"
    return None


def tool_error(tool: str, code: str, message: str) -> dict:
    return {
        "ok": False,
        "tool": tool,
        "error": {"code": code, "message": message, "tool": tool},
    }

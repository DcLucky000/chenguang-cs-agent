"""Command line: ask one question, run the gold set, chat, or serve a demo HTTP port."""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from chenguang_agent import __version__
from chenguang_agent.agent import CustomerServiceAgent, Turn
from chenguang_agent.eval_runner import format_report, report_json, run_eval
from chenguang_agent.kb import load_kb
from chenguang_agent.server import serve
from chenguang_agent.session_store import SessionStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chenguang_agent",
        description="澄光肌研客服 Agent：由 DeepSeek 决定下一步和工具，检查护栏不通过则再决定。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m chenguang_agent eval\n"
            "  python -m chenguang_agent ask 敏感肌能用洁面吗\n"
            "  python -m chenguang_agent ask --json --session demo 你们就是骗子\n"
            "  python -m chenguang_agent history --session demo\n"
            "  python -m chenguang_agent serve\n"
            "  python -m chenguang_agent chat\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"chenguang-agent {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="问一句，打印意图、工具轨迹和回复")
    ask.add_argument("question", nargs="+", help="顾客原话")
    ask.add_argument("--llm", action="store_true", help="强制用 DEEPSEEK_API_KEY 决定下一步")
    ask.add_argument("--offline", action="store_true", help="强制离线逐步决策，不调用模型")
    ask.add_argument("--json", action="store_true", help="输出 JSON")
    ask.add_argument("--session", default="", help="可选会话 ID，按 turn_id 追加到 runs/sessions/")

    evaluate = sub.add_parser("eval", help="跑 15 条金标和护栏探针")
    evaluate.add_argument("-v", "--verbose", action="store_true", help="打印每条回复和编排笔记")
    evaluate.add_argument("--json", action="store_true", help="输出 JSON")

    chat = sub.add_parser("chat", help="连续对话，输入 exit 结束")
    chat.add_argument("--llm", action="store_true", help="强制用 DEEPSEEK_API_KEY 决定下一步")
    chat.add_argument("--offline", action="store_true", help="强制离线逐步决策，不调用模型")
    chat.add_argument("--session", default="", help="可选会话 ID，缺省时自动生成")

    history = sub.add_parser("history", help="查看某个会话已经落盘的轮次")
    history.add_argument("--session", required=True, help="会话 ID")

    server = sub.add_parser("serve", help="本机演示 HTTP：/health 和 /ask")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--llm", action="store_true", help="强制用 DEEPSEEK_API_KEY 决定下一步")
    server.add_argument("--offline", action="store_true", help="强制离线逐步决策，不调用模型")

    args = parser.parse_args(argv)
    if args.cmd == "eval":
        report = run_eval()
        if args.json:
            print(report_json(report))
        else:
            print(format_report(report, verbose=args.verbose))
        return 0 if report["ok"] else 1
    if args.cmd == "ask":
        question = " ".join(args.question)
        forced = _planner_flag(args)
        if forced is None:
            return 2
        turn = CustomerServiceAgent(planner=forced).run(
            question,
            session_id=args.session or None,
            observe=True,
        )
        if args.json:
            print(json.dumps(turn.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(format_turn(turn))
        return 0
    if args.cmd == "chat":
        forced = _planner_flag(args)
        if forced is None:
            return 2
        return _chat(planner=forced, session_id=args.session)
    if args.cmd == "history":
        return _history(args.session)
    if args.cmd == "serve":
        forced = _planner_flag(args)
        if forced is None:
            return 2
        return serve(args.host, args.port, planner=forced)
    parser.print_help()
    return 2


def _planner_flag(args) -> str | None:
    if getattr(args, "offline", False) and getattr(args, "llm", False):
        print("不要同时使用 --llm 和 --offline", file=sys.stderr)
        return None
    if getattr(args, "offline", False):
        return "offline"
    if getattr(args, "llm", False):
        return "llm"
    return "auto"


def _chat(*, planner: str, session_id: str) -> int:
    kb = load_kb()
    agent = CustomerServiceAgent(kb, planner=planner)
    session = session_id or ("chat-" + uuid.uuid4().hex[:8])
    print(f"{kb.store_name} · 客服")
    print(f"会话 {session}。输入问题，exit 结束。每一步由规划器决定调用哪个工具。没有 DEEPSEEK_API_KEY 时走离线逐步决策。")
    print(f"知识库 {kb.kb_version}")
    while True:
        try:
            question = input("\n顾客> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            return 0
        print(format_turn(agent.run(question, session_id=session, observe=True)))


def _history(session_id: str) -> int:
    rows = SessionStore().read(session_id)
    if not rows:
        print(f"没有会话 {session_id} 的记录。崩溃后看 runs/sessions/ 里的 JSONL，进程退出后文件还在。")
        return 1
    for row in rows:
        brief = {
            "turn_id": row.get("turn_id"),
            "intent": row.get("intent"),
            "tool_calls": [
                {"name": call.get("name"), "latency_ms": call.get("latency_ms")}
                for call in row.get("tool_calls") or []
            ],
            "send_backs": row.get("send_backs"),
            "escalate": row.get("escalate"),
            "kb_version": row.get("kb_version"),
            "reply": row.get("reply"),
        }
        print(json.dumps(brief, ensure_ascii=False))
    return 0


def format_turn(turn: Turn) -> str:
    lines = [
        f"模式   {_mode(turn)}",
        f"意图   {turn.intent}",
        f"知识库   {turn.kb_version}",
        f"升级原因 {turn.escalate_reason or '无'}",
        f"回环   {turn.send_backs} 次打回" if turn.send_backs else "回环   检查通过，未打回",
        f"决策   {_decider(turn)}",
        "协作",
    ]
    for index, event in enumerate(turn.trace, start=1):
        lines.append(f"  {index}. {event.role} · {event.action}：{event.detail}")
    if turn.tool_calls:
        lines.append("工具")
        for index, call in enumerate(turn.tool_calls, start=1):
            lines.append(f"  {index}. {call.name} · {call.latency_ms:.1f}ms")
            lines.append("     入参 " + _dumps(call.arguments))
            lines.append("     出参 " + _summarize(call.name, call.output))
    else:
        lines.append("工具   无")
    if turn.llm_fallback:
        lines.append(f"回落   {turn.llm_fallback}")
    lines.append("回复")
    lines.append(turn.reply)
    return "\n".join(lines)


def _decider(turn: Turn) -> str:
    if turn.used_llm:
        return turn.planner or "deepseek-flash"
    if turn.llm_fallback:
        return f"离线策略（{turn.llm_fallback}）"
    return "离线策略"


def _mode(turn: Turn) -> str:
    if turn.flags.get("rate_limited"):
        return "限流"
    if turn.flags.get("input_too_long"):
        return "拒绝超长输入"
    if turn.flags.get("tracking") and not turn.escalated:
        return "拒绝编造物流"
    if turn.intent == "未识别":
        return "澄清"
    if turn.flags.get("complaint") and turn.escalated:
        return "强制转人工"
    if turn.escalated:
        return "转人工"
    if turn.flags.get("efficacy"):
        return "拒绝功效承诺"
    return "直接答复"


def _dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(", ", ": "))


def _summarize(name: str, output: dict) -> str:
    if output.get("ok") is False:
        error = output.get("error") or {}
        return f"结构化错误 {error.get('code', 'tool_failure')} {error.get('message', '')}"
    if name == "lookup_faq":
        if not output.get("found"):
            return "没有命中"
        permission = "允许直接答" if output["bot_may_answer"] else "不允许直接答"
        return f"{output['q_id']} {output['question']} · {permission} · {output['answer']}"
    if name == "lookup_product":
        if not output.get("found"):
            return "没有命中"
        parts = []
        for item in output["matches"]:
            parts.append(
                f"{item['sku_id']} {item['name']} · 注意事项 {item['caution'].replace('|', '；')}"
            )
        return "；".join(parts)
    if name == "escalate_to_human":
        return f"{output['ticket_id']} · {output['queue']} · {output['reason']}"
    return _dumps(output)


if __name__ == "__main__":
    sys.exit(main())

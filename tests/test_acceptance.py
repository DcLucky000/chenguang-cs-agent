"""Session append, safety, checker send-back, kb version, HTTP, and eval totals."""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from chenguang_agent.agent import CustomerServiceAgent
from chenguang_agent.checker import Checker
from chenguang_agent.cli import main
from chenguang_agent.coordinator import Coordinator
from chenguang_agent.eval_runner import run_eval
from chenguang_agent.kb import load_kb
from chenguang_agent.llm import LLMConfig
from chenguang_agent.safety import RateLimiter
from chenguang_agent.specialist import Specialist
from chenguang_agent.schemas import tool_error
from chenguang_agent.server import make_server
from chenguang_agent.session_store import SessionStore
from tests.support import isolated_runs

ROOT = Path(__file__).resolve().parent.parent


class AcceptanceTest(unittest.TestCase):
    def test_kb_version_is_stable_and_on_the_turn(self) -> None:
        first = load_kb().kb_version
        second = load_kb().kb_version
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("2026-10-12.1+"))
        with isolated_runs():
            turn = CustomerServiceAgent().run("满多少包邮")
        self.assertEqual(turn.kb_version, first)

    def test_session_appends_by_turn_id_and_does_not_overwrite(self) -> None:
        with isolated_runs():
            agent = CustomerServiceAgent()
            agent.run("满多少包邮", session_id="demo", turn_id="t1")
            agent.run("你们就是骗子", session_id="demo", turn_id="t2")
            CustomerServiceAgent().run("快递到哪了", session_id="demo", turn_id="t1")
            rows = SessionStore().read("demo")
        self.assertEqual([row["turn_id"] for row in rows], ["t1", "t2"])
        self.assertEqual(rows[0]["intent"], "售前咨询")
        self.assertEqual(rows[1]["intent"], "投诉情绪")
        self.assertIn("tool_calls", rows[0])
        self.assertIn("send_backs", rows[0])
        self.assertIn("trace", rows[0])
        self.assertTrue(rows[1]["escalate"]["escalated"])
        self.assertTrue(rows[1]["escalate"]["ticket_id"])
        self.assertEqual(rows[1]["escalate"]["reason"], rows[1]["escalate_reason"])
        self.assertNotEqual(rows[0]["intent"], "物流查询")

    def test_three_roles_and_checker_send_back(self) -> None:
        from chenguang_agent.llm import parse_message

        with isolated_runs():
            agent = CustomerServiceAgent(planner="offline")
            self.assertIsInstance(agent.coordinator, Coordinator)
            self.assertIsInstance(agent.coordinator.specialist, Specialist)
            self.assertIsInstance(agent.coordinator.checker, Checker)
            normal = agent.run("满多少包邮")
            actions = [event.action for event in normal.trace]
            observe_at = actions.index("观察")
            self.assertIn("思考", actions[:observe_at])
            self.assertIn("行动", actions[:observe_at])
            self.assertIn("思考", actions[observe_at + 1 :])
            self.assertIn("通过", actions)
            self.assertEqual([call.name for call in normal.tool_calls], ["lookup_faq"])
            looped = agent.run("敏感肌能用洁面吗")
            thoughts = [event.detail for event in looped.trace if event.action == "思考"]
            self.assertTrue(any("F07" in detail for detail in thoughts))
            self.assertEqual([call.name for call in looped.tool_calls], ["lookup_faq", "lookup_product"])
            self.assertNotIn("卖点", looped.reply)
            self.assertNotIn("温和清洁不紧绷", looped.reply)

            class Poison:
                def complete(self, messages, tools, timeout):
                    del messages, tools, timeout
                    return parse_message({"role": "assistant", "content": "这款保证美白，同意退全款。"})

            rewritten = CustomerServiceAgent(
                planner="llm",
                llm_client=Poison(),
                llm_config=LLMConfig("test-key", "http://127.0.0.1:9", "deepseek-flash"),
            ).run("烟酰胺精华能美白到什么程度")
        self.assertFalse(rewritten.used_llm)
        self.assertGreaterEqual(rewritten.send_backs, 1)
        self.assertEqual(rewritten.llm_fallback, "checker_rejected")
        self.assertNotIn("保证美白", rewritten.reply)
        self.assertNotIn("同意退全款", rewritten.reply)
        self.assertTrue(any(event.action == "打回" and "模型回复" in event.detail for event in rewritten.trace))
        self.assertTrue(any(event.action == "回环" for event in rewritten.trace))

    def test_llm_fallbacks_are_recorded(self) -> None:
        from chenguang_agent.llm import PlannerError

        old = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with isolated_runs():
                missing = CustomerServiceAgent(use_llm=True).run("满多少包邮")

                class Boom:
                    def complete(self, messages, tools, timeout):
                        del messages, tools, timeout
                        raise PlannerError("timeout")

                broken = CustomerServiceAgent(
                    use_llm=True,
                    llm_client=Boom(),
                    llm_config=LLMConfig("test-key", "http://127.0.0.1:9", "deepseek-flash"),
                ).run("满多少包邮")
        finally:
            if old is not None:
                os.environ["DEEPSEEK_API_KEY"] = old
        self.assertFalse(missing.used_llm)
        self.assertEqual(missing.llm_fallback, "no_api_key")
        self.assertEqual(missing.planner, "offline")
        self.assertTrue(any("DEEPSEEK_API_KEY" in note for note in missing.notes))
        self.assertIn("99", missing.reply)
        self.assertFalse(broken.used_llm)
        self.assertEqual(broken.llm_fallback, "call_failed")
        self.assertIn("99", broken.reply)
        self.assertTrue(any("模型调用失败" in event.detail for event in broken.trace))

    def test_length_and_rate_limit(self) -> None:
        old = os.environ.get("CHENGUANG_MAX_INPUT_CHARS")
        os.environ["CHENGUANG_MAX_INPUT_CHARS"] = "8"
        try:
            with isolated_runs():
                too_long = CustomerServiceAgent().run("敏感肌能用洁面吗？请把注意事项、成分和用法都展开说清楚")
                limiter = RateLimiter(limit=2, window_s=60)
                agent = CustomerServiceAgent(limiter=limiter)
                first = agent.run("满多少包邮", session_id="burst")
                second = agent.run("发什么快递", session_id="burst")
                third = agent.run("你们就是骗子", session_id="burst")
        finally:
            if old is None:
                os.environ.pop("CHENGUANG_MAX_INPUT_CHARS", None)
            else:
                os.environ["CHENGUANG_MAX_INPUT_CHARS"] = old
        self.assertTrue(too_long.flags["input_too_long"])
        self.assertEqual(too_long.tool_calls, [])
        self.assertIn("没有检索", too_long.reply)
        self.assertFalse(first.flags.get("rate_limited"))
        self.assertFalse(second.flags.get("rate_limited"))
        self.assertTrue(third.flags["rate_limited"])
        self.assertEqual(third.tool_calls, [])
        self.assertNotIn("转接", third.reply)

    def test_tool_failure_stays_inside_the_turn(self) -> None:
        with isolated_runs():
            agent = CustomerServiceAgent()

            def broken(name: str, arguments: dict, timeout_s: float = 2.0) -> dict:
                return tool_error(name, "tool_failure", "RuntimeError")

            agent.tools.call = broken  # type: ignore[method-assign]
            turn = agent.run("满多少包邮")
        self.assertTrue(turn.reply)
        self.assertTrue(any("结构化错误" in note for note in turn.notes))
        self.assertFalse(turn.escalated)

    def test_cli_json_history_and_observability(self) -> None:
        with isolated_runs() as root:
            sink = StringIO()
            with redirect_stdout(sink):
                code = main(["ask", "--json", "--session", "demo", "满多少包邮"])
            self.assertEqual(code, 0)
            with redirect_stdout(sink):
                code = main(["ask", "--session", "demo", "你们就是骗子"])
            self.assertEqual(code, 0)
            buf = StringIO()
            with redirect_stdout(buf):
                code = main(["ask", "--json", "--session", "demo", "满多少包邮"])
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            for key in ("intent", "send_backs", "tools", "escalate_reason", "kb_version", "trace"):
                self.assertIn(key, payload)
            self.assertEqual(payload["intent"], "售前咨询")
            self.assertIn("latency_ms", payload["tools"][0])
            self.assertEqual(payload["tools"][0]["name"], "lookup_faq")
            history = StringIO()
            with redirect_stdout(history):
                status = main(["history", "--session", "demo"])
            self.assertEqual(status, 0)
            lines = [json.loads(line) for line in history.getvalue().splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 2)
            self.assertEqual(lines[0]["intent"], "售前咨询")
            observed = [
                json.loads(line)
                for line in (root / "observability.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreaterEqual(len(observed), 2)
            self.assertIn("escalate_reason", observed[-1])
            self.assertIn("kb_version", observed[-1])
            self.assertIn("send_backs", observed[-1])

    def test_serve_health_and_ask(self) -> None:
        with isolated_runs():
            server = make_server("127.0.0.1", 0, limiter=RateLimiter(limit=1, window_s=60))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
                    health = json.loads(response.read().decode("utf-8"))
                self.assertTrue(health["ok"])
                self.assertTrue(health["kb_version"].startswith("2026-10-12.1+"))
                body = json.dumps({"question": "满多少包邮", "session_id": "http"}).encode("utf-8")
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/ask",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request) as response:
                    turn = json.loads(response.read().decode("utf-8"))
                self.assertEqual(turn["intent"], "售前咨询")
                self.assertIn("99", turn["reply"])
                self.assertIn("latency_ms", turn["tools"][0])
                again = urllib.request.Request(
                    f"http://127.0.0.1:{port}/ask",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(again)
                self.assertEqual(caught.exception.code, 429)
                limited = json.loads(caught.exception.read().decode("utf-8"))
                self.assertTrue(limited["flags"]["rate_limited"])
            finally:
                server.shutdown()
                server.server_close()

    def test_eval_totals_do_not_regress(self) -> None:
        with isolated_runs():
            report = run_eval()
        self.assertEqual(report["passed"], 15)
        self.assertEqual(report["total"], 15)
        self.assertEqual(report["intent_correct"], 15)
        self.assertEqual(report["escalation_recall"], [6, 6])
        self.assertEqual(report["escalation_false"], [0, 9])
        self.assertEqual(report["probes_passed"], 5)
        self.assertEqual(report["probes_total"], 5)
        self.assertTrue(report["ok"])
        self.assertTrue(str(report["kb_version"]).startswith("2026-10-12.1+"))
        self.assertTrue((ROOT / "scripts" / "ci_eval.sh").is_file())
        self.assertTrue((ROOT / ".github" / "workflows" / "eval.yml").is_file())


if __name__ == "__main__":
    unittest.main()

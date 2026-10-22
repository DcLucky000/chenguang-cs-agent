"""Evidence for the six agent criteria.

Criterion 1 and 4 are the ones a fixed router fails. These tests inject a model
transcript. The loop does not know which tool comes next; it runs whatever the
transcript returns, including a second argument that exists only in the previous
observation.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chenguang_agent.agent import CustomerServiceAgent
from chenguang_agent.llm import (
    DEFAULT_MODEL_ID,
    MODEL_SNAPSHOT,
    LLMConfig,
    PlannerError,
    parse_message,
)
from chenguang_agent.render import CLARIFY
from tests.support import isolated_runs


def _tool_message(name: str, arguments: dict, thought: str, call_id: str) -> object:
    return parse_message(
        {
            "role": "assistant",
            "content": thought,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            ],
        }
    )


def _final_message(content: str) -> object:
    return parse_message({"role": "assistant", "content": content})


class ScriptedClient:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.seen: list[list[dict]] = []
        self.tools: list[dict] = []

    def complete(self, messages, tools, timeout):
        del timeout
        self.seen.append(copy.deepcopy(messages))
        self.tools = tools
        return self.responder(messages)


class CriterionTests(unittest.TestCase):
    def test_model_id_is_deepseek_v41_flash_1017(self) -> None:
        self.assertEqual(DEFAULT_MODEL_ID, "deepseek-flash")
        self.assertIn("V4.1-Flash-1017", MODEL_SNAPSHOT)
        self.assertIn("1017", MODEL_SNAPSHOT)

    def test_model_choice_not_fixed_route(self) -> None:
        """Same utterance, two transcripts, two tool paths. The loop obeys the model."""

        question = "满多少包邮"

        def lookup_then_finish(messages):
            if not any(item.get("role") == "tool" for item in messages):
                return _tool_message("lookup_faq", {"question": question}, "先查包邮规则", "call_faq")
            return _final_message("演示店满99包邮，下单页以结算为准。")

        def finish_immediately(messages):
            del messages
            return _final_message("演示店满99包邮，下单页以结算为准。")

        with isolated_runs():
            looked = CustomerServiceAgent(
                planner="llm",
                llm_client=ScriptedClient(lookup_then_finish),
                llm_config=LLMConfig("test", "http://127.0.0.1:9", DEFAULT_MODEL_ID),
            ).run(question)
            skipped = CustomerServiceAgent(
                planner="llm",
                llm_client=ScriptedClient(finish_immediately),
                llm_config=LLMConfig("test", "http://127.0.0.1:9", DEFAULT_MODEL_ID),
            ).run(question)
        self.assertEqual([call.name for call in looked.tool_calls], ["lookup_faq"])
        self.assertEqual(looked.tool_calls[0].arguments["question"], question)
        self.assertEqual(skipped.tool_calls, [])
        self.assertTrue(looked.used_llm)
        self.assertTrue(skipped.used_llm)
        self.assertEqual(looked.planner, DEFAULT_MODEL_ID)
        self.assertIn("99", looked.reply)
        self.assertIn("99", skipped.reply)
        self.assertNotEqual(
            [call.name for call in looked.tool_calls],
            [call.name for call in skipped.tool_calls],
        )

    def test_think_act_observe_think_order(self) -> None:
        question = "满多少包邮"

        def responder(messages):
            if not any(item.get("role") == "tool" for item in messages):
                return _tool_message("lookup_faq", {"question": question}, "先查这一件事", "call_faq")
            blob = json.dumps(messages, ensure_ascii=False)
            if "F03" not in blob:
                raise AssertionError("second think did not see the FAQ observation")
            return _final_message("演示店满99包邮，下单页以结算为准。")

        client = ScriptedClient(responder)
        with isolated_runs():
            turn = CustomerServiceAgent(
                planner="llm",
                llm_client=client,
                llm_config=LLMConfig("test", "http://127.0.0.1:9", DEFAULT_MODEL_ID),
            ).run(question)
        actions = [event.action for event in turn.trace]
        first_observe = actions.index("观察")
        self.assertEqual(actions[actions.index("思考") : first_observe + 1], ["思考", "行动", "观察"])
        self.assertIn("思考", actions[first_observe + 1 :])
        self.assertEqual(len(client.seen), 2)
        self.assertEqual(client.seen[1][-1]["role"], "tool")
        self.assertIn("F03", client.seen[1][-1]["content"])

    def test_decomposes_using_observation(self) -> None:
        """SKU08 is not in the utterance. The model may only learn it from the FAQ observation."""

        utterance = "帮我确认一下说明里点名的那件库存"

        class StubTools:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict]] = []

            def call(self, name, arguments, timeout_s=2.0):
                del timeout_s
                self.calls.append((name, dict(arguments)))
                if name == "lookup_faq":
                    return {
                        "found": True,
                        "q_id": "F99",
                        "question": "库存",
                        "answer": "库存以目录为准，请改查 SKU08，不要猜。",
                        "intent": "售前咨询",
                        "bot_may_answer": True,
                        "matched": [],
                        "score": 1,
                        "alternatives": [],
                    }
                if name == "lookup_product":
                    return {"found": False, "query": arguments.get("query", ""), "matches": []}
                return {"ok": False, "error": {"code": "unknown_tool", "message": name, "tool": name}}

        def responder(messages):
            tool_msgs = [item for item in messages if item.get("role") == "tool"]
            if not tool_msgs:
                return _tool_message(
                    "lookup_faq",
                    {"question": "库存说明"},
                    "原话里有两件事，先只查库存说明",
                    "call_faq",
                )
            if len(tool_msgs) == 1:
                body = tool_msgs[0]["content"]
                sku = body[body.index("SKU08") : body.index("SKU08") + 5]
                return _tool_message(
                    "lookup_product",
                    {"query": sku},
                    "观察里点了名，下一步只查这个 SKU",
                    "call_sku",
                )
            return _final_message(CLARIFY)

        client = ScriptedClient(responder)
        with isolated_runs():
            agent = CustomerServiceAgent(
                planner="llm",
                llm_client=client,
                llm_config=LLMConfig("test", "http://127.0.0.1:9", DEFAULT_MODEL_ID),
            )
            stub = StubTools()
            agent.tools.call = stub.call  # type: ignore[method-assign]
            turn = agent.run(utterance)
        self.assertNotIn("SKU08", utterance)
        self.assertEqual(stub.calls[0], ("lookup_faq", {"question": "库存说明"}))
        self.assertEqual(stub.calls[1], ("lookup_product", {"query": "SKU08"}))
        self.assertEqual([call.name for call in turn.tool_calls], ["lookup_faq", "lookup_product"])
        notes = [event.detail for event in turn.trace if event.action == "拆解"]
        self.assertTrue(any("库存说明" in detail for detail in notes))
        self.assertTrue(any("SKU08" in detail and "观察" in detail for detail in notes))
        self.assertIn("SKU08", client.seen[1][-1]["content"])
        self.assertTrue(turn.used_llm)

    def test_one_tool_per_think_even_if_model_lists_several(self) -> None:
        seen_assistant: list[dict] = []

        def responder(messages):
            tool_msgs = [item for item in messages if item.get("role") == "tool"]
            if not tool_msgs:
                return parse_message(
                    {
                        "role": "assistant",
                        "content": "我想两步一起做",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {
                                    "name": "lookup_faq",
                                    "arguments": json.dumps({"question": "满多少包邮"}, ensure_ascii=False),
                                },
                            },
                            {
                                "id": "call_b",
                                "type": "function",
                                "function": {
                                    "name": "lookup_product",
                                    "arguments": json.dumps({"query": "洁面"}, ensure_ascii=False),
                                },
                            },
                        ],
                    }
                )
            assistant = [item for item in messages if item.get("role") == "assistant"][-1]
            seen_assistant.append(assistant)
            return _final_message("演示店满99包邮，下单页以结算为准。")

        with isolated_runs():
            turn = CustomerServiceAgent(
                planner="llm",
                llm_client=ScriptedClient(responder),
                llm_config=LLMConfig("test", "http://127.0.0.1:9", DEFAULT_MODEL_ID),
            ).run("满多少包邮")
        self.assertEqual(len(seen_assistant[0].get("tool_calls") or []), 1)
        self.assertEqual([call.name for call in turn.tool_calls], ["lookup_faq"])
        self.assertNotIn("lookup_product", [call.name for call in turn.tool_calls])

    def test_session_memory_reaches_the_model_and_the_offline_followup(self) -> None:
        captured: list[list[dict]] = []

        def responder(messages):
            captured.append(messages)
            return _final_message("澄光氨基酸洁面乳是 69 元。")

        with isolated_runs():
            agent = CustomerServiceAgent(planner="offline")
            agent.run("敏感肌能用洁面吗", session_id="mem")
            follow = agent.run("那款多少钱", session_id="mem")
            CustomerServiceAgent(
                planner="llm",
                llm_client=ScriptedClient(responder),
                llm_config=LLMConfig("test", "http://127.0.0.1:9", DEFAULT_MODEL_ID),
            ).run("那款多少钱", session_id="mem")
            alone = CustomerServiceAgent(planner="offline").run("那款多少钱")
        self.assertIn("69", follow.reply)
        self.assertTrue(any(call.name == "lookup_product" and call.arguments.get("query") == "SKU01" for call in follow.tool_calls))
        self.assertNotIn("69", alone.reply)
        blob = json.dumps(captured[0], ensure_ascii=False)
        self.assertIn("敏感肌能用洁面吗", blob)
        self.assertIn("SKU01", blob)

    def test_guardrail_max_iterations_timeout_and_failure(self) -> None:
        def always_faq(messages):
            del messages
            return _tool_message("lookup_faq", {"question": "满多少包邮"}, "再查一次", "call_faq")

        class Boom:
            def complete(self, messages, tools, timeout):
                del messages, tools, timeout
                raise PlannerError("timeout")

        with isolated_runs():
            limited = CustomerServiceAgent(
                planner="llm",
                llm_client=ScriptedClient(always_faq),
                llm_config=LLMConfig("test", "http://127.0.0.1:9", DEFAULT_MODEL_ID),
                max_iterations=2,
            ).run("满多少包邮")
            timed = CustomerServiceAgent(planner="offline", timeout_s=0).run("满多少包邮")
            failed = CustomerServiceAgent(
                planner="llm",
                llm_client=Boom(),
                llm_config=LLMConfig("test", "http://127.0.0.1:9", DEFAULT_MODEL_ID),
            ).run("满多少包邮")
        self.assertTrue(limited.flags.get("max_iterations"))
        self.assertEqual(len(limited.tool_calls), 2)
        self.assertTrue(limited.reply)
        self.assertTrue(any(event.action == "停止" for event in limited.trace))
        self.assertTrue(timed.flags.get("timed_out"))
        self.assertEqual(timed.tool_calls, [])
        self.assertTrue(timed.reply)
        self.assertFalse(failed.used_llm)
        self.assertEqual(failed.llm_fallback, "call_failed")
        self.assertIn("99", failed.reply)
        self.assertTrue(any("模型调用失败" in event.detail for event in failed.trace))

    def test_deepseek_http_client_sends_flash_and_tools(self) -> None:
        seen: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or "0")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                seen.append(body)
                tool_turns = sum(1 for item in body["messages"] if item.get("role") == "tool")
                if tool_turns == 0:
                    message = {
                        "role": "assistant",
                        "content": "先查包邮",
                        "tool_calls": [
                            {
                                "id": "call_live",
                                "type": "function",
                                "function": {
                                    "name": "lookup_faq",
                                    "arguments": json.dumps({"question": "满多少包邮"}, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                else:
                    message = {"role": "assistant", "content": "演示店满99包邮，下单页以结算为准。"}
                payload = json.dumps({"choices": [{"message": message}]}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt: str, *args) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        old = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with isolated_runs():
                turn = CustomerServiceAgent(
                    planner="llm",
                    llm_config=LLMConfig("test-key", f"http://127.0.0.1:{port}", DEFAULT_MODEL_ID),
                ).run("满多少包邮")
        finally:
            if old is not None:
                os.environ["DEEPSEEK_API_KEY"] = old
            server.shutdown()
            server.server_close()
        self.assertGreaterEqual(len(seen), 2)
        self.assertEqual(seen[0]["model"], "deepseek-flash")
        self.assertEqual(seen[0]["thinking"], {"type": "disabled"})
        names = [item["function"]["name"] for item in seen[0]["tools"]]
        self.assertEqual(names, ["lookup_faq", "lookup_product", "escalate_to_human"])
        self.assertTrue(any(item.get("role") == "tool" for item in seen[1]["messages"]))
        self.assertIn("F03", json.dumps(seen[1]["messages"], ensure_ascii=False))
        self.assertTrue(turn.used_llm)
        self.assertEqual(turn.planner, "deepseek-flash")
        self.assertEqual([call.name for call in turn.tool_calls], ["lookup_faq"])
        self.assertIn("99", turn.reply)


if __name__ == "__main__":
    unittest.main()

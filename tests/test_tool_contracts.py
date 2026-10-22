"""Tool schemas, rejected arguments, stable empty results, escalation tickets."""

from __future__ import annotations

import json
import time
import unittest

from chenguang_agent.kb import load_kb
from chenguang_agent.schemas import TOOL_SCHEMAS, validate_tool_arguments
from chenguang_agent.tools import Toolbelt
from tests.support import isolated_runs


class ToolContractTest(unittest.TestCase):
    def test_schemas_cover_the_three_tools(self) -> None:
        self.assertEqual(set(TOOL_SCHEMAS), {"lookup_product", "lookup_faq", "escalate_to_human"})
        encoded = json.dumps(TOOL_SCHEMAS)
        self.assertIn("lookup_product", encoded)
        self.assertIsNone(validate_tool_arguments(TOOL_SCHEMAS["lookup_faq"], {"question": "包邮"}))
        self.assertIsNone(validate_tool_arguments(TOOL_SCHEMAS["escalate_to_human"], {"reason": "投诉"}))

    def test_illegal_arguments_are_rejected(self) -> None:
        with isolated_runs():
            belt = Toolbelt(load_kb())
            cases = [
                ("lookup_product", {"query": 1}),
                ("lookup_product", {}),
                ("lookup_product", {"query": "洁面", "extra": True}),
                ("lookup_product", {"query": "   "}),
                ("lookup_faq", {"question": None}),
                ("lookup_faq", []),
                ("escalate_to_human", {"reason": ""}),
                ("escalate_to_human", {"reason": "  "}),
                ("escalate_to_human", {}),
                ("not_a_tool", {"query": "洁面"}),
            ]
            for name, arguments in cases:
                with self.subTest(name=name, arguments=arguments):
                    output = belt.call(name, arguments)  # type: ignore[arg-type]
                    self.assertFalse(output["ok"])
                    self.assertIn(output["error"]["code"], {"invalid_arguments", "unknown_tool"})
                    self.assertNotIn("Traceback", json.dumps(output, ensure_ascii=False))
            direct = belt.lookup_product(12)  # type: ignore[arg-type]
            self.assertFalse(direct["ok"])
            direct_faq = belt.lookup_faq("")
            self.assertFalse(direct_faq["ok"])
            direct_ticket = belt.escalate_to_human("")
            self.assertFalse(direct_ticket["ok"])
            self.assertNotIn("ticket_id", direct_ticket)

    def test_empty_results_keep_a_stable_shape(self) -> None:
        belt = Toolbelt(load_kb())
        product = belt.call("lookup_product", {"query": "zzzzzz-not-a-sku"})
        again = belt.call("lookup_product", {"query": "yyyyyy-not-a-sku"})
        self.assertEqual(set(product), set(again))
        self.assertEqual(product, {"found": False, "query": "zzzzzz-not-a-sku", "matches": []})
        faq = belt.call("lookup_faq", {"question": "zzzzzz完全无关的句子"})
        faq_again = belt.call("lookup_faq", {"question": "yyyyyy也不相关的句子"})
        self.assertEqual(set(faq), set(faq_again))
        self.assertFalse(faq["found"])
        self.assertEqual(faq["q_id"], "")
        self.assertEqual(faq["answer"], "")
        self.assertEqual(faq["matched"], [])
        self.assertEqual(faq["alternatives"], [])
        self.assertEqual(faq["score"], 0)
        self.assertFalse(faq["bot_may_answer"])

    def test_escalate_requires_reason_and_returns_ticket_id(self) -> None:
        with isolated_runs() as root:
            belt = Toolbelt(load_kb())
            belt.bind_context(user_text="你们就是骗子，我要投诉", session_id="s", turn_id="t")
            first = belt.call("escalate_to_human", {"reason": "命中投诉"})
            second = Toolbelt(load_kb()).call("escalate_to_human", {"reason": "另一条"})
            self.assertTrue(first["ticket_id"])
            self.assertEqual(first["reason"], "命中投诉")
            self.assertNotEqual(first["ticket_id"], second["ticket_id"])
            self.assertTrue(first["ticket_id"].startswith("CG-"))
            rows = [
                json.loads(line)
                for line in (root / "tickets.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            saved = next(row for row in rows if row["ticket_id"] == first["ticket_id"])
            self.assertEqual(saved["reason"], "命中投诉")
            self.assertIn("骗子", saved["summary"])
            self.assertTrue(saved["created_at"])
            self.assertEqual(saved["status"], "open")
            self.assertEqual(saved["session_id"], "s")
            self.assertEqual(saved["turn_id"], "t")

    def test_failure_and_timeout_are_structured(self) -> None:
        belt = Toolbelt(load_kb())

        def boom(query: str) -> dict:
            raise RuntimeError("disk")

        belt.lookup_product = boom  # type: ignore[method-assign]
        failed = belt.call("lookup_product", {"query": "洁面"})
        self.assertEqual(failed["error"]["code"], "tool_failure")
        self.assertNotIn("Traceback", failed["error"]["message"])

        def slow(query: str) -> dict:
            time.sleep(0.35)
            return {"found": False, "query": query, "matches": []}

        belt.lookup_product = slow  # type: ignore[method-assign]
        timed = belt.call("lookup_product", {"query": "洁面"}, timeout_s=0.05)
        self.assertEqual(timed["error"]["code"], "timeout")


if __name__ == "__main__":
    unittest.main()

"""Holds the checker and the offline specialist. Scheduling lives in the agent loop."""

from __future__ import annotations

from chenguang_agent.checker import Checker
from chenguang_agent.guardrails import Guardrails
from chenguang_agent.llm import LLMConfig
from chenguang_agent.loop import AgentLoop
from chenguang_agent.models import Turn
from chenguang_agent.planner import OfflinePlanner
from chenguang_agent.router import Router
from chenguang_agent.specialist import Specialist
from chenguang_agent.tools import Toolbelt


class Coordinator:
    def __init__(
        self,
        *,
        router: Router,
        guardrails: Guardrails,
        tools: Toolbelt,
        use_llm: bool = False,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.router = router
        self.guardrails = guardrails
        self.tools = tools
        self.use_llm = use_llm
        self.llm_config = llm_config
        self.specialist = Specialist(tools, guardrails)
        self.checker = Checker(guardrails)

    def handle(self, user_text: str) -> Turn:
        """Offline loop. The product path with a model is CustomerServiceAgent.run."""

        planner = OfflinePlanner(self.router, self.guardrails, self.tools)
        return AgentLoop(
            planner=planner,
            fallback=planner,
            tools=self.tools,
            guardrails=self.guardrails,
            checker=self.checker,
        ).run(user_text, [])

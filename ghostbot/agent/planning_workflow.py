"""Multi-stage planning workflow for structured feature development."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WorkflowPhase:
    name: str
    prompt_template: str
    tools_used: list[str] = field(default_factory=list)
    result_text: str = ""


class PlanningWorkflow:
    """Runs a lightweight 5-stage planning workflow."""

    def __init__(
        self,
        loop,
        session,
        request: str,
        history: list[dict[str, Any]],
        channel: str,
        chat_id: str,
        session_summary: str | None = None,
    ):
        self.loop = loop
        self.session = session
        self.request = request
        self.history = history
        self.channel = channel
        self.chat_id = chat_id
        self.session_summary = session_summary
        self.phases: list[WorkflowPhase] = []

    async def run(self) -> tuple[str, list[str]]:
        all_tools: list[str] = []

        discovery_text = f"User request: {self.request}\n\nSummarize what is being asked and identify obvious constraints."
        )
        self.phases.append(WorkflowPhase(
            name="Discovery",
            prompt_template="inline_discovery",
            result_text=discovery_text,
        ))

        exploration_plan, tools_used = await self.loop._run_planning_loop(
            request=self.request,
            history=self.history,
            channel=self.channel,
            chat_id=self.chat_id,
            session_summary=self.session_summary,
            execution_mode="executable",
            session=self.session,
        )
        all_tools.extend(tools_used)
        self.phases.append(WorkflowPhase(
            name="Exploration",
            prompt_template="planning_mode.md",
            tools_used=tools_used,
            result_text=exploration_plan,
        ))

        clarification_text = (
            f"Understood request: {self.request}\n\n"
            "Based on exploration, identify ambiguities and list clarifying questions.\n\n"
            f"Exploration summary:\n{exploration_plan[:1200]}"
        )
        )
        self.phases.append(WorkflowPhase(
            name="Clarification",
            prompt_template="planning_clarify.md",
            result_text=clarification_text,
        ))

        architecture_prompt = (
            f"User request: {self.request}\n\n"
            f"Exploration findings:\n{exploration_plan[:1600]}\n\n"
            "Compare 2 approaches and recommend one."
        )
        )
        architecture_text, arch_tools = await self.loop._run_planning_loop(
            request=architecture_prompt,
            history=self.history,
            channel=self.channel,
            chat_id=self.chat_id,
            session_summary=self.session_summary,
            execution_mode="executable",
            session=self.session,
        )
        all_tools.extend(arch_tools)
        self.phases.append(WorkflowPhase(
            name="Architecture",
            prompt_template="planning_architecture.md",
            tools_used=arch_tools,
            result_text=architecture_text,
        ))

        final_request = (
            f"{self.request}\n\n"
            f"Architecture guidance:\n{architecture_text}"
        )
        )
        plan_text, final_tools = await self.loop._run_planning_loop(
            request=final_request,
            history=self.history,
            channel=self.channel,
            chat_id=self.chat_id,
            session_summary=self.session_summary,
            execution_mode="executable",
            session=self.session,
        )
        all_tools.extend(final_tools)
        self.phases.append(WorkflowPhase(
            name="Plan Generation",
            prompt_template="planning_mode.md",
            tools_used=final_tools,
            result_text=plan_text,
        ))

        return plan_text, all_tools

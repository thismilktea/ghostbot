"""Trace recording hook for GhostBot evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ghostbot.agent.hook import AgentHook, AgentHookContext


class EvalTraceHook(AgentHook):
    def __init__(self, trace_path: Path):
        super().__init__()
        self.trace_path = trace_path
        self.iterations: list[dict[str, Any]] = []

    async def after_iteration(self, context: AgentHookContext) -> None:
        self.iterations.append({
            "iteration": context.iteration,
            "usage": dict(context.usage),
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": dict(tc.arguments or {})}
                for tc in context.tool_calls
            ],
            "tool_events": [dict(event) for event in context.tool_events],
            "final_content": context.final_content,
            "stop_reason": context.stop_reason,
            "error": context.error,
        })

    def write(self) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text(
            json.dumps({"iterations": self.iterations}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

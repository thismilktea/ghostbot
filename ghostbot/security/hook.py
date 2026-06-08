"""Security pattern hook — scans file writes for known-dangerous patterns."""
from __future__ import annotations

from typing import Any

from loguru import logger

from ghostbot.agent.hook import AgentHook, AgentHookContext
from ghostbot.security.patterns import SecurityFinding, check_patterns


_WRITE_TOOLS = frozenset({"write_file", "edit_file", "notebook_edit"})


class SecurityPatternHook(AgentHook):
    """Scans write_file/edit_file tool calls for security anti-patterns.

    When a dangerous pattern is detected, injects a warning into the
    tool call arguments so the runner can surface it as context.
    Findings are accumulated per-turn for reporting.
    """

    def __init__(self) -> None:
        super().__init__()
        self._turn_findings: list[SecurityFinding] = []

    @property
    def turn_findings(self) -> list[SecurityFinding]:
        return list(self._turn_findings)

    def reset_turn(self) -> None:
        self._turn_findings.clear()

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for tool_call in context.tool_calls:
            if tool_call.name not in _WRITE_TOOLS:
                continue
            args = tool_call.arguments or {}
            file_path = str(args.get("path") or args.get("file_path") or "")
            content = str(args.get("content") or args.get("new_content") or "")
            if not file_path or not content:
                continue
            findings = check_patterns(file_path, content)
            if findings:
                self._turn_findings.extend(findings)
                warning_lines = [
                    f"[Security Warning] {len(findings)} pattern(s) detected in {file_path}:"
                ]
                for f in findings:
                    warning_lines.append(
                        f"  - [{f.severity.upper()}] {f.rule_name} (line {f.line_number}): {f.message}"
                    )
                warning = "\n".join(warning_lines)
                logger.warning(warning)

    async def after_iteration(self, context: AgentHookContext) -> None:
        if self._turn_findings:
            summary = f"[SecurityPatternHook] {len(self._turn_findings)} finding(s) this turn"
            logger.info(summary)

    def get_findings_summary(self) -> str | None:
        if not self._turn_findings:
            return None
        lines = [f"Security scan: {len(self._turn_findings)} finding(s)"]
        for f in self._turn_findings:
            lines.append(f"  [{f.severity}] {f.rule_name} in {f.file_path}:{f.line_number}")
        return "\n".join(lines)

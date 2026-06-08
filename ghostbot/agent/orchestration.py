"""Orchestration primitives for multi-agent pipelines."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ghostbot.agent.subagent import SubagentManager


@dataclass(slots=True)
class Finding:
    """A single finding from a finder or verifier agent."""

    description: str
    source: str = ""
    confidence: float = 0.8
    severity: str = "medium"
    verified: bool | None = None
    verification_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrchestrationResult:
    """Result of a FindVerifyFilter pipeline run."""

    findings: list[Finding] = field(default_factory=list)
    verified: list[Finding] = field(default_factory=list)
    rejected: list[Finding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class FindVerifyFilter:
    """Reusable find→verify→filter orchestration pattern.

    Usage:
        fvf = FindVerifyFilter(subagent_manager)
        result = await fvf.run(
            finder_prompts=["Find bugs in auth module", "Find bugs in API layer"],
            verifier_prompt_template="Verify this finding is real: {description}",
            finder_model="fast",
            verifier_model="strong",
        )
        confirmed = result.verified
    """

    def __init__(
        self,
        manager: "SubagentManager",
        *,
        confidence_threshold: float = 0.6,
        max_parallel_finders: int = 4,
        max_parallel_verifiers: int = 6,
    ):
        self._manager = manager
        self._confidence_threshold = confidence_threshold
        self._max_parallel_finders = max_parallel_finders
        self._max_parallel_verifiers = max_parallel_verifiers

    async def run(
        self,
        finder_prompts: list[str],
        verifier_prompt_template: str = "Verify whether this finding is a real issue. Respond with CONFIRMED or REJECTED followed by a brief explanation.\n\nFinding: {description}\nSource: {source}",
        finder_model: str | None = None,
        verifier_model: str | None = None,
        parse_findings: Any = None,
    ) -> OrchestrationResult:
        """Run the full find→verify→filter pipeline."""
        result = OrchestrationResult()

        raw_findings = await self._find_phase(finder_prompts, finder_model, parse_findings)
        result.findings = raw_findings

        if not raw_findings:
            return result

        verified = await self._verify_phase(raw_findings, verifier_prompt_template, verifier_model)

        for finding in verified:
            if finding.verified:
                result.verified.append(finding)
            else:
                result.rejected.append(finding)

        return result

    async def _find_phase(
        self,
        prompts: list[str],
        model: str | None,
        parse_findings: Any = None,
    ) -> list[Finding]:
        """Launch finder agents in parallel and collect findings."""
        semaphore = asyncio.Semaphore(self._max_parallel_finders)

        async def _run_one(prompt: str) -> list[Finding]:
            async with semaphore:
                try:
                    text = await self._spawn_and_wait(prompt, model)
                    if parse_findings is not None:
                        return parse_findings(text)
                    return self._default_parse_findings(text)
                except Exception as e:
                    logger.warning("Finder agent failed: {}", e)
                    return []

        tasks = [asyncio.create_task(_run_one(p)) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_findings: list[Finding] = []
        for r in results:
            if isinstance(r, list):
                all_findings.extend(r)
            elif isinstance(r, Exception):
                logger.warning("Finder task exception: {}", r)
        return all_findings

    async def _verify_phase(
        self,
        findings: list[Finding],
        prompt_template: str,
        model: str | None,
    ) -> list[Finding]:
        """Launch verifier agents in parallel for each finding."""
        semaphore = asyncio.Semaphore(self._max_parallel_verifiers)

        async def _verify_one(finding: Finding) -> Finding:
            async with semaphore:
                try:
                    prompt = prompt_template.format(
                        description=finding.description,
                        source=finding.source,
                        severity=finding.severity,
                        confidence=finding.confidence,
                    )
                    text = await self._spawn_and_wait(prompt, model)
                    upper = text.upper().strip()
                    if upper.startswith("CONFIRMED"):
                        finding.verified = True
                        finding.verification_note = text
                    elif upper.startswith("REJECTED"):
                        finding.verified = False
                        finding.verification_note = text
                    else:
                        finding.verified = finding.confidence >= self._confidence_threshold
                        finding.verification_note = f"Ambiguous response, defaulted by confidence ({finding.confidence}): {text[:200]}"
                except Exception as e:
                    logger.warning("Verifier agent failed for '{}': {}", finding.description[:50], e)
                    finding.verified = finding.confidence >= self._confidence_threshold
                    finding.verification_note = f"Verification failed: {e}"
                return finding

        tasks = [asyncio.create_task(_verify_one(f)) for f in findings]
        return await asyncio.gather(*tasks)

    async def _spawn_and_wait(self, prompt: str, model: str | None) -> str:
        """Spawn a subagent synchronously and return its text result."""
        from ghostbot.agent.runner import AgentRunSpec
        from ghostbot.agent.tools.registry import ToolRegistry
        from ghostbot.agent.tools.filesystem import ReadFileTool, ListDirTool
        from ghostbot.agent.tools.search import GlobTool, GrepTool
        from ghostbot.agent.skills import BUILTIN_SKILLS_DIR

        resolved_model = self._manager.resolve_model(model)
        tools = ToolRegistry()
        workspace = self._manager.workspace
        allowed_dir = workspace if self._manager.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        tools.register(ReadFileTool(workspace=workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        tools.register(ListDirTool(workspace=workspace, allowed_dir=allowed_dir))
        tools.register(GlobTool(workspace=workspace, allowed_dir=allowed_dir))
        tools.register(GrepTool(workspace=workspace, allowed_dir=allowed_dir))

        messages = [
            {"role": "system", "content": "You are a focused analysis agent. Complete the task and return your findings concisely."},
            {"role": "user", "content": prompt},
        ]

        result = await self._manager.runner.run(AgentRunSpec(
            initial_messages=messages,
            tools=tools,
            model=resolved_model,
            max_iterations=10,
            max_tool_result_chars=self._manager.max_tool_result_chars,
            fail_on_tool_error=True,
            mode="analysis",
            allow_background_tools=False,
            allow_external_tools=False,
            allowed_tool_names={"read_file", "list_dir", "glob", "grep"},
            blocked_tool_prefixes=("mcp_",),
        ))
        return result.final_content or ""

    @staticmethod
    def _default_parse_findings(text: str) -> list[Finding]:
        """Parse findings from agent text output using simple heuristics."""
        findings: list[Finding] = []
        if not text.strip():
            return findings

        lines = text.strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ", "• ")):
                line = line[2:].strip()
            if len(line) < 10:
                continue
            severity = "medium"
            if any(w in line.lower() for w in ("critical", "severe", "dangerous")):
                severity = "high"
            elif any(w in line.lower() for w in ("minor", "low", "nitpick")):
                severity = "low"
            findings.append(Finding(
                description=line,
                source="finder_agent",
                severity=severity,
            ))
        return findings

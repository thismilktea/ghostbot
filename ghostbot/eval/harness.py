"""Evaluation harness for GhostBot dogfooding suites."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from ghostbot import __version__
from ghostbot.agent.loop import AgentLoop
from ghostbot.bus.queue import MessageBus
from ghostbot.eval.fixtures import FixturePreparer
from ghostbot.eval.judge import judge_scenario_result
from ghostbot.eval.metrics import attach_result_metrics, attach_summary
from ghostbot.eval.report import render_summary_markdown
from ghostbot.eval.schema import (
    DeterministicCheckResult,
    EvalAgentSummary,
    EvalCheck,
    EvalCommandResult,
    EvalFileSummary,
    EvalRunSummary,
    EvalScenario,
    EvalScenarioResult,
    EvalSuite,
    EvalTiming,
    EvalToolSummary,
)
from ghostbot.eval.trace import EvalTraceHook
from ghostbot.interface import _make_provider


def load_suite(path: Path) -> EvalSuite:
    return EvalSuite.model_validate(json.loads(path.read_text(encoding="utf-8")))


class EvalHarness:
    def __init__(self, config, suite_path: Path, out_dir: Path):
        self.config = config
        self.suite_path = suite_path
        self.out_dir = out_dir
        self.suite = load_suite(suite_path)
        self.preparer = FixturePreparer(suite_path.parent.parent)

    def list_scenarios(self) -> list[EvalScenario]:
        return list(self.suite.scenarios)

    async def run(self, scenario_ids: list[str] | None = None) -> EvalRunSummary:
        selected = [
            scenario for scenario in self.suite.scenarios
            if not scenario_ids or scenario.id in scenario_ids
        ]
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_root = self.out_dir / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        results: list[EvalScenarioResult] = []
        for scenario in selected:
            results.append(await self._run_scenario(run_id, run_root, scenario))
        summary = EvalRunSummary(
            run_id=run_id,
            suite_name=self.suite.name,
            generated_at=datetime.now(UTC),
            results=results,
        )
        attach_summary(summary)
        (run_root / "suite.json").write_text(
            self.suite.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (run_root / "results.json").write_text(
            summary.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (run_root / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
        return summary

    async def _run_scenario(self, run_id: str, run_root: Path, scenario: EvalScenario) -> EvalScenarioResult:
        fixture = self.suite.fixtures[scenario.fixture]
        workspace = self.preparer.prepare(scenario.id.replace("/", "_"), fixture, run_root)
        trace_path = run_root / "scenarios" / f"{scenario.id}.trace.json"
        trace = EvalTraceHook(trace_path)
        provider = _make_provider(self.config)
        mcp_servers = {} if self.suite.defaults.disable_mcp else self.config.tools.mcp_servers
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=workspace,
            model=self.config.agents.defaults.model,
            max_iterations=scenario.expected.max_tool_calls or self.suite.defaults.max_iterations,
            context_window_tokens=self.config.agents.defaults.context_window_tokens,
            context_block_limit=self.config.agents.defaults.context_block_limit,
            approved_plan_context_block_limit=self.config.agents.defaults.approved_plan_context_block_limit,
            max_tool_result_chars=self.config.agents.defaults.max_tool_result_chars,
            provider_retry_mode=self.config.agents.defaults.provider_retry_mode,
            web_config=self.config.tools.web,
            exec_config=self.config.tools.exec,
            restrict_to_workspace=self.config.tools.restrict_to_workspace,
            mcp_servers=mcp_servers,
            timezone=self.config.agents.defaults.timezone,
            unified_session=self.config.agents.defaults.unified_session,
            disabled_skills=self.config.agents.defaults.disabled_skills,
            session_ttl_minutes=self.config.agents.defaults.session_ttl_minutes,
            planning_config=self.config.agents.defaults.planning,
        )
        loop._extra_hooks = [trace]
        baseline_snapshot = self._snapshot_files(workspace)
        start = datetime.now(UTC)
        started_perf = perf_counter()
        status = "pass"
        error = None
        response = None
        try:
            prompt = scenario.prompt or (scenario.turns[0].input if scenario.turns else "")
            response = await loop.process_direct(prompt, session_key=f"eval:{scenario.id}")
        except Exception as exc:
            status = "error"
            error = str(exc)
        ended_perf = perf_counter()
        end = datetime.now(UTC)
        trace.write()
        final_snapshot = self._snapshot_files(workspace)
        changed_files, added_files, deleted_files = self._diff_snapshots(baseline_snapshot, final_snapshot)
        all_changed_files = sorted(set(changed_files) | set(added_files) | set(deleted_files))
        final_content = (response.content if response else None) or ""
        tool_call_count = sum(len(item.get("tool_calls", [])) for item in trace.iterations)
        check_results = await self._run_checks(workspace, scenario.expected.checks)
        failures, scope_violations = self._run_deterministic_checks(
            scenario=scenario,
            final_content=final_content,
            changed_files=all_changed_files,
            tool_call_count=tool_call_count,
            check_results=check_results,
        )
        redundant_calls = self._find_redundant_tool_calls(trace.iterations)
        self_correction_rate, recovery_cost = self._measure_self_correction(trace.iterations)
        if status != "error" and failures:
            status = "fail"
        result = EvalScenarioResult(
            run_id=run_id,
            scenario_id=scenario.id,
            category=scenario.category,
            fixture=scenario.fixture,
            model=f"{self.config.agents.defaults.model} ({__version__})",
            status=status,
            timing=EvalTiming(
                started_at=start,
                ended_at=end,
                latency_ms=int((ended_perf - started_perf) * 1000),
            ),
            agent=EvalAgentSummary(
                stop_reason=getattr(response, "metadata", {}).get("stop_reason", "completed") if response else "error",
                iterations=len(trace.iterations),
                final_content=final_content,
                error=error,
            ),
            usage=self._usage_from_trace(trace.iterations),
            tools=EvalToolSummary(
                tools_used=sorted({call["name"] for item in trace.iterations for call in item.get("tool_calls", [])}),
                tool_call_count=tool_call_count,
                redundant_tool_call_count=len(redundant_calls),
                failed_tool_call_count=sum(
                    1 for item in trace.iterations for event in item.get("tool_events", []) if event.get("status") == "error"
                ),
                redundant_calls=redundant_calls,
            ),
            files=EvalFileSummary(
                changed_files=all_changed_files,
                added_files=added_files,
                deleted_files=deleted_files,
                scope_violations=scope_violations,
            ),
            checks=check_results,
            deterministic=DeterministicCheckResult(passed=not failures, failures=failures),
            trace_file=str(trace_path),
        )
        attach_result_metrics(
            result,
            requirement_hit_rate_value=self._measure_requirement_hit_rate(
                scenario,
                final_content,
                all_changed_files,
                check_results,
            ),
            self_correction_rate_value=self_correction_rate,
            recovery_cost_value=recovery_cost,
        )
        if scenario.judging.rubric:
            result.judge = await judge_scenario_result(
                provider,
                self.config.agents.defaults.model,
                scenario,
                result,
            )
            if result.judge.passed is False and result.status == "pass":
                result.status = "fail"
        scenario_path = run_root / "scenarios" / f"{scenario.id}.json"
        scenario_path.parent.mkdir(parents=True, exist_ok=True)
        scenario_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    @staticmethod
    def _snapshot_files(root: Path) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(root)).replace("\\", "/")
            if EvalHarness._should_ignore_path(relative):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            snapshot[relative] = digest
        return snapshot

    @staticmethod
    def _should_ignore_path(relative_path: str) -> bool:
        return relative_path.startswith("projects/")

    @staticmethod
    def _diff_snapshots(
        before: dict[str, str],
        after: dict[str, str],
    ) -> tuple[list[str], list[str], list[str]]:
        before_paths = set(before)
        after_paths = set(after)
        changed = sorted(path for path in before_paths & after_paths if before[path] != after[path])
        added = sorted(after_paths - before_paths)
        deleted = sorted(before_paths - after_paths)
        return changed, added, deleted

    @staticmethod
    async def _run_checks(workspace: Path, checks: list[EvalCheck]) -> list[EvalCommandResult]:
        results: list[EvalCommandResult] = []
        for check in checks:
            cwd = workspace / check.cwd if check.cwd else workspace
            command = EvalHarness._resolve_check_command(check.command)
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=check.timeout_seconds,
                )
                exit_code = process.returncode or 0
            except TimeoutError:
                process.kill()
                stdout, stderr = await process.communicate()
                exit_code = -1
                stderr = (stderr or b"") + b"\nTimed out"
            results.append(EvalCommandResult(
                name=check.name,
                command=command,
                cwd=str(cwd.relative_to(workspace)).replace("\\", "/") if cwd != workspace else ".",
                exit_code=exit_code,
                passed=exit_code == 0,
                stdout=(stdout or b"").decode("utf-8", errors="replace"),
                stderr=(stderr or b"").decode("utf-8", errors="replace"),
            ))
        return results

    @staticmethod
    def _resolve_check_command(command: str) -> str:
        stripped = command.strip()
        if stripped.startswith("python -m "):
            module_args = stripped[len("python -m "):]
            executable = shlex.quote(sys.executable)
            return f"{executable} -m {module_args}"
        return command

    @staticmethod
    def _run_deterministic_checks(
        scenario: EvalScenario,
        final_content: str,
        changed_files: list[str],
        tool_call_count: int,
        check_results: list[EvalCommandResult],
    ) -> tuple[list[str], list[str]]:
        expected = scenario.expected
        failures: list[str] = []
        scope_violations: list[str] = []
        if expected.must_not_modify_files and changed_files:
            failures.append("Scenario must not modify files")
        for phrase in expected.required_phrases:
            if phrase not in final_content:
                failures.append(f"Missing required phrase: {phrase}")
        for phrase in expected.forbidden_phrases:
            if phrase in final_content:
                failures.append(f"Contains forbidden phrase: {phrase}")
        if expected.max_tool_calls is not None and tool_call_count > expected.max_tool_calls:
            failures.append("Tool call budget exceeded")
        if expected.max_changed_files is not None and len(changed_files) > expected.max_changed_files:
            failures.append("Changed file budget exceeded")
        missing_required = [path for path in expected.required_changed_files if path not in changed_files]
        if missing_required:
            failures.extend(f"Missing required changed file: {path}" for path in missing_required)
        forbidden_hits = [path for path in changed_files if path in expected.forbidden_changed_files]
        if forbidden_hits:
            scope_violations.extend(forbidden_hits)
            failures.extend(f"Modified forbidden file: {path}" for path in forbidden_hits)
        if expected.allowed_changed_files:
            out_of_scope = [path for path in changed_files if path not in expected.allowed_changed_files]
            if out_of_scope:
                scope_violations.extend(out_of_scope)
                failures.extend(f"Modified file outside allowed scope: {path}" for path in out_of_scope)
        failed_checks = [check for check in check_results if not check.passed]
        if failed_checks:
            failures.extend(f"Check failed: {check.name}" for check in failed_checks)
        return failures, sorted(set(scope_violations))

    @staticmethod
    def _measure_requirement_hit_rate(
        scenario: EvalScenario,
        final_content: str,
        changed_files: list[str],
        check_results: list[EvalCommandResult],
    ) -> float:
        matched = 0
        total = 0
        for phrase in scenario.expected.required_phrases:
            total += 1
            if phrase in final_content:
                matched += 1
        for path in scenario.expected.required_changed_files:
            total += 1
            if path in changed_files:
                matched += 1
        for criterion in scenario.expected.acceptance_criteria:
            total += 1
            if criterion in final_content:
                matched += 1
        for check in check_results:
            total += 1
            if check.passed:
                matched += 1
        if total == 0:
            return 1.0
        return matched / total

    @staticmethod
    def _find_redundant_tool_calls(iterations: list[dict]) -> list[str]:
        signatures: list[str] = []
        for item in iterations:
            for call in item.get("tool_calls", []):
                name = call.get("name") or ""
                arguments = json.dumps(call.get("arguments") or {}, sort_keys=True, ensure_ascii=False)
                signatures.append(f"{name}:{arguments}")
        counts = Counter(signatures)
        return sorted(signature for signature, count in counts.items() if count > 1)

    @staticmethod
    def _measure_self_correction(iterations: list[dict]) -> tuple[float, int]:
        recoverable_failures = 0
        recovered_failures = 0
        recovery_cost = 0
        seen_failure = False
        extra_calls_after_failure = 0
        for item in iterations:
            events = item.get("tool_events", [])
            calls_this_iteration = len(item.get("tool_calls", []))
            if seen_failure:
                extra_calls_after_failure += calls_this_iteration
            if any(event.get("status") == "error" for event in events):
                recoverable_failures += 1
                seen_failure = True
                continue
            if seen_failure and calls_this_iteration > 0:
                recovered_failures += 1
                recovery_cost += extra_calls_after_failure
                seen_failure = False
                extra_calls_after_failure = 0
        if recoverable_failures == 0:
            return 0.0, 0
        return recovered_failures / recoverable_failures, recovery_cost

    @staticmethod
    def _usage_from_trace(iterations: list[dict]) -> dict[str, int]:
        combined: dict[str, int] = {}
        for item in iterations:
            for key, value in (item.get("usage") or {}).items():
                combined[key] = combined.get(key, 0) + int(value or 0)
        return combined

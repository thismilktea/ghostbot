"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from ghostbot.agent.autocompact import AutoCompact
from ghostbot.agent.context import ContextBuilder
from ghostbot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from ghostbot.agent.memory import Consolidator, Dream
from ghostbot.agent.planning import (
    PlanQualityResult,
    PlanState,
    aggregate_plan_checklist,
    build_execution_contract,
    detect_execution_mode,
    exploration_tools_used,
    extract_plan_checklist,
    extract_plan_file_paths,
    plan_execution_options,
    plan_requires_exploration,
    plan_section,
    plan_section_lines,
    should_plan_request,
    validate_plan_quality,
)
from ghostbot.agent.policy import PolicyContext, PolicyEngine
from ghostbot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunSpec, AgentRunner
from ghostbot.agent.subagent import SubagentManager
from ghostbot.agent.tools.cron import CronTool
from ghostbot.agent.skills import BUILTIN_SKILLS_DIR
from ghostbot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from ghostbot.agent.tools.message import MessageTool
from ghostbot.agent.tools.notebook import NotebookEditTool
from ghostbot.agent.tools.registry import ToolRegistry
from ghostbot.agent.tools.search import GlobTool, GrepTool
from ghostbot.agent.tools.shell import ExecTool
from ghostbot.agent.tools.spawn import SpawnTool
from ghostbot.agent.tools.web import WebFetchTool, WebSearchTool
from ghostbot.bus.events import InboundMessage, OutboundMessage
from ghostbot.command import CommandContext, CommandRouter, register_builtin_commands
from ghostbot.bus.queue import MessageBus
from ghostbot.config.schema import AgentDefaults
from ghostbot.providers.base import LLMProvider
from ghostbot.project import DEFAULT_PROJECT_ID, ProjectManager, ProjectState
from ghostbot.session.manager import SessionManager
from ghostbot.utils.helpers import estimate_prompt_tokens_chain, image_placeholder_text, truncate_text as truncate_text_fn
from ghostbot.utils.prompt_templates import render_template
from ghostbot.utils.progress import extract_plan_progress
from ghostbot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

if TYPE_CHECKING:
    from ghostbot.config.schema import ChannelsConfig, ExecToolConfig, PlanningConfig, WebToolsConfig
    from ghostbot.cron.service import CronService


UNIFIED_SESSION_KEY = "unified:default"


class _LoopHook(AgentHook):
    """Core hook for the main loop."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> None:
        super().__init__(reraise=True)
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        from ghostbot.utils.helpers import strip_think

        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean) :]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None
                )
                plan_progress = extract_plan_progress(thought)
                if plan_progress:
                    await self._on_progress(plan_progress, plan_stage=True)
                elif thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            await self._on_progress(tool_hint, tool_hint=True)
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(self._channel, self._chat_id, self._message_id)

    async def after_iteration(self, context: AgentHookContext) -> None:
        u = context.usage or {}
        logger.debug(
            "LLM usage: prompt={} completion={} cached={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_APPROVAL_KEY = "pending_tool_approval"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        approved_plan_context_block_limit: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        web_config: WebToolsConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        session_ttl_minutes: int = 0,
        coding_config: CodingModeConfig | None = None,
        planning_config: PlanningConfig | None = None,
        fast_provider: LLMProvider | None = None,
        fast_model: str | None = None,
        hooks: list[AgentHook] | None = None,
        unified_session: bool = False,
        disabled_skills: list[str] | None = None,
    ):
        from ghostbot.config.schema import CodingModeConfig, ExecToolConfig, WebToolsConfig

        defaults = AgentDefaults()
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.fast_provider = fast_provider or provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.fast_model = fast_model or self.model
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.context_window_tokens = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        self.context_block_limit = context_block_limit
        self.approved_plan_context_block_limit = (
            approved_plan_context_block_limit
            if approved_plan_context_block_limit is not None
            else defaults.approved_plan_context_block_limit
        )
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.web_config = web_config or WebToolsConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.coding_config = coding_config or CodingModeConfig()
        self.planning_config = planning_config or defaults.planning
        if self.coding_config.enable:
            restrict_to_workspace = True
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        self.context = ContextBuilder(workspace, timezone=timezone, disabled_skills=disabled_skills)
        self.projects = ProjectManager(workspace)
        self.sessions = self.projects
        self.tools = ToolRegistry()
        self.policy_engine = PolicyEngine()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_config=self.web_config,
            max_tool_result_chars=self.max_tool_result_chars,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
        )
        self._unified_session = unified_session
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stacks: dict[str, AsyncExitStack] = {}
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # project_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._project_locks: dict[str, asyncio.Lock] = {}
        self._pending_queues: dict[str, asyncio.Queue] = {}
        # GHOSTBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("GHOSTBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.consolidator = Consolidator(
            store=self.context.memory,
            provider=provider,
            model=self.model,
            sessions=self.projects,
            context_window_tokens=self.context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        self.auto_compact = AutoCompact(
            sessions=self.projects,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )
        self.dream = Dream(
            store=self.context.memory,
            provider=provider,
            model=self.model,
        )
        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _allowed_tool_names(self) -> set[str] | None:
        if not self.coding_config.enable:
            return None
        allowed = {"read_file", "list_dir", "glob", "grep"}
        if self.coding_config.allow_write:
            allowed.update({"write_file", "edit_file", "notebook_edit"})
        if self.coding_config.allow_exec:
            allowed.add("exec")
        if self.coding_config.allow_web:
            allowed.update({"web_search", "web_fetch"})
        if self.coding_config.allow_spawn:
            allowed.add("spawn")
        if self.coding_config.allow_cron:
            allowed.add("cron")
        return allowed

    def _blocked_tool_prefixes(self) -> tuple[str, ...]:
        if self.coding_config.enable and not self.coding_config.allow_mcp:
            return ("mcp_",)
        return ()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = (
            self.workspace if (self.restrict_to_workspace or self.exec_config.sandbox) else None
        )
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(
            ReadFileTool(
                workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read
            )
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        for cls in (GlobTool, GrepTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(NotebookEditTool(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    sandbox=self.exec_config.sandbox,
                    path_append=self.exec_config.path_append,
                    allowed_env_keys=self.exec_config.allowed_env_keys,
                )
            )
        if self.web_config.enable:
            self.tools.register(
                WebSearchTool(config=self.web_config.search, proxy=self.web_config.proxy)
            )
            self.tools.register(WebFetchTool(proxy=self.web_config.proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from ghostbot.agent.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stacks = await connect_mcp_servers(self._mcp_servers, self.tools)
            if self._mcp_stacks:
                self._mcp_connected = True
            else:
                logger.warning("No MCP servers connected successfully (will retry next message)")
        except asyncio.CancelledError:
            logger.warning("MCP connection cancelled (will retry next message)")
            self._mcp_stacks.clear()
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            self._mcp_stacks.clear()
        finally:
            self._mcp_connecting = False

    def _allow_active_project_tools(self, project: ProjectState | None, plan: PlanState | None = None) -> None:
        active_project_path = self._active_project_path(project, plan)
        if not active_project_path:
            return
        for name in ("read_file", "write_file", "edit_file", "list_dir", "glob", "grep", "notebook_edit"):
            tool = self.tools.get(name)
            if tool and hasattr(tool, "add_allowed_dir"):
                tool.add_allowed_dir(active_project_path)

    @staticmethod
    def _active_project_path(project: ProjectState | None, plan: PlanState | None = None) -> str | None:
        path = None
        if project is not None:
            path = project.path or project.metadata.get("active_project_path")
        if not path and plan is not None:
            path = plan.active_project_path
        if not path:
            return None
        try:
            return str(Path(str(path)).expanduser().resolve())
        except Exception:
            return str(path)

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from ghostbot.utils.helpers import strip_think

        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hints with smart abbreviation."""
        from ghostbot.utils.tool_hints import format_tool_hints

        return format_tool_hints(tool_calls)

    def _origin_key(self, msg: InboundMessage) -> str:
        return f"{msg.channel}:{msg.chat_id}"

    def _effective_project_key(self, msg: InboundMessage) -> str:
        if msg.session_key_override:
            return msg.session_key_override
        active = self.projects.get_active_for_origin(self._origin_key(msg))
        return active or DEFAULT_PROJECT_ID

    def _effective_session_key(self, msg: InboundMessage) -> str:
        return self._effective_project_key(msg)

    async def _run_planning_loop(
        self,
        *,
        request: str,
        history: list[dict],
        channel: str,
        chat_id: str,
        session_summary: str | None = None,
        revision_feedback: str | None = None,
        previous_plan: str | None = None,
        execution_mode: str = "executable",
        rewrite_quality: PlanQualityResult | None = None,
        tools_used: list[str] | None = None,
        session: ProjectState | None = None,
    ) -> tuple[str, list[str]]:
        prompt = self._build_planning_prompt(
            request=request,
            revision_feedback=revision_feedback,
            previous_plan=previous_plan,
            execution_mode=execution_mode,
            rewrite_quality=rewrite_quality,
            tools_used=tools_used,
        )
        messages = self.context.build_messages(
            history=history,
            current_message=prompt,
            session_summary=session_summary,
            channel=channel,
            chat_id=chat_id,
            session_metadata=session.metadata if session else None,
            session=session,
        )
        result = await self.runner.run(AgentRunSpec(
            initial_messages=messages,
            tools=self.tools.read_only(),
            model=self.model,
            max_iterations=self.planning_config.max_exploration_iterations,
            max_tool_result_chars=self.max_tool_result_chars,
            error_message="Sorry, I encountered an error while creating the plan.",
            concurrent_tools=True,
            workspace=self.workspace,
            context_window_tokens=self.context_window_tokens,
            context_block_limit=self.context_block_limit,
            provider_retry_mode=self.provider_retry_mode,
            mode="planning",
            allow_background_tools=False,
            allow_external_tools=self.web_config.enable,
        ))
        self._last_usage = result.usage
        return result.final_content or "Planning completed but no plan was generated.", result.tools_used

    def _build_planning_prompt(
        self,
        *,
        request: str,
        revision_feedback: str | None = None,
        previous_plan: str | None = None,
        execution_mode: str = "executable",
        rewrite_quality: PlanQualityResult | None = None,
        tools_used: list[str] | None = None,
    ) -> str:
        template = "agent/planning_rewrite.md" if rewrite_quality else "agent/planning_mode.md"
        return render_template(
            template,
            strip=True,
            request=request,
            revision_feedback=revision_feedback,
            previous_plan=previous_plan,
            execution_mode=execution_mode,
            min_exploration_steps=self.planning_config.min_exploration_steps,
            quality_failures=rewrite_quality.failures if rewrite_quality else [],
            rewrite_instructions=rewrite_quality.rewrite_instructions if rewrite_quality else [],
            tools_used=tools_used or [],
        )

    @staticmethod
    def _format_plan_response(plan: PlanState, *, full: bool = False) -> str:
        details = [f"计划已生成（id: {plan.id}）。"]
        details.append(f"状态：{plan.status}。")
        details.append(f"执行模式：{plan.execution_mode}。")
        if plan.block_reason:
            details.append(f"阻塞原因：{plan.block_reason}")
        if plan.checklist:
            completed = sum(1 for item in plan.checklist if item.get("status") == "completed")
            details.append(f"检查清单：已完成 {completed}/{len(plan.checklist)} 项。")
        if full:
            return (
                "\n".join(details)
                + f"\n\n{plan.plan}\n\n"
                + "使用 `/plan-status` 查看简洁版。"
                + "回复 `yes` 或使用 `/plan-approve` 开始执行。"
            )

        lines = details + [""]
        intent = plan_section(plan.plan, "User Intent")
        summary = plan_section(plan.plan, "Summary") or intent
        approach = plan_section_lines(plan.plan, "Proposed Approach", limit=4)
        files = extract_plan_file_paths(plan.plan)[:8]
        acceptance = plan_section_lines(plan.plan, "Acceptance Criteria", limit=6)
        risks = plan_section_lines(plan.plan, "Risks and Open Questions", limit=5)
        visible = aggregate_plan_checklist(plan.checklist, limit=20)

        if intent:
            lines.extend(["## 目标", intent.strip(), ""])
        if summary or approach:
            lines.append("## 摘要")
            if summary:
                lines.append(summary.strip())
            for item in approach:
                lines.append(f"- {item}")
            lines.append("")
        if visible:
            lines.append("## 阶段 / 检查清单")
            for item in visible:
                mark = "x" if item.get("status") == "completed" else " "
                count = f" ({item.get('completed', 0)}/{item.get('count')})" if item.get("count") else ""
                lines.append(f"- [{mark}] {item.get('description', '')}{count}")
            if len(visible) < len(plan.checklist):
                lines.append(f"- 完整检查清单：`/plan-checklist --full`（共 {len(plan.checklist)} 项）")
            lines.append("")
        if files:
            lines.append("## 关键文件")
            for path in files:
                lines.append(f"- `{path}`")
            lines.append("")
        if acceptance:
            lines.append("## 验收摘要")
            for item in acceptance:
                lines.append(f"- {item}")
            lines.append("")
        if risks:
            lines.append("## 风险 / 开放问题")
            for item in risks:
                lines.append(f"- {item}")
            lines.append("")
        lines.extend(["## 下一步", *plan_execution_options(plan)])
        lines.append("使用 `/plan-status --full` 查看完整计划，`/plan-cancel` 放弃计划，或 `/plan-revise <feedback>` 调整计划。")
        return "\n".join(lines)

    @staticmethod
    def _format_plan_revision_response(plan: PlanState, previous_plan: str, feedback: str, *, full: bool = False) -> str:
        if full:
            return AgentLoop._format_plan_response(plan, full=True)
        old_checklist = extract_plan_checklist(previous_plan)
        old_files = set(extract_plan_file_paths(previous_plan))
        new_files = set(extract_plan_file_paths(plan.plan))
        changed_files = sorted(new_files - old_files)[:8]
        visible = aggregate_plan_checklist(plan.checklist, limit=12)
        lines = [
            f"计划已更新（id: {plan.id}）。",
            f"状态：{plan.status}。",
            f"执行模式：{plan.execution_mode}。",
            "",
            "## 变更摘要",
        ]
        for part in re.split(r"[。.;；\n]+", feedback):
            item = part.strip(" -\t")
            if item:
                lines.append(f"- {item}")
        if len(plan.checklist) != len(old_checklist):
            lines.append(f"- 检查清单从 {len(old_checklist)} 项调整为 {len(plan.checklist)} 项。")
        if changed_files:
            lines.append("- 新增受影响文件：" + ", ".join(f"`{path}`" for path in changed_files))
        lines.extend(["", "## 受影响阶段"])
        for item in visible:
            lines.append(f"- {item.get('description', '')}")
        lines.extend(["", "## 是否可执行", "当前计划仍可执行。" if plan.execution_mode == "executable" else "当前计划为只读模式，不会执行写入。", "", "## 下一步", *plan_execution_options(plan)])
        lines.append("使用 `/plan-status --full` 查看完整修订计划。")
        return "\n".join(lines)

    def _quality_failure_response(self, quality: PlanQualityResult) -> str:
        failures = "\n".join(f"- {failure}" for failure in quality.failures)
        return (
            "计划质量检查未通过，因此没有创建待批准计划。\n\n"
            f"{failures}\n\n"
            "请使用 `/plan-revise <feedback>`，或发送更具体的请求后重试。"
        )

    async def _create_pending_plan(
        self,
        *,
        session: ProjectState,
        key: str,
        request: str,
        channel: str,
        chat_id: str,
        session_summary: str | None = None,
        revision_feedback: str | None = None,
        previous_plan: PlanState | None = None,
    ) -> PlanState | str:
        history = session.get_history(max_messages=0)
        execution_mode = detect_execution_mode(request)
        task_class = "code_change_small"
        plan_text, tools_used = await self._run_planning_loop(
            request=request,
            history=history,
            channel=channel,
            chat_id=chat_id,
            session_summary=session_summary,
            revision_feedback=revision_feedback,
            previous_plan=previous_plan.plan if previous_plan else None,
            execution_mode=execution_mode,
            session=session,
        )
        min_steps = (
            self.planning_config.min_exploration_steps
            if self.planning_config.force_exploration and plan_requires_exploration(task_class)
            else 0
        )
        quality = validate_plan_quality(
            plan_text,
            task_class=task_class,
            tools_used=tools_used,
            min_exploration_steps=min_steps,
        )
        rewrites = 0
        while not quality.passed and rewrites < self.planning_config.max_rewrites:
            rewrites += 1
            plan_text, rewrite_tools = await self._run_planning_loop(
                request=request,
                history=history,
                channel=channel,
                chat_id=chat_id,
                session_summary=session_summary,
                revision_feedback=revision_feedback,
                previous_plan=plan_text,
                execution_mode=execution_mode,
                session=session,
                rewrite_quality=quality,
                tools_used=tools_used,
            )
            tools_used = tools_used + rewrite_tools
            quality = validate_plan_quality(
                plan_text,
                task_class=task_class,
                tools_used=tools_used,
                min_exploration_steps=min_steps,
            )
        if not quality.passed:
            return self._quality_failure_response(quality)
        if previous_plan:
            plan = previous_plan
            plan.revise(
                plan_text,
                task_class=task_class,
                execution_mode=execution_mode,
                quality_failures=quality.failures,
                tools_used=tools_used,
                history_limit=self.planning_config.history_limit,
            )
        else:
            plan = PlanState.create(
                original_request=request,
                plan=plan_text,
                task_class=task_class,
                execution_mode=execution_mode,
                quality_failures=quality.failures,
                tools_used=tools_used,
            )
        plan.active_project = session.metadata.get("active_project")
        plan.active_project_path = session.metadata.get("active_project_path")
        plan.save_to_session(session)
        self.sessions.save(session)
        return plan

    @staticmethod
    def _build_approved_execution_prompt(plan: PlanState, execution_scope: dict[str, Any] | None = None) -> str:
        contract = build_execution_contract(plan, execution_scope)
        checklist = "\n".join(
            f"- [ ] {item.get('id', f'step-{idx}')}: {item.get('description', '')}"
            for idx, item in enumerate(contract.get("checklist") or [], start=1)
        ) or "- [ ] 按执行契约完成当前范围并报告验证结果。"
        constraints = "\n".join(f"- {item}" for item in contract.get("non_negotiable_constraints") or []) or "- 遵守批准计划中的 Requirements、Acceptance Criteria 和 Non-goals / Out of Scope。"
        checks = "\n".join(f"- {item}" for item in contract.get("acceptance_checks") or []) or "- 当前范围完成后说明验证结果。"
        scope_note = ""
        if execution_scope and execution_scope.get("kind") == "phases":
            scope_note = "只执行当前范围；不要实现后续阶段，除非这是保持当前范围可运行所必需的。"
        return (
            f"你正在执行已批准计划 `{contract['plan_id']}`。\n\n"
            "# 执行契约\n"
            f"- 执行模式：{contract['execution_mode']}\n"
            f"- 批准内容哈希：{contract['content_hash']}\n"
            f"- 任务摘要：{contract['summary']}\n"
            f"- 当前范围：{contract['current_scope']}\n"
            f"- 冲突规则：{contract['conflict_rule']}\n"
            "- 不可违背的约束：\n"
            f"{constraints}\n\n"
            "# 上下文优先级\n"
            "1. 系统/开发者安全规则。\n"
            "2. 本执行契约。\n"
            "3. 当前执行范围。\n"
            "4. 验收检查。\n"
            "5. 完整批准计划参考。\n"
            "6. Active Project / 当前项目上下文。\n"
            "7. 工具返回结果。\n"
            "8. 旧聊天历史：本次执行默认不提供；即使出现也不具备权威性。\n\n"
            "# 当前执行范围\n"
            f"{scope_note}\n"
            f"{checklist}\n\n"
            "# 验收检查\n"
            f"{checks}\n\n"
            "# 完整批准计划参考\n"
            "以下完整计划仅供参考；如果它与执行契约、当前范围或验收检查冲突，以执行契约和当前范围为准。\n"
            f"<approved_plan_reference>\n{plan.plan}\n</approved_plan_reference>\n\n"
            "# 执行规则\n"
            f"原始请求：\n{plan.original_request}\n\n"
            "- 如果用户可见范围变化、违反非目标、或任务变成高风险/破坏性操作，停止并要求 `/plan-revise` 或单独确认。\n"
            "- 如果工具提示需要额外批准，停止并询问用户，不要重试或绕过。\n"
            "- 基础文件创建、读取和编辑允许在 workspace 或 `/scan`/`/use` 选择的 active project 内进行。\n"
            "- 危险或破坏性操作需要单独确认。\n"
            "- 减少工具往返：一次性理解后，尽量在同一轮批量执行相互独立且安全的工具调用。\n"
            "- 创建多个文件时，组合相关写入，不要交替 write/read。\n"
            "- 写入后不要立刻重读，除非验证失败、文件原本已存在，或需要确认外部状态。\n"
            "- 在 Windows 上优先使用 Python 单行命令或专用工具，避免 cat、ls、grep、find、rm、heredoc、/dev/null 等 Unix shell 语法。\n"
            "- 复用现有项目模式，避免无关改动。\n"
            "- 实现后尽量运行计划中的验证步骤；如果不能验证，说明原因。\n"
            "- 最终回复必须包含一段简短的执行契约符合性检查，说明当前范围、关键约束和验收检查是否满足。"
        )

    async def _execute_approved_plan(
        self,
        *,
        msg: InboundMessage,
        session: ProjectState,
        plan: PlanState,
        session_summary: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue | None = None,
        execution_scope: dict[str, Any] | None = None,
    ) -> OutboundMessage | None:
        if plan.status not in {"approved", "blocked"} or plan.approved_content_hash != plan.content_hash:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="计划未批准或批准后内容已变化。请使用 `/plan-revise <feedback>` 调整，或重新使用 `/plan-approve` 批准。",
                metadata={"render_as": "text"},
            )
        if plan.execution_mode == "read_only":
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="该计划是只读/不执行模式。如果需要实际修改，请使用 `/plan <request>` 创建新的可执行计划。",
                metadata={"render_as": "text"},
            )
        if not plan.checklist:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="已批准计划没有可执行检查清单。请先使用 `/plan-revise <feedback>` 补充后再执行。",
                metadata={"render_as": "text"},
            )
        plan.mark_executing()
        plan.save_to_session(session)
        self.sessions.save(session)
        self._allow_active_project_tools(session, plan)
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()
        before_count = len(session.messages)
        await self.consolidator.maybe_consolidate_by_tokens(session)
        if len(session.messages) != before_count:
            logger.info(
                "Approved plan pre-consolidation changed session history for {}: {} -> {} messages",
                session.key,
                before_count,
                len(session.messages),
            )
        history = []
        execution_prompt = self._build_approved_execution_prompt(plan, execution_scope)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=execution_prompt,
            session_summary=session_summary,
            channel=msg.channel,
            chat_id=msg.chat_id,
            session_metadata=session.metadata,
            session=session,
        )
        try:
            estimate, source = estimate_prompt_tokens_chain(
                self.provider,
                self.model,
                initial_messages,
                self.tools.get_definitions(),
            )
        except Exception:
            logger.exception("Failed to estimate approved plan prompt size for {}", session.key)
        else:
            system_chars = sum(
                len(str(item.get("content") or ""))
                for item in initial_messages
                if item.get("role") == "system"
            )
            try:
                tool_definitions = self.tools.get_definitions()
                approx_message_bytes = len(json.dumps(initial_messages, ensure_ascii=False).encode("utf-8"))
                approx_tool_bytes = len(json.dumps(tool_definitions, ensure_ascii=False).encode("utf-8"))
                approx_bytes = approx_message_bytes + approx_tool_bytes
            except Exception:
                tool_definitions = self.tools.get_definitions()
                approx_message_bytes = -1
                approx_tool_bytes = -1
                approx_bytes = -1
            logger.info(
                "Approved plan prompt size for {}: tokens={} via {}, messages={}, history={}, system_chars={}, tools={}, approx_message_bytes={}, approx_tool_bytes={}, approx_request_bytes={}",
                session.key,
                estimate,
                source,
                len(initial_messages),
                len(history),
                system_chars,
                len(tool_definitions),
                approx_message_bytes,
                approx_tool_bytes,
                approx_bytes,
            )
        policy_context = self._build_policy_context(plan, session)

        async def _bus_progress(
            content: str,
            *,
            tool_hint: bool = False,
            plan_stage: bool = False,
            change_summary: bool = False,
        ) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            meta["_plan_stage"] = plan_stage
            meta["_change_summary"] = change_summary
            await self.bus.publish_outbound(OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta))

        final_content, _, all_msgs, stop_reason, had_injections, pending_approval = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=None,
            on_stream_end=None,
            session=session,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            pending_queue=pending_queue,
            policy_context=policy_context,
            context_block_limit=self.approved_plan_context_block_limit,
        )
        if final_content is None or not final_content.strip():
            final_content = EMPTY_FINAL_RESPONSE_MESSAGE
        self._save_turn(session, all_msgs, 1 + len(history))
        if stop_reason == "approval_required" and pending_approval is not None:
            session.metadata[self._PENDING_APPROVAL_KEY] = pending_approval
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=self._format_tool_approval_prompt(pending_approval),
                metadata={"render_as": "text"},
            )
        if stop_reason == "error" or final_content.startswith("Error: Tool '"):
            plan.mark_blocked(final_content)
            plan.save_to_session(session)
        else:
            plan.mark_completed(history_limit=self.planning_config.history_limit)
            plan.archive(status="completed")
            session.metadata.pop("pending_plan", None)
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)
        self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return None
        meta = dict(msg.metadata or {})
        if on_stream is not None and stop_reason != "error":
            meta["_streamed"] = True
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=final_content, metadata=meta)

    async def _resolve_pending_tool_approval(
        self,
        ctx: CommandContext,
        *,
        approved: bool,
    ) -> OutboundMessage:
        session = ctx.session or self.sessions.get_or_create(ctx.key)
        pending = session.metadata.get(self._PENDING_APPROVAL_KEY)
        if not isinstance(pending, dict):
            session.metadata.pop(self._PENDING_APPROVAL_KEY, None)
            self.sessions.save(session)
            return OutboundMessage(
                channel=ctx.msg.channel,
                chat_id=ctx.msg.chat_id,
                content="No pending tool approval.",
                metadata={"render_as": "text"},
            )

        tool_call_id = str(pending.get("id") or pending.get("tool_call_id") or "")
        tool_name = str(pending.get("tool_name") or "tool")
        if not tool_call_id:
            session.metadata.pop(self._PENDING_APPROVAL_KEY, None)
            self.sessions.save(session)
            return OutboundMessage(
                channel=ctx.msg.channel,
                chat_id=ctx.msg.chat_id,
                content="The pending tool approval was stale and has been cleared.",
                metadata={"render_as": "text"},
            )

        if approved:
            try:
                tool, params, prep_error = self.tools.prepare_call(tool_name, pending.get("arguments") or {})
                if prep_error:
                    content = prep_error
                elif tool is not None:
                    content = await tool.execute(**params)
                else:
                    content = await self.tools.execute(tool_name, params)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                content = f"Error: {type(exc).__name__}: {exc}"
        else:
            content = "Tool call denied by user."

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content,
        }
        session.messages.append(tool_message)
        session.metadata.pop(self._PENDING_APPROVAL_KEY, None)
        self.sessions.save(session)

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message="The pending tool approval has been resolved. Continue from the tool result.",
            session_summary=None,
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            session_metadata=session.metadata,
            session=session,
        )
        final_content, _, all_msgs, stop_reason, had_injections, next_pending = await self._run_agent_loop(
            initial_messages,
            session=session,
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            message_id=ctx.msg.metadata.get("message_id"),
            policy_context=self._build_policy_context(PlanState.from_session(session), session),
        )
        if final_content is None or not final_content.strip():
            final_content = EMPTY_FINAL_RESPONSE_MESSAGE
        self._save_turn(session, all_msgs, 1 + len(history))
        if stop_reason == "approval_required" and next_pending is not None:
            session.metadata[self._PENDING_APPROVAL_KEY] = next_pending
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            return OutboundMessage(
                channel=ctx.msg.channel,
                chat_id=ctx.msg.chat_id,
                content=self._format_tool_approval_prompt(next_pending),
                metadata={"render_as": "text"},
            )
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content="", metadata=dict(ctx.msg.metadata or {}))
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=final_content,
            metadata=dict(ctx.msg.metadata or {}),
        )

    @staticmethod
    def _format_tool_approval_prompt(pending_approval: dict[str, Any]) -> str:
        tool_name = pending_approval.get("tool_name") or "tool"
        reason = pending_approval.get("reason") or "Tool requires approval"
        return (
            "Tool approval required.\n\n"
            f"Tool: {tool_name}\n"
            f"Reason: {reason}\n\n"
            "Reply `approve` to run it or `deny` to skip it."
        )

    @staticmethod
    def _build_policy_context(plan: PlanState | None = None, session: ProjectState | None = None) -> PolicyContext:
        allowed_roots = []
        active_project_path = AgentLoop._active_project_path(session, plan)
        if active_project_path:
            allowed_roots.append(active_project_path)
        if plan is None:
            return PolicyContext(allowed_roots=allowed_roots)
        return PolicyContext(
            approved_plan_id=plan.id,
            approved_content_hash=plan.approved_content_hash or plan.content_hash,
            allowed_roots=allowed_roots,
        )

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        session: ProjectState | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        pending_queue: asyncio.Queue | None = None,
        policy_context: PolicyContext | None = None,
        context_block_limit: int | None = None,
    ) -> tuple[str | None, list[str], list[dict], str, bool, dict[str, Any] | None]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.

        Returns (final_content, tools_used, messages, stop_reason, had_injections, pending_approval).
        """
        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
        )
        hook: AgentHook = (
            CompositeHook([loop_hook] + self._extra_hooks) if self._extra_hooks else loop_hook
        )

        async def _checkpoint(payload: dict[str, Any]) -> None:
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        async def _drain_pending(*, limit: int = _MAX_INJECTIONS_PER_TURN) -> list[dict[str, Any]]:
            """Non-blocking drain of follow-up messages from the pending queue."""
            if pending_queue is None:
                return []
            items: list[dict[str, Any]] = []
            while len(items) < limit:
                try:
                    pending_msg = pending_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                user_content = self.context._build_user_content(
                    pending_msg.content,
                    pending_msg.media if pending_msg.media else None,
                )
                runtime_ctx = self.context._build_runtime_context(
                    pending_msg.channel,
                    pending_msg.chat_id,
                    self.context.timezone,
                )
                if isinstance(user_content, str):
                    merged: str | list[dict[str, Any]] = f"{runtime_ctx}\n\n{user_content}"
                else:
                    merged = [{"type": "text", "text": runtime_ctx}] + user_content
                items.append({"role": "user", "content": merged})
            return items

        project_key = session.key if session else None
        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=self.tools,
            model=self.model,
            max_iterations=self.max_iterations,
            max_tool_result_chars=self.max_tool_result_chars,
            hook=hook,
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
            workspace=self.workspace,
            project_key=project_key,
            context_window_tokens=self.context_window_tokens,
            context_block_limit=context_block_limit if context_block_limit is not None else self.context_block_limit,
            provider_retry_mode=self.provider_retry_mode,
            progress_callback=on_progress,
            checkpoint_callback=_checkpoint,
            injection_callback=_drain_pending,
            mode="coding" if self.coding_config.enable else "general",
            approval_mode=self.coding_config.approval_mode,
            require_approval_for_write=(
                self.coding_config.enable and self.coding_config.approval_mode == "manual" and not self.coding_config.allow_write
            ),
            require_approval_for_exec=(
                self.coding_config.enable and self.coding_config.approval_mode == "manual" and not self.coding_config.allow_exec
            ),
            allow_background_tools=(
                not self.coding_config.enable or self.coding_config.allow_spawn or self.coding_config.allow_cron
            ),
            allow_external_tools=(not self.coding_config.enable or self.coding_config.allow_web),
            allowed_tool_names=self._allowed_tool_names(),
            blocked_tool_prefixes=self._blocked_tool_prefixes(),
            policy_engine=self.policy_engine,
            policy_context=policy_context or PolicyContext(),
        ))
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections, result.pending_approval

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                self.auto_compact.check_expired(self._schedule_background)
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                origin_key = self._origin_key(msg)
                ctx = CommandContext(msg=msg, project=None, project_key=origin_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            effective_key = self._effective_project_key(msg)
            # If this project already has an active pending queue (i.e. a task
            # is processing this project), route the message there for mid-turn
            # injection instead of creating a competing task.
            if effective_key in self._pending_queues:
                pending_msg = msg
                if effective_key != msg.session_key:
                    pending_msg = dataclasses.replace(
                        msg,
                        session_key_override=effective_key,
                    )
                try:
                    self._pending_queues[effective_key].put_nowait(pending_msg)
                except asyncio.QueueFull:
                    logger.warning(
                        "Pending queue full for project {}, falling back to queued task",
                        effective_key,
                    )
                else:
                    logger.info(
                        "Routed follow-up message to pending queue for project {}",
                        effective_key,
                    )
                    continue
            # Compute the effective project key before dispatching
            # This ensures /stop command can find tasks correctly when unified session is enabled
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(effective_key, []).append(task)
            task.add_done_callback(
                lambda t, k=effective_key: self._active_tasks.get(k, [])
                and self._active_tasks[k].remove(t)
                if t in self._active_tasks.get(k, [])
                else None
            )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-project serial, cross-project concurrent."""
        project_key = self._effective_project_key(msg)
        if project_key != msg.session_key:
            msg = dataclasses.replace(msg, session_key_override=project_key)
        lock = self._project_locks.setdefault(project_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()

        # Register a pending queue so follow-up messages for this project are
        # routed here (mid-turn injection) instead of spawning a new task.
        pending = asyncio.Queue(maxsize=20)
        self._pending_queues[project_key] = pending

        try:
            async with lock, gate:
                try:
                    on_stream = on_stream_end = None
                    if msg.metadata.get("_wants_stream"):
                        # Split one answer into distinct stream segments.
                        stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                        stream_segment = 0

                        def _current_stream_id() -> str:
                            return f"{stream_base_id}:{stream_segment}"

                        async def on_stream(delta: str) -> None:
                            meta = dict(msg.metadata or {})
                            meta["_stream_delta"] = True
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content=delta,
                                metadata=meta,
                            ))

                        async def on_stream_end(*, resuming: bool = False) -> None:
                            nonlocal stream_segment
                            meta = dict(msg.metadata or {})
                            meta["_stream_end"] = True
                            meta["_resuming"] = resuming
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="",
                                metadata=meta,
                            ))
                            stream_segment += 1

                    response = await self._process_message(
                        msg, on_stream=on_stream, on_stream_end=on_stream_end,
                        pending_queue=pending,
                    )
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="", metadata=msg.metadata or {},
                        ))
                except asyncio.CancelledError:
                    logger.info("Task cancelled for project {}", project_key)
                    raise
                except Exception:
                    logger.exception("Error processing message for project {}", project_key)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    ))
        finally:
            # Drain any messages still in the pending queue and re-publish
            # them to the bus so they are processed as fresh inbound messages
            # rather than silently lost.
            queue = self._pending_queues.pop(project_key, None)
            if queue is not None:
                leftover = 0
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await self.bus.publish_inbound(item)
                    leftover += 1
                if leftover:
                    logger.info(
                        "Re-published {} leftover message(s) to bus for project {}",
                        leftover, project_key,
                    )

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        for name, stack in self._mcp_stacks.items():
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP server '{}' cleanup error (can be ignored)", name)
        self._mcp_stacks.clear()

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        project_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            origin_key = f"{channel}:{chat_id}"
            key = self.projects.get_active_for_origin(origin_key) or DEFAULT_PROJECT_ID
            session = self.projects.get_or_create(key)
            if self._restore_runtime_checkpoint(session):
                self.projects.save(session)

            session, pending = self.auto_compact.prepare_session(session, key)

            await self.consolidator.maybe_consolidate_by_tokens(session)
            self._allow_active_project_tools(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"

            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                session_summary=pending,
                current_role=current_role,
                session_metadata=session.metadata,
                session=session,
            )
            final_content, _, all_msgs, _, _, _ = await self._run_agent_loop(
                messages, session=session, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = project_key or self._effective_project_key(msg)
        session = self.projects.get_or_create(key)
        if self._restore_runtime_checkpoint(session):
            self.projects.save(session)

        session, pending = self.auto_compact.prepare_session(session, key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, project=session, project_key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        existing_plan = PlanState.from_session(session)
        if existing_plan and existing_plan.status == "pending":
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    "有一个计划正在等待批准。请回复 `yes`，或使用 `/plan-approve`、"
                    "`/plan-cancel`、`/plan-revise <feedback>`。"
                ),
                metadata={"render_as": "text"},
            )
        if should_plan_request(msg.content, self.planning_config):
            plan = await self._create_pending_plan(
                session=session,
                key=key,
                request=msg.content,
                channel=msg.channel,
                chat_id=msg.chat_id,
                session_summary=pending,
            )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=self._format_plan_response(plan) if isinstance(plan, PlanState) else plan,
                metadata={"render_as": "text"},
            )

        await self.consolidator.maybe_consolidate_by_tokens(session)

        self._allow_active_project_tools(session)
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)

        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            session_summary=pending,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            session_metadata=session.metadata,
            session=session,
        )

        async def _bus_progress(
            content: str,
            *,
            tool_hint: bool = False,
            plan_stage: bool = False,
            change_summary: bool = False,
        ) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            meta["_plan_stage"] = plan_stage
            meta["_change_summary"] = change_summary
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        final_content, _, all_msgs, stop_reason, had_injections, pending_approval = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            session=session,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            pending_queue=pending_queue,
        )

        if final_content is None or not final_content.strip():
            final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        self._save_turn(session, all_msgs, 1 + len(history))
        if stop_reason == "approval_required" and pending_approval is not None:
            session.metadata[self._PENDING_APPROVAL_KEY] = pending_approval
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=self._format_tool_approval_prompt(pending_approval),
                metadata={"render_as": "text"},
            )
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)
        self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))

        # When follow-up messages were injected mid-turn, a later natural
        # language reply may address those follow-ups and should not be
        # suppressed just because MessageTool was used earlier in the turn.
        # However, if the turn falls back to the empty-final-response
        # placeholder, suppress it when the real user-visible output already
        # came from MessageTool.
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None and stop_reason != "error":
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
        )

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        should_truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": image_placeholder_text(path)})
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: ProjectState, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the entire runtime-context block (including any session summary).
                    # The block is bounded by _RUNTIME_CONTEXT_TAG and _RUNTIME_CONTEXT_END.
                    end_marker = ContextBuilder._RUNTIME_CONTEXT_END
                    end_pos = content.find(end_marker)
                    if end_pos >= 0:
                        after = content[end_pos + len(end_marker):].lstrip("\n")
                        if after:
                            entry["content"] = after
                        else:
                            continue
                    else:
                        # Fallback: no end marker found, strip the tag prefix
                        after_tag = content[len(ContextBuilder._RUNTIME_CONTEXT_TAG):].lstrip("\n")
                        if after_tag.strip():
                            entry["content"] = after_tag
                        else:
                            continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    def _set_runtime_checkpoint(self, session: ProjectState, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _clear_runtime_checkpoint(self, session: ProjectState) -> None:
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _restore_runtime_checkpoint(self, session: ProjectState) -> bool:
        """Materialize an unfinished turn into session history before a new request."""
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": "Error: Task interrupted before this tool finished.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])

        self._clear_runtime_checkpoint(session)
        return True

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        return await self._process_message(
            msg,
            project_key=session_key,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )

"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

from ghostbot import __version__
from ghostbot.bus.events import OutboundMessage
from ghostbot.command.router import CommandContext, CommandRouter
from ghostbot.agent.planning import (
    PlanState,
    is_affirmative_response,
    is_negative_response,
    list_archived_plans,
    parse_plan_execution_scope,
    plan_execution_options,
)
from ghostbot.utils.helpers import build_status_content
from ghostbot.utils.restart import set_restart_notice_to_env


def _extract_project_path(project_file) -> str | None:
    try:
        for line in project_file.read_text(encoding="utf-8").splitlines()[:10]:
            match = re.match(r">\s*Project Path:\s*(.+?)\s*$", line)
            if match:
                return match.group(1)
    except Exception:
        return None
    return None


async def cmd_scan(ctx: CommandContext) -> OutboundMessage:
    import os
    from pathlib import Path
    from ghostbot.utils.project_analyzer import sync_project_structure

    args = ctx.args.strip()
    if args:
        target_path = Path(args).expanduser()
    else:
        target_path = Path(os.getcwd())

    if not target_path.exists():
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=f"❌ 抱歉，找不到路径：`{target_path}`",
        )

    if not ctx.loop:
        return OutboundMessage(content="❌ 系统内部错误：AgentLoop 未绑定。")

    memory_dir = ctx.loop.workspace / "memory"

    ignore_dirs = {".venv", "venv", "env", "sessions", ".git", ".pytest_cache", "__pycache__", "node_modules"}

    is_success, saved_file = sync_project_structure(
        target_path,
        memory_dir,
        force=True,
        ignore_dirs=ignore_dirs
    )

    if is_success:
        project_name = saved_file.stem
        project_path = str(target_path.resolve())
        project = ctx.loop.projects.get_or_create(project_name, path=project_path, name=project_name)
        project.metadata["active_project"] = project_name
        project.metadata["active_project_path"] = project_path
        ctx.loop.projects.save(project)
        ctx.loop.projects.set_active_for_origin(ctx.loop._origin_key(ctx.msg), project.key)

        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=f"✅ 扫描完成并已切换至项目: `{project_name}`\n路径: `{project_path}`",
            metadata=dict(ctx.msg.metadata or {})
        )
    else:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=f"❌ 扫描失败，请检查路径权限或项目结构：`{target_path}`",
            metadata=dict(ctx.msg.metadata or {})
        )


async def cmd_use(ctx: CommandContext) -> OutboundMessage:
    """手动切换当前关注的项目。Usage: /use <project_name>"""
    project_name = ctx.args.strip()
    if not project_name:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="❌ 请指定项目名称。用法: `/use <project_name>`"
        )

    if not ctx.loop:
        return OutboundMessage(content="❌ 系统内部错误：AgentLoop 未绑定。")

    project_file = ctx.loop.workspace / "memory" / "projects" / f"{project_name}.md"
    if not project_file.exists():
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=f"❌ 找不到项目 `{project_name}` 的记录，请先使用 `/scan` 扫描。"
        )

    project_path = _extract_project_path(project_file)
    project = ctx.loop.projects.get_or_create(project_name, path=project_path, name=project_name)
    project.metadata["active_project"] = project_name
    if project_path:
        project.metadata["active_project_path"] = project_path
    ctx.loop.projects.save(project)
    ctx.loop.projects.set_active_for_origin(ctx.loop._origin_key(ctx.msg), project.key)

    path_note = f" 路径: `{project_path}`" if project_path else ""

    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=f"🔄 已切换至项目: `{project_name}`{path_note}。后续对话将基于此架构展开。",
    )

async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the project."""
    loop = ctx.loop
    msg = ctx.msg
    project_key = loop._effective_project_key(msg)
    tasks = loop._active_tasks.pop(project_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    sub_cancelled = await loop.subagents.cancel_by_session(project_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content=content,
        metadata=dict(msg.metadata or {})
    )


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg
    set_restart_notice_to_env(channel=msg.channel, chat_id=msg.chat_id)

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "ghostbot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        metadata=dict(msg.metadata or {})
    )


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a project."""
    loop = ctx.loop
    project_key = loop._effective_project_key(ctx.msg)
    session = ctx.session or loop.projects.get_or_create(project_key)
    ctx_est = 0
    try:
        ctx_est, _ = loop.consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)

    # Fetch web search provider usage (best-effort, never blocks the response)
    search_usage_text: str | None = None
    try:
        from ghostbot.utils.searchusage import fetch_search_usage
        web_cfg = getattr(loop, "web_config", None)
        search_cfg = getattr(web_cfg, "search", None) if web_cfg else None
        if search_cfg is not None:
            provider = getattr(search_cfg, "provider", "duckduckgo")
            api_key = getattr(search_cfg, "api_key", "") or None
            usage = await fetch_search_usage(provider=provider, api_key=api_key)
            search_usage_text = usage.format()
    except Exception:
        pass  # Never let usage fetch break /status
    metadata = session.metadata or {}
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
            search_usage_text=search_usage_text,
            mode=metadata.get("mode"),
            repo_root=metadata.get("repo_root"),
            branch=metadata.get("active_branch"),
            approval_mode=metadata.get("approval_mode"),
            isolation=metadata.get("isolation"),
            last_action=metadata.get("last_action"),
            active_project=metadata.get("active_project"),
            active_project_path=metadata.get("active_project_path"),
        ),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Clear the current project conversation history."""
    loop = ctx.loop
    project_key = loop._effective_project_key(ctx.msg)
    session = ctx.session or loop.projects.get_or_create(project_key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    loop.projects.save(session)
    loop.projects.invalidate(session.key)
    if snapshot:
        loop._schedule_background(loop.consolidator.archive(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New project conversation started.",
        metadata=dict(ctx.msg.metadata or {})
    )


async def cmd_dream(ctx: CommandContext) -> OutboundMessage:
    """Manually trigger a Dream consolidation run."""
    import time

    loop = ctx.loop
    msg = ctx.msg

    async def _run_dream():
        t0 = time.monotonic()
        try:
            did_work = await loop.dream.run()
            elapsed = time.monotonic() - t0
            if did_work:
                content = f"Dream completed in {elapsed:.1f}s."
            else:
                content = "Dream: nothing to process."
        except Exception as e:
            elapsed = time.monotonic() - t0
            content = f"Dream failed after {elapsed:.1f}s: {e}"
        await loop.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    asyncio.create_task(_run_dream())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="Dreaming...",
    )


def _extract_changed_files(diff: str) -> list[str]:
    """Extract changed file paths from a unified diff."""
    files: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def _format_changed_files(diff: str) -> str:
    files = _extract_changed_files(diff)
    if not files:
        return "No tracked memory files changed."
    return ", ".join(f"`{path}`" for path in files)


def _format_dream_log_content(commit, diff: str, *, requested_sha: str | None = None) -> str:
    files_line = _format_changed_files(diff)
    lines = [
        "## Dream Update",
        "",
        "Here is the selected Dream memory change." if requested_sha else "Here is the latest Dream memory change.",
        "",
        f"- Commit: `{commit.sha}`",
        f"- Time: {commit.timestamp}",
        f"- Changed files: {files_line}",
    ]
    if diff:
        lines.extend([
            "",
            f"Use `/dream-restore {commit.sha}` to undo this change.",
            "",
            "```diff",
            diff.rstrip(),
            "```",
        ])
    else:
        lines.extend([
            "",
            "Dream recorded this version, but there is no file diff to display.",
        ])
    return "\n".join(lines)


def _format_dream_restore_list(commits: list) -> str:
    lines = [
        "## Dream Restore",
        "",
        "Choose a Dream memory version to restore. Latest first:",
        "",
    ]
    for c in commits:
        lines.append(f"- `{c.sha}` {c.timestamp} - {c.message.splitlines()[0]}")
    lines.extend([
        "",
        "Preview a version with `/dream-log <sha>` before restoring it.",
        "Restore a version with `/dream-restore <sha>`.",
    ])
    return "\n".join(lines)


async def cmd_dream_log(ctx: CommandContext) -> OutboundMessage:
    """Show what the last Dream changed.

    Default: diff of the latest commit (HEAD~1 vs HEAD).
    With /dream-log <sha>: diff of that specific commit.
    """
    store = ctx.loop.consolidator.store
    git = store.git

    if not git.is_initialized():
        if store.get_last_dream_cursor() == 0:
            msg = "Dream has not run yet. Run `/dream`, or wait for the next scheduled Dream cycle."
        else:
            msg = "Dream history is not available because memory versioning is not initialized."
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=msg, metadata={"render_as": "text"},
        )

    args = ctx.args.strip()

    if args:
        # Show diff of a specific commit
        sha = args.split()[0]
        result = git.show_commit_diff(sha)
        if not result:
            content = (
                f"Couldn't find Dream change `{sha}`.\n\n"
                "Use `/dream-restore` to list recent versions, "
                "or `/dream-log` to inspect the latest one."
            )
        else:
            commit, diff = result
            content = _format_dream_log_content(commit, diff, requested_sha=sha)
    else:
        # Default: show the latest commit's diff
        commits = git.log(max_entries=1)
        result = git.show_commit_diff(commits[0].sha) if commits else None
        if result:
            commit, diff = result
            content = _format_dream_log_content(commit, diff)
        else:
            content = "Dream memory has no saved versions yet."

    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


async def cmd_dream_restore(ctx: CommandContext) -> OutboundMessage:
    """Restore memory files from a previous dream commit.

    Usage:
        /dream-restore          — list recent commits
        /dream-restore <sha>    — revert a specific commit
    """
    store = ctx.loop.consolidator.store
    git = store.git
    if not git.is_initialized():
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="Dream history is not available because memory versioning is not initialized.",
        )

    args = ctx.args.strip()
    if not args:
        # Show recent commits for the user to pick
        commits = git.log(max_entries=10)
        if not commits:
            content = "Dream memory has no saved versions to restore yet."
        else:
            content = _format_dream_restore_list(commits)
    else:
        sha = args.split()[0]
        result = git.show_commit_diff(sha)
        changed_files = _format_changed_files(result[1]) if result else "the tracked memory files"
        new_sha = git.revert(sha)
        if new_sha:
            content = (
                f"Restored Dream memory to the state before `{sha}`.\n\n"
                f"- New safety commit: `{new_sha}`\n"
                f"- Restored files: {changed_files}\n\n"
                f"Use `/dream-log {new_sha}` to inspect the restore diff."
            )
        else:
            content = (
                f"Couldn't restore Dream change `{sha}`.\n\n"
                "It may not exist, or it may be the first saved version with no earlier state to restore."
            )
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_help_text(),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_plan(ctx: CommandContext) -> OutboundMessage:
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    request = ctx.args.strip()
    full = request == "--full"
    if full:
        request = ""
    if not request:
        plan = PlanState.from_session(session)
        content = ctx.loop._format_plan_response(plan, full=full) if plan else "没有待处理计划。使用 `/plan <request>` 创建新计划。"
        return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content, metadata={"render_as": "text"})
    plan = await ctx.loop._create_pending_plan(
        session=session,
        key=session.key,
        request=request,
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
    )
    content = ctx.loop._format_plan_response(plan) if isinstance(plan, PlanState) else plan
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content, metadata={"render_as": "text"})


async def cmd_plan_status(ctx: CommandContext) -> OutboundMessage:
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    plan = PlanState.from_session(session)
    full = ctx.args.strip() == "--full"
    if not plan:
        content = "没有待处理计划。"
    else:
        content = ctx.loop._format_plan_response(plan, full=full)
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content, metadata={"render_as": "text"})


async def cmd_plan_history(ctx: CommandContext) -> OutboundMessage:
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    plan = PlanState.from_session(session)
    archived = list_archived_plans(limit=10)
    lines = ["计划历史："]
    if plan and plan.history:
        for item in plan.history[-10:]:
            lines.append(
                f"- {item.get('status', 'unknown')} {item.get('id', '')}: {item.get('summary', '')}"
            )
    if archived:
        lines.append("本地归档计划：")
        for item in archived:
            lines.append(f"- {item.id} [{item.status}]: {item.original_request[:100]}")
    content = "\n".join(lines) if len(lines) > 1 else "没有计划历史。"
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content, metadata={"render_as": "text"})


async def cmd_plan_checklist(ctx: CommandContext) -> OutboundMessage:
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    plan = PlanState.from_session(session)
    if not plan:
        content = "没有待处理计划。"
    elif not plan.checklist:
        content = "当前计划没有检查清单。"
    else:
        full = ctx.args.strip() == "--full"
        from ghostbot.agent.planning import aggregate_plan_checklist
        items = plan.checklist if full else aggregate_plan_checklist(plan.checklist, limit=20)
        lines = [f"计划 {plan.id} 的检查清单："]
        for item in items:
            mark = "x" if item.get("status") == "completed" else " "
            count = f" ({item.get('completed', 0)}/{item.get('count')})" if item.get("count") else ""
            lines.append(f"- [{mark}] {item.get('id')}: {item.get('description', '')}{count}")
        if not full and len(items) < len(plan.checklist):
            lines.append(f"使用 `/plan-checklist --full` 查看全部 {len(plan.checklist)} 项。")
        content = "\n".join(lines)
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content, metadata={"render_as": "text"})


async def cmd_plan_cancel(ctx: CommandContext) -> OutboundMessage:
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    plan = PlanState.from_session(session)
    if not plan:
        content = "没有可取消的待处理计划。"
    else:
        plan.mark_cancelled(history_limit=ctx.loop.planning_config.history_limit)
        plan.archive(status="cancelled")
        PlanState.clear_from_session(session)
        ctx.loop.projects.save(session)
        content = f"已取消计划 {plan.id}。"
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content, metadata={"render_as": "text"})


async def cmd_plan_revise(ctx: CommandContext) -> OutboundMessage:
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    plan = PlanState.from_session(session)
    feedback = ctx.args.strip()
    full = False
    if feedback.startswith("--full "):
        full = True
        feedback = feedback[len("--full "):].strip()
    if not plan:
        content = "没有可修订的待处理计划。"
    elif not feedback:
        content = "用法：`/plan-revise <feedback>`。"
    else:
        previous_plan_text = plan.plan
        revised = await ctx.loop._create_pending_plan(
            session=session,
            key=session.key,
            request=plan.original_request,
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            revision_feedback=feedback,
            previous_plan=plan,
        )
        content = ctx.loop._format_plan_revision_response(revised, previous_plan_text, feedback, full=full) if isinstance(revised, PlanState) else revised
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content, metadata={"render_as": "text"})


async def cmd_plan_load(ctx: CommandContext) -> OutboundMessage:
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    arg = ctx.args.strip()
    if not arg:
        return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content="用法：`/plan-load <plan_id-or-path>`。", metadata={"render_as": "text"})
    candidate = Path(arg).expanduser()
    if candidate.exists():
        plan = PlanState.load_from_path(candidate)
    else:
        plan = PlanState.load_history(arg)
    if plan is None:
        return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=f"无法加载计划 `{arg}`。", metadata={"render_as": "text"})
    plan.reset_for_loading()
    plan.save_to_session(session)
    ctx.loop.projects.save(session)
    content = "已将计划加载到当前项目。\n\n" + ctx.loop._format_plan_response(plan)
    return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=content, metadata={"render_as": "text"})


async def cmd_plan_approve(ctx: CommandContext) -> OutboundMessage | None:
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    plan = PlanState.from_session(session)
    if not plan:
        return OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content="没有可批准的待处理计划。", metadata={"render_as": "text"})
    scope = parse_plan_execution_scope(ctx.raw, plan)
    if scope is None and ctx.args.strip() == "" and len(plan_execution_options(plan)) > 1:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="\n".join(plan_execution_options(plan)),
            metadata={"render_as": "text"},
        )
    plan.mark_approved(ctx.raw)
    plan.save_to_session(session)
    ctx.loop.projects.save(session)
    return await ctx.loop._execute_approved_plan(msg=ctx.msg, session=session, plan=plan, execution_scope=scope)


async def tool_approval_interceptor(ctx: CommandContext) -> OutboundMessage | None:
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    pending = session.metadata.get(ctx.loop._PENDING_APPROVAL_KEY)
    if not pending:
        return None
    normalized = ctx.raw.strip().lower()
    if normalized in {"approve", "/approve", "yes", "y", "ok", "确认", "同意", "批准"}:
        return await ctx.loop._resolve_pending_tool_approval(ctx, approved=True)
    if normalized in {"deny", "/deny", "no", "n", "cancel", "取消", "拒绝", "不同意"}:
        return await ctx.loop._resolve_pending_tool_approval(ctx, approved=False)
    tool_name = pending.get("tool_name") or "tool"
    reason = pending.get("reason") or "Tool requires approval"
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=(
            "A tool approval is pending.\n\n"
            f"Tool: {tool_name}\n"
            f"Reason: {reason}\n\n"
            "请回复 `approve` 执行，或回复 `deny` 跳过。"
        ),
        metadata={"render_as": "text"},
    )


async def plan_approval_interceptor(ctx: CommandContext) -> OutboundMessage | None:
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    plan = PlanState.from_session(session)
    if not plan or plan.status != "pending":
        return None
    if is_affirmative_response(ctx.raw):
        return await cmd_plan_approve(ctx)
    if is_negative_response(ctx.raw):
        return await cmd_plan_cancel(ctx)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="有一个计划正在等待批准。请回复 `yes`，或使用 `/plan-approve`、`/plan-cancel`、`/plan-revise <feedback>`。",
        metadata={"render_as": "text"},
    )


from ghostbot.utils.gitstore import GitStore

from ghostbot.command.router import CommandContext
from ghostbot.utils.gitstore import GitStore


# 假设你的工程里有 OutboundMessage，如果没有可以直接省略类型注解
# from ghostbot.bus.events import OutboundMessage

async def cmd_undo(ctx: CommandContext) -> "OutboundMessage":
    """⏪ 撤销 AI 的上一次代码修改"""
    import subprocess
    from ghostbot.bus.events import OutboundMessage
    from loguru import logger

    # 1. 动态获取当前活跃的项目路径
    session = ctx.session or ctx.loop.projects.get_or_create(ctx.loop._effective_project_key(ctx.msg))
    active_path = session.metadata.get("active_project_path")

    # 优先用项目路径，没设定则用工作区兜底
    target_dir = active_path if active_path else str(ctx.loop.workspace)

    logger.info(f"[/undo] Attempting to rollback in target directory: {target_dir}")

    try:
        # 2. 探测真正的 Git 根目录（防止在子目录执行导致失败）
        # ⚠️ 加上 timeout 防止底层 Git 卡死阻塞整个 Agent
        root_res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=target_dir, capture_output=True, text=True, encoding="utf-8", timeout=3
        )
        if root_res.returncode != 0:
            return OutboundMessage(
                channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
                content=f"❌ 无法在该路径执行 Git 操作 (不是合法的 Git 仓库): `{target_dir}`"
            )

        git_root = root_res.stdout.strip()
        logger.info(f"[/undo] Resolved Git Toplevel root: {git_root}")

    except subprocess.TimeoutExpired:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=f"❌ 撤销超时：检查 Git 仓库状态时卡死，可能是目录被编辑器锁住: `{target_dir}`"
        )
    except Exception as e:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=f"❌ Git 仓库探测出错: {e}"
        )

    try:
        # 3. 执行回退
        logger.info(f"[/undo] Executing git reset --hard HEAD~1 in {git_root}")
        # ⚠️ capture_output=True 极其重要，防止 Windows 管道堵塞
        reset_res = subprocess.run(
            ["git", "reset", "--hard", "HEAD~1"],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=10
        )
        logger.debug(f"[/undo] Git reset output: {reset_res.stdout.strip()}")

        # 4. 清理上下文，防止大模型记忆混乱
        project = ctx.project or ctx.loop.projects.get_or_create(ctx.project_key)
        project.clear()
        ctx.loop.projects.save(project)

        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=f"⏪ **回滚成功！** 代码已安全恢复至 `{git_root}` 的上一个版本。\n\n🧹 项目上下文已强制清理，你可以重新发号施令了。"
        )

    except subprocess.TimeoutExpired:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="❌ 撤销超时：执行 `git reset` 时卡死，可能是文件正被其他程序占用。"
        )
    except subprocess.CalledProcessError as e:
        # 🟢 提取底层真实的 Git 报错信息
        err_msg = e.stderr.strip() if e.stderr else str(e)
        logger.error(f"[/undo] Git reset 真实报错: {err_msg}")

        # 把真实原因打印到前端！
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=f"❌ 撤销命令执行失败！\n**真实原因:**\n```text\n{err_msg}\n```"
        )
    except Exception as e:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=f"❌ 撤销过程中发生未知错误: {e}"
        )

def build_help_text() -> str:
    """Build canonical help text shared across channels."""
    lines = [
        "👻 ghostbot 命令：",
        "/new — 清空当前项目对话",
        "/stop — 停止当前任务",
        "/restart — 重启 bot",
        "/status — 显示 bot 状态",
        "/dream — 手动触发 Dream 整理",
        "/dream-log — 查看上次 Dream 修改了什么",
        "/dream-restore — 将记忆恢复到历史版本",
        "/plan <request> — 创建需要批准的可执行计划",
        "/plan-status [--full] — 查看待处理计划",
        "/plan-history — 查看最近和本地归档计划",
        "/plan-checklist [--full] — 查看待处理计划的检查清单",
        "/plan-approve [all|phase N|phases A-B] — 执行待处理计划或指定范围",
        "/plan-load <plan_id-or-path> — 加载本地保存的计划",
        "/plan-cancel — 取消待处理计划",
        "/plan-revise <feedback> — 修订待处理计划",
        "/scan [路径] — 扫描真实项目路径，保存架构图，并设为 Active Project",
        "/use [项目名] — 从已扫描项目切换当前 Active Project",
        "/help — 显示可用命令",
    ]
    return "\n".join(lines)


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/dream", cmd_dream)
    router.exact("/dream-log", cmd_dream_log)
    router.prefix("/dream-log ", cmd_dream_log)
    router.exact("/dream-restore", cmd_dream_restore)
    router.prefix("/dream-restore ", cmd_dream_restore)
    router.exact("/plan", cmd_plan)
    router.prefix("/plan ", cmd_plan)
    router.exact("/plan-status", cmd_plan_status)
    router.prefix("/plan-status ", cmd_plan_status)
    router.exact("/plan-history", cmd_plan_history)
    router.exact("/plan-checklist", cmd_plan_checklist)
    router.prefix("/plan-checklist ", cmd_plan_checklist)
    router.exact("/plan-approve", cmd_plan_approve)
    router.prefix("/plan-approve ", cmd_plan_approve)
    router.prefix("/plan-load ", cmd_plan_load)
    router.exact("/plan-cancel", cmd_plan_cancel)
    router.prefix("/plan-revise ", cmd_plan_revise)
    router.exact("/help", cmd_help)
    router.exact("/scan", cmd_scan)  # 匹配纯 "/scan"
    router.prefix("/scan ", cmd_scan)  # 匹配 "/scan D:/项目"
    router.prefix("/use ", cmd_use)
    router.exact("/undo", cmd_undo)
    router.intercept(tool_approval_interceptor)
    router.intercept(plan_approval_interceptor)

"""Runtime assembly helpers shared by CLI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

from ghostbot.agent.loop import AgentLoop
from ghostbot.bus.queue import MessageBus
from ghostbot.config.paths import is_default_workspace
from ghostbot.config.schema import Config
from ghostbot.cron.service import CronService
from ghostbot.project import ProjectManager
from ghostbot.providers.base import LLMProvider
from ghostbot.providers.factory import build_provider_for_fast_model
from ghostbot.utils.helpers import sync_workspace_templates


@dataclass(slots=True)
class AgentRuntime:
    config: Config
    bus: MessageBus
    provider: LLMProvider
    agent_loop: AgentLoop
    cron_service: CronService


@dataclass(slots=True)
class ApiRuntime:
    config: Config
    provider: LLMProvider
    agent_loop: AgentLoop


def migrate_cron_store(config: Config) -> None:
    """One-time migration: move legacy global cron store into the workspace."""
    from ghostbot.config.paths import get_cron_dir

    legacy_path = get_cron_dir() / "jobs.json"
    new_path = config.workspace_path / "cron" / "jobs.json"
    if legacy_path.is_file() and not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.move(str(legacy_path), str(new_path))


def prepare_workspace(config: Config) -> None:
    sync_workspace_templates(config.workspace_path)
    if is_default_workspace(config.workspace_path):
        migrate_cron_store(config)


def create_cron_service(config: Config) -> CronService:
    cron_store_path = config.workspace_path / "cron" / "jobs.json"
    return CronService(cron_store_path)


def build_agent_loop(
    config: Config,
    *,
    bus: MessageBus,
    provider: LLMProvider,
    cron_service: CronService | None = None,
    fast_provider: LLMProvider | None = None,
) -> AgentLoop:
    project_manager = ProjectManager(config.workspace_path)
    return AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        fast_provider=fast_provider,
        fast_model=config.get_effective_fast_model(),
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_config=config.tools.web,
        context_block_limit=config.agents.defaults.context_block_limit,
        approved_plan_context_block_limit=config.agents.defaults.approved_plan_context_block_limit,
        max_tool_result_chars=config.agents.defaults.max_tool_result_chars,
        provider_retry_mode=config.agents.defaults.provider_retry_mode,
        exec_config=config.tools.exec,
        cron_service=cron_service,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=project_manager,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        timezone=config.agents.defaults.timezone,
        unified_session=config.agents.defaults.unified_session,
        disabled_skills=config.agents.defaults.disabled_skills,
        session_ttl_minutes=config.agents.defaults.session_ttl_minutes,
        planning_config=config.agents.defaults.planning,
    )


def build_api_runtime(config: Config, provider: LLMProvider) -> ApiRuntime:
    prepare_workspace(config)
    bus = MessageBus()
    agent_loop = build_agent_loop(config, bus=bus, provider=provider)
    return ApiRuntime(config=config, provider=provider, agent_loop=agent_loop)


def build_agent_runtime(config: Config, provider: LLMProvider) -> AgentRuntime:
    prepare_workspace(config)
    bus = MessageBus()
    cron_service = create_cron_service(config)
    fast_build = build_provider_for_fast_model(config)
    fast_provider = fast_build.provider if fast_build is not None else provider
    agent_loop = build_agent_loop(
        config,
        bus=bus,
        provider=provider,
        cron_service=cron_service,
        fast_provider=fast_provider,
    )
    return AgentRuntime(
        config=config,
        bus=bus,
        provider=provider,
        agent_loop=agent_loop,
        cron_service=cron_service,
    )

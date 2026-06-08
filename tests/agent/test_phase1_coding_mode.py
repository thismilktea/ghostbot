from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

import pytest

from ghostbot.agent.context import ContextBuilder
from ghostbot.agent.loop import AgentLoop
from ghostbot.agent.policy import PolicyContext, PolicyEngine
from ghostbot.agent.planning import PlanState
from ghostbot.agent.runner import AgentRunSpec, AgentRunner
from ghostbot.bus.events import InboundMessage
from ghostbot.bus.queue import MessageBus
from ghostbot.command.router import CommandContext
from ghostbot.config.schema import CodingModeConfig
from ghostbot.agent.tools.base import ToolResult
from ghostbot.agent.tools.shell import ExecTool
from ghostbot.providers.base import LLMResponse, ToolCallRequest
from ghostbot.utils.helpers import build_status_content


class _FakeProvider:
    def __init__(self, response: LLMResponse):
        self._response = response
        self.generation = MagicMock(max_tokens=4096)

    def get_default_model(self):
        return "test-model"

    async def chat_with_retry(self, **kwargs):
        return self._response

    async def chat_stream_with_retry(self, **kwargs):
        return self._response


class _SequencedProvider:
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.generation = MagicMock(max_tokens=4096)

    def get_default_model(self):
        return "test-model"

    async def chat_with_retry(self, **kwargs):
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    async def chat_stream_with_retry(self, **kwargs):
        return await self.chat_with_retry(**kwargs)


class _LoopProvider:
    generation = MagicMock(max_tokens=4096)

    def get_default_model(self):
        return "test-model"



def test_coding_mode_config_defaults_are_restrictive():
    cfg = CodingModeConfig()
    assert cfg.enable is False
    assert cfg.approval_mode == "manual"
    assert cfg.allow_write is False
    assert cfg.allow_exec is False
    assert cfg.allow_web is False
    assert cfg.allow_mcp is False
    assert cfg.allow_spawn is False
    assert cfg.allow_cron is False


def test_windows_shell_preflight_flags_unix_only_patterns():
    assert "`/dev/null` is unavailable" in ExecTool._windows_preflight_error("python app.py >/dev/null 2>&1")
    assert "read_file" in ExecTool._windows_preflight_error("cat README.md")
    assert "grep tool" in ExecTool._windows_preflight_error("grep -R needle .")
    assert "heredoc" in ExecTool._windows_preflight_error("python - <<'PY'")
    assert ExecTool._windows_preflight_error("python -m pytest") is None


@pytest.mark.asyncio
async def test_runner_blocks_write_tool_when_approval_is_required(tmp_path):
    provider = _FakeProvider(
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="write_file", arguments={"path": "a.txt", "content": "x"})],
            usage={},
        )
    )
    runner = AgentRunner(provider)
    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=MagicMock(),
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        require_approval_for_write=True,
        mode="coding",
        approval_mode="manual",
    )
    spec.tools.get_definitions.return_value = []
    spec.tools.execute = AsyncMock()

    result = await runner.run(spec)

    assert result.stop_reason == "tool_error"
    assert "requires approval" in (result.error or "")
    spec.tools.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_blocks_web_tools_when_external_tools_disabled(tmp_path):
    provider = _FakeProvider(
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="web_search", arguments={"query": "x"})],
            usage={},
        )
    )
    runner = AgentRunner(provider)
    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=MagicMock(),
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        allow_external_tools=False,
        mode="coding",
    )
    spec.tools.get_definitions.return_value = []
    spec.tools.execute = AsyncMock()

    result = await runner.run(spec)

    assert result.stop_reason == "tool_error"
    assert "disabled" in (result.error or "")
    spec.tools.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_allows_read_tool_calls(tmp_path):
    provider = _SequencedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="1", name="read_file", arguments={"path": "a.txt"})],
                usage={},
            ),
            LLMResponse(content="done", tool_calls=[], usage={}),
        ]
    )
    runner = AgentRunner(provider)
    tool = MagicMock()
    tool.read_only = True
    tool.concurrency_safe = True
    tool.execute = AsyncMock(return_value="ok")
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.prepare_call.return_value = (tool, {"path": "a.txt"}, None)
    tools.execute = AsyncMock(return_value="ok")

    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        mode="coding",
        allowed_tool_names={"read_file"},
    )

    result = await runner.run(spec)

    assert result.stop_reason == "completed"
    assert result.final_content == "done"
    tool.execute.assert_called_once_with(path="a.txt")


@pytest.mark.asyncio
async def test_runner_omits_repeated_lookup_until_write_invalidates(tmp_path):
    provider = _SequencedProvider([
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="read_file", arguments={"path": "a.txt"})],
            usage={},
        ),
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="2", name="read_file", arguments={"path": "a.txt"})],
            usage={},
        ),
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="3", name="write_file", arguments={"path": "a.txt", "content": "x"})],
            usage={},
        ),
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="4", name="read_file", arguments={"path": "a.txt"})],
            usage={},
        ),
        LLMResponse(content="done", tool_calls=[], usage={}),
    ])
    runner = AgentRunner(provider)
    read_tool = MagicMock(read_only=True, concurrency_safe=True)
    read_tool.execute = AsyncMock(return_value="file contents")
    write_tool = MagicMock(read_only=False, side_effect_level="workspace_write", risk_tags=frozenset({"filesystem_write"}))
    write_tool.execute = AsyncMock(return_value="wrote")
    tools = MagicMock()
    tools.get_definitions.return_value = []

    def prepare_call(name, args):
        return (write_tool if name == "write_file" else read_tool), args, None

    tools.prepare_call.side_effect = prepare_call
    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=tools,
        model="test-model",
        max_iterations=5,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        mode="coding",
        policy_engine=PolicyEngine(),
        policy_context=PolicyContext(approved_plan_id="plan_1", allowed_paths=["a.txt"]),
    )

    result = await runner.run(spec)

    assert result.stop_reason == "completed"
    assert result.final_content == "done"
    assert read_tool.execute.await_count == 2
    assert write_tool.execute.await_count == 1
    repeated = [m for m in result.messages if m.get("role") == "tool" and "Repeated read_file omitted" in m.get("content", "")]
    assert len(repeated) == 1


@pytest.mark.asyncio
async def test_runner_keeps_tool_result_metadata_out_of_model_content(tmp_path):
    provider = _SequencedProvider([
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="write_file", arguments={"path": "a.txt", "content": "x"})],
            usage={},
        ),
        LLMResponse(content="done", tool_calls=[], usage={}),
    ])
    runner = AgentRunner(provider)
    tool = MagicMock(read_only=False, side_effect_level="workspace_write", risk_tags=frozenset({"filesystem_write"}))
    tool.execute = AsyncMock(return_value=ToolResult(
        "wrote",
        {"change_summary": {"formatted": "Modified a.txt (+1 -0)", "additions": 1, "removals": 0}},
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.prepare_call.return_value = (tool, {"path": "a.txt", "content": "x"}, None)
    progress = AsyncMock()
    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        progress_callback=progress,
    )

    result = await runner.run(spec)

    tool_message = next(m for m in result.messages if m.get("role") == "tool")
    assert tool_message["content"] == "wrote"
    assert result.tool_events[0]["change_summary"]["formatted"] == "Modified a.txt (+1 -0)"
    progress.assert_awaited_with("Modified a.txt (+1 -0)", change_summary=True)


    provider = _SequencedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="1", name="write_file", arguments={"path": "b.txt", "content": "x"})],
                usage={},
            ),
            LLMResponse(content="done", tool_calls=[], usage={}),
        ]
    )
    runner = AgentRunner(provider)
    tool = MagicMock()
    tool.read_only = False
    tool.side_effect_level = "workspace_write"
    tool.risk_tags = frozenset({"filesystem_write"})
    tool.execute = AsyncMock(return_value="wrote")
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.prepare_call.return_value = (tool, {"path": "b.txt", "content": "x"}, None)

    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        mode="coding",
        policy_engine=PolicyEngine(),
        policy_context=PolicyContext(approved_plan_id="plan_1", allowed_paths=["a.txt"]),
    )

    result = await runner.run(spec)

    assert result.stop_reason == "completed"
    assert result.final_content == "done"
    tool.execute.assert_awaited_once_with(path="b.txt", content="x")


@pytest.mark.asyncio
async def test_policy_denies_write_outside_allowed_roots(tmp_path):
    outside = tmp_path.parent / "outside" / "b.txt"
    provider = _FakeProvider(
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="write_file", arguments={"path": str(outside), "content": "x"})],
            usage={},
        )
    )
    runner = AgentRunner(provider)
    tool = MagicMock()
    tool.read_only = False
    tool.side_effect_level = "workspace_write"
    tool.risk_tags = frozenset({"filesystem_write"})
    tool.execute = AsyncMock(return_value="wrote")
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.prepare_call.return_value = (tool, {"path": str(outside), "content": "x"}, None)

    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        mode="coding",
        policy_engine=PolicyEngine(),
        policy_context=PolicyContext(approved_plan_id="plan_1"),
    )

    result = await runner.run(spec)

    assert result.stop_reason == "completed"
    assert "outside allowed roots" in result.messages[-1]["content"]
    tool.execute.assert_not_awaited()
    assert result.tool_events[0]["policy_decision"] == "deny"


@pytest.mark.asyncio
async def test_policy_allows_write_inside_active_project_scope(tmp_path):
    project_dir = tmp_path.parent / "active_project"
    provider = _SequencedProvider([
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="write_file", arguments={"path": str(project_dir / "app.py"), "content": "x"})],
            usage={},
        ),
        LLMResponse(content="done", tool_calls=[], usage={}),
    ])
    runner = AgentRunner(provider)
    tool = MagicMock()
    tool.read_only = False
    tool.side_effect_level = "workspace_write"
    tool.risk_tags = frozenset({"filesystem_write"})
    tool.execute = AsyncMock(return_value="wrote")
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.prepare_call.return_value = (tool, {"path": str(project_dir / "app.py"), "content": "x"}, None)

    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        mode="coding",
        policy_engine=PolicyEngine(),
        policy_context=PolicyContext(approved_plan_id="plan_1", allowed_paths=[str(project_dir / "app.py")], allowed_roots=[str(project_dir)]),
    )

    result = await runner.run(spec)

    assert result.stop_reason == "completed"
    assert result.final_content == "done"
    tool.execute.assert_awaited_once_with(path=str(project_dir / "app.py"), content="x")


@pytest.mark.asyncio
async def test_policy_allows_write_inside_approved_plan_scope(tmp_path):
    provider = _SequencedProvider([
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="write_file", arguments={"path": "a.txt", "content": "x"})],
            usage={},
        ),
        LLMResponse(content="done", tool_calls=[], usage={}),
    ])
    runner = AgentRunner(provider)
    tool = MagicMock()
    tool.read_only = False
    tool.side_effect_level = "workspace_write"
    tool.risk_tags = frozenset({"filesystem_write"})
    tool.execute = AsyncMock(return_value="wrote")
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.prepare_call.return_value = (tool, {"path": "a.txt", "content": "x"}, None)

    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        mode="coding",
        policy_engine=PolicyEngine(),
        policy_context=PolicyContext(approved_plan_id="plan_1", allowed_paths=["a.txt"]),
    )

    result = await runner.run(spec)

    assert result.stop_reason == "completed"
    assert result.final_content == "done"
    tool.execute.assert_awaited_once_with(path="a.txt", content="x")


@pytest.mark.asyncio
async def test_policy_allows_shell_inside_approved_plan_scope(tmp_path):
    provider = _SequencedProvider([
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="exec", arguments={"command": "python -m pip --version"})],
            usage={},
        ),
        LLMResponse(content="done", tool_calls=[], usage={}),
    ])
    runner = AgentRunner(provider)
    tool = MagicMock()
    tool.read_only = False
    tool.side_effect_level = "process"
    tool.risk_tags = frozenset({"shell"})
    tool.execute = AsyncMock(return_value="pip 1.0")
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.prepare_call.return_value = (tool, {"command": "python -m pip --version"}, None)

    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        mode="coding",
        policy_engine=PolicyEngine(),
        policy_context=PolicyContext(approved_plan_id="plan_1", allowed_paths=["a.txt"]),
    )

    result = await runner.run(spec)

    assert result.stop_reason == "completed"
    assert result.final_content == "done"
    tool.execute.assert_awaited_once_with(command="python -m pip --version")


@pytest.mark.asyncio
async def test_policy_requires_approval_for_shell_without_approved_plan(tmp_path):
    provider = _FakeProvider(
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="exec", arguments={"command": "python -m pip --version"})],
            usage={},
        )
    )
    runner = AgentRunner(provider)
    tool = MagicMock()
    tool.read_only = False
    tool.side_effect_level = "process"
    tool.risk_tags = frozenset({"shell"})
    tool.execute = AsyncMock(return_value="should not run")
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.prepare_call.return_value = (tool, {"command": "python -m pip --version"}, None)

    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        mode="coding",
        policy_engine=PolicyEngine(),
        policy_context=PolicyContext(),
    )

    result = await runner.run(spec)

    assert result.stop_reason == "approval_required"
    assert result.error is None
    assert result.pending_approval is not None
    assert result.pending_approval["id"] == "1"
    assert result.pending_approval["tool_name"] == "exec"
    assert result.pending_approval["arguments"] == {"command": "python -m pip --version"}
    assert result.pending_approval["reason"] == "Shell command requires an approved executable plan"
    assert result.pending_approval["assistant_message"] == result.messages[-1]
    assert len(result.messages) == 2
    assert result.messages[-1]["role"] == "assistant"
    assert result.messages[-1].get("tool_calls")
    assert "approved executable plan" in (result.final_content or "")
    assert "Analyze the error above" not in (result.final_content or "")
    tool.execute.assert_not_awaited()
    assert result.tool_events[0]["policy_decision"] == "require_approval"


@pytest.mark.asyncio
async def test_policy_denies_destructive_shell_before_execution(tmp_path):
    provider = _FakeProvider(
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="exec", arguments={"command": "rm -rf ."})],
            usage={},
        )
    )
    runner = AgentRunner(provider)
    tool = MagicMock()
    tool.read_only = False
    tool.side_effect_level = "process"
    tool.risk_tags = frozenset({"shell"})
    tool.execute = AsyncMock(return_value="should not run")
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.prepare_call.return_value = (tool, {"command": "rm -rf ."}, None)

    spec = AgentRunSpec(
        initial_messages=[{"role": "user", "content": "test"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1024,
        workspace=tmp_path,
        session_key="cli:test",
        mode="coding",
        policy_engine=PolicyEngine(),
        policy_context=PolicyContext(),
    )

    result = await runner.run(spec)

    assert result.stop_reason == "completed"
    assert "destructive pattern" in result.messages[-1]["content"]
    assert "Analyze the error above" not in result.messages[-1]["content"]
    tool.execute.assert_not_awaited()
    assert result.tool_events[0]["policy_decision"] == "deny"


    text = build_status_content(
        version="1.0",
        model="m",
        start_time=0,
        last_usage={},
        context_window_tokens=1000,
        session_msg_count=3,
        context_tokens_estimate=120,
        search_usage_text=None,
        mode="coding",
        repo_root="/repo",
        branch="main",
        approval_mode="manual",
        isolation="none",
        last_action="completed",
        active_project="demo",
        active_project_path="/project/demo",
    )
    assert "ghostbot" in text
    assert "Model:" in text
    assert "Mode: coding" in text
    assert "Repo: main (/repo)" in text
    assert "Approval: manual" in text
    assert "Isolation: none" in text
    assert "Last action: completed" in text
    assert "Active Project: demo (/project/demo)" in text


def test_context_builder_uses_active_project_name(tmp_path):
    projects_dir = tmp_path / "memory" / "projects"
    projects_dir.mkdir(parents=True)
    (projects_dir / "demo.md").write_text("demo body", encoding="utf-8")

    builder = ContextBuilder(tmp_path)
    prompt = builder.build_system_prompt(
        session_metadata={
            "active_project": "demo",
            "active_project_path": r"D:\\demo",
        }
    )

    assert '<active_project name="demo" path="D:\\\\demo">' in prompt
    assert "Project path: D:\\\\demo" in prompt
    assert "demo body" in prompt
    assert "Workspace" in prompt
    assert "internal storage" in prompt


def test_loop_allows_active_project_for_file_tools(tmp_path):
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_LoopProvider(),
        workspace=tmp_path,
        restrict_to_workspace=True,
    )
    session = loop.sessions.get_or_create("cli:direct")
    active_project = tmp_path.parent / "active_project"
    session.metadata["active_project_path"] = str(active_project)

    loop._allow_active_project_tools(session)

    read_tool = loop.tools.get("read_file")
    write_tool = loop.tools.get("write_file")
    assert read_tool is not None
    assert write_tool is not None
    assert active_project.resolve() in [Path(p).resolve() for p in read_tool._extra_allowed_dirs]
    assert active_project.resolve() in [Path(p).resolve() for p in write_tool._extra_allowed_dirs]


def test_policy_context_falls_back_to_plan_active_project_path(tmp_path):
    from ghostbot.agent.planning import PlanState

    active_project = tmp_path.parent / "active_project"
    plan = PlanState.create("build app", "## Plan\n- do it")
    plan.active_project_path = str(active_project)

    context = AgentLoop._build_policy_context(plan, None)

    assert context.allowed_roots == [str(active_project.resolve())]


@pytest.mark.asyncio
async def test_loop_stores_pending_approval_and_prompts_user(tmp_path):
    provider = _FakeProvider(
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="exec", arguments={"command": "python script.py"})],
            usage={},
        )
    )
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="run it")

    PlanState.clear_active()
    PlanState.clear_active(tmp_path)
    result = await loop.process_direct("run it")

    assert result is not None
    assert result.content == (
        "Tool approval required.\n\n"
        "Tool: exec\n"
        "Reason: Shell command requires an approved executable plan\n\n"
        "Reply `approve` to run it or `deny` to skip it."
    )
    session = loop.sessions.get_or_create("cli:direct")
    pending = session.metadata[loop._PENDING_APPROVAL_KEY]
    assert pending["id"] == "1"
    assert pending["tool_name"] == "exec"
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1].get("tool_calls")


@pytest.mark.asyncio
async def test_pending_approval_intercepts_normal_text_with_reminder(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_LoopProvider(), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    session.metadata[loop._PENDING_APPROVAL_KEY] = {
        "id": "1",
        "tool_name": "exec",
        "reason": "Shell command requires approval",
    }
    loop.runner.run = AsyncMock()
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="what now?")

    result = await loop.process_direct("what now?")

    assert result is not None
    assert "A tool approval is pending." in result.content
    assert "Tool: exec" in result.content
    assert "approve" in result.content
    loop.runner.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_approval_deny_appends_tool_result_and_resumes(tmp_path):
    provider = _SequencedProvider([LLMResponse(content="skipped", tool_calls=[], usage={})])
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    session.messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "1",
            "type": "function",
            "function": {"name": "exec", "arguments": '{"command":"python script.py"}'},
        }],
    })
    session.metadata[loop._PENDING_APPROVAL_KEY] = {
        "id": "1",
        "tool_name": "exec",
        "arguments": {"command": "python script.py"},
        "reason": "Shell command requires approval",
    }
    ctx = CommandContext(
        msg=InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="deny"),
        session=session,
        key="cli:direct",
        raw="deny",
        loop=loop,
    )

    result = await loop._resolve_pending_tool_approval(ctx, approved=False)

    assert result.content == "skipped"
    assert loop._PENDING_APPROVAL_KEY not in session.metadata
    tool_messages = [msg for msg in session.messages if msg.get("role") == "tool"]
    assert tool_messages[0]["tool_call_id"] == "1"
    assert tool_messages[0]["name"] == "exec"
    assert tool_messages[0]["content"] == "Tool call denied by user."


@pytest.mark.asyncio
async def test_pending_approval_approve_executes_saved_tool_once_and_resumes(tmp_path):
    provider = _SequencedProvider([LLMResponse(content="continued", tool_calls=[], usage={})])
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    tool = MagicMock()
    tool.execute = AsyncMock(return_value="ran")
    loop.tools.prepare_call = MagicMock(return_value=(tool, {"command": "python script.py"}, None))
    session = loop.sessions.get_or_create("cli:direct")
    session.messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "1",
            "type": "function",
            "function": {"name": "exec", "arguments": '{"command":"python script.py"}'},
        }],
    })
    session.metadata[loop._PENDING_APPROVAL_KEY] = {
        "id": "1",
        "tool_name": "exec",
        "arguments": {"command": "python script.py"},
        "reason": "Shell command requires approval",
    }
    ctx = CommandContext(
        msg=InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="approve"),
        session=session,
        key="cli:direct",
        raw="approve",
        loop=loop,
    )

    result = await loop._resolve_pending_tool_approval(ctx, approved=True)

    assert result.content == "continued"
    loop.tools.prepare_call.assert_called_once_with("exec", {"command": "python script.py"})
    tool.execute.assert_awaited_once_with(command="python script.py")
    assert loop._PENDING_APPROVAL_KEY not in session.metadata
    tool_messages = [msg for msg in session.messages if msg.get("role") == "tool"]
    assert tool_messages[0]["tool_call_id"] == "1"
    assert tool_messages[0]["name"] == "exec"
    assert tool_messages[0]["content"] == "ran"

def test_default_allowed_tool_names_include_graph_queries(tmp_path):
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_LoopProvider(),
        workspace=tmp_path,
        coding_config=CodingModeConfig(enable=True),
    )

    assert loop.restrict_to_workspace is True
    assert loop._allowed_tool_names() == {
        "read_file",
        "list_dir",
        "glob",
        "grep",
        "find_symbol",
        "find_callers",
        "find_callees",
        "find_related_files",
        "find_impacted_files",
    }
    assert loop._blocked_tool_prefixes() == ("mcp_",)

from __future__ import annotations

from unittest.mock import MagicMock
import re

import pytest

from ghostbot.agent.loop import AgentLoop
from ghostbot.agent.planning import (
    build_execution_contract,
    ExecutionMode,
    PlanState,
    active_plan_path,
    plan_content_hash,
    plan_history_path,
    validate_plan_quality,
)
from ghostbot.bus.events import InboundMessage, OutboundMessage
from ghostbot.bus.queue import MessageBus
from ghostbot.command.builtin import (
    cmd_plan_approve,
    cmd_plan_cancel,
    cmd_plan_checklist,
    cmd_plan_history,
    cmd_plan_load,
    cmd_plan_revise,
    cmd_plan_status,
    cmd_use,
    plan_approval_interceptor,
)
from ghostbot.command.router import CommandContext
from ghostbot.config.schema import PlanningConfig
from ghostbot.providers.base import LLMResponse


VALID_PLAN = """## Execution Mode
executable.

## Summary
Update the focused planning behavior and verify it with targeted tests.

## User Intent
- Make the focused planning test behavior pass without changing unrelated planning behavior.

## Requirements
- Fix the requested test behavior.

## Acceptance Criteria
- [ ] The focused planning flow test passes.
- [ ] The change is limited to planning validation behavior.

## Non-goals / Out of Scope
- Do not refactor unrelated agent loop behavior.

## Exploration Evidence
- `grep` searched `tests` for the failing behavior; this identifies the target test file.

## Proposed Approach
Update the focused implementation used by the failing test.

## Files Likely to Change
- `ghostbot/agent/planning.py` — planning validation logic.

## Executable Checklist
- [ ] Update the focused implementation.
- [ ] Run focused tests.

## Verification Plan
- `python -m pytest tests/agent/test_planning_flow.py`

## Risks and Open Questions
- Risk is limited to planning behavior.
"""


LARGE_PLAN = """## Execution Mode
executable.

## Summary
Implement a staged shop upgrade without doing all phases at once.

## User Intent
- Upgrade the shop in phases.

## Requirements
- Use staged execution.

## Acceptance Criteria
- [ ] 阶段 1：cart storage is updated.
- [ ] 阶段 1：cart badge still works.
- [ ] 阶段 2：cart preview API exists.
- [ ] 阶段 2：cart page uses preview API.
- [ ] 阶段 3：orders are persisted.
- [ ] 阶段 3：checkout creates pending order.
- [ ] 阶段 4：pay endpoint exists.
- [ ] 阶段 4：payment deducts stock.
- [ ] 阶段 5：search works.
- [ ] 阶段 5：sorting works.
- [ ] 阶段 6：admin token is required.
- [ ] 阶段 6：admin products can be edited.

## Non-goals / Out of Scope
- Do not implement real payment.

## Exploration Evidence
- `read_file` inspected relevant files.

## Proposed Approach
- 阶段 1：购物车结构。
- 阶段 2：购物车预览 API。
- 阶段 3：订单持久化。
- 阶段 4：模拟支付。
- 阶段 5：搜索排序。
- 阶段 6：后台管理。

## Files Likely to Change
- `app/models.py` — data model.
- `frontend/js/cart.js` — cart storage.

## Executable Checklist
- [ ] 阶段 1：修改购物车结构。
- [ ] 阶段 1：迁移旧购物车。
- [ ] 阶段 2：新增购物车预览 API。
- [ ] 阶段 2：前端接入预览 API。
- [ ] 阶段 3：新增订单模型。
- [ ] 阶段 3：checkout 创建待支付订单。
- [ ] 阶段 4：新增模拟支付接口。
- [ ] 阶段 4：支付成功扣库存。
- [ ] 阶段 5：新增搜索筛选排序。
- [ ] 阶段 6：新增后台 token 校验。
- [ ] 阶段 6：新增后台页面。

## Verification Plan
- Run pytest.

## Risks and Open Questions
- Admin token default needs documentation.
"""


class _Provider:
    def __init__(self, contents: str | list[str] = VALID_PLAN) -> None:
        self.contents = list(contents) if isinstance(contents, list) else [contents]
        self.calls: list[dict] = []
        self.generation = MagicMock(max_tokens=4096)

    def get_default_model(self):
        return "test-model"

    async def chat_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        return LLMResponse(content=content, tool_calls=[], usage={})

    async def chat_stream_with_retry(self, **kwargs):
        return await self.chat_with_retry(**kwargs)


def _msg(content: str) -> InboundMessage:
    return InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=content)


def _ctx(loop: AgentLoop, content: str) -> CommandContext:
    msg = _msg(content)
    session = loop.sessions.get_or_create(msg.session_key)
    return CommandContext(msg=msg, session=session, key=msg.session_key, raw=content, loop=loop)


async def _plan(loop: AgentLoop, request: str = "fix the bug in the test file") -> OutboundMessage:
    session = loop.sessions.get_or_create("cli:direct")
    plan = await loop._create_pending_plan(
        session=session,
        key="cli:direct",
        request=request,
        channel="cli",
        chat_id="direct",
    )
    return OutboundMessage(
        channel="cli",
        chat_id="direct",
        content=loop._format_plan_response(plan) if isinstance(plan, PlanState) else plan,
        metadata={"render_as": "text"},
    )


def _saved_plan(session, *, execution_mode: ExecutionMode = "executable") -> PlanState:
    plan = PlanState.create("fix tests", VALID_PLAN, task_class="research_only", execution_mode=execution_mode)
    plan.save_to_session(session)
    return plan


@pytest.fixture(autouse=True)
def _plan_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("ghostbot.agent.planning.get_workspace_path", lambda workspace=None: tmp_path)


@pytest.mark.asyncio
async def test_plan_response_is_concise_by_default_and_full_on_request(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider(LARGE_PLAN), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = PlanState.create("upgrade shop", LARGE_PLAN)
    plan.save_to_session(session)

    concise = loop._format_plan_response(plan)
    full = loop._format_plan_response(plan, full=True)

    assert "## 摘要" in concise
    assert "## 阶段 / 检查清单" in concise
    assert "使用 `/plan-status --full`" in concise
    assert "## Exploration Evidence" not in concise
    assert "## Exploration Evidence" in full


@pytest.mark.asyncio
async def test_plan_save_creates_active_plan_file(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider(), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")

    plan = _saved_plan(session)

    assert active_plan_path().exists()
    loaded = PlanState.load_active()
    assert loaded is not None
    assert loaded.id == plan.id


@pytest.mark.asyncio
async def test_plan_load_recovers_from_active_file_without_session_metadata(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider(), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = _saved_plan(session)
    session.metadata.clear()

    loaded = PlanState.from_session(session)

    assert loaded is not None
    assert loaded.id == plan.id


@pytest.mark.asyncio
async def test_legacy_metadata_plan_loads_without_active_file(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider(), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = PlanState.create("fix tests", VALID_PLAN)
    session.metadata["pending_plan"] = plan.to_dict()
    PlanState.clear_active()

    loaded = PlanState.from_session(session)

    assert loaded is not None
    assert loaded.id == plan.id


@pytest.mark.asyncio
async def test_plan_cancel_archives_and_clears_active_file(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider(), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = _saved_plan(session)

    result = await cmd_plan_cancel(_ctx(loop, "/plan-cancel"))

    assert f"已取消计划 {plan.id}" in result.content
    assert not active_plan_path().exists()
    assert plan_history_path(plan.id).exists()
    assert PlanState.from_session(session) is None


@pytest.mark.asyncio
async def test_plan_approve_archives_completed_plan(tmp_path):
    provider = _Provider("implemented")
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = _saved_plan(session)

    result = await cmd_plan_approve(_ctx(loop, "/plan-approve"))

    assert isinstance(result, OutboundMessage)
    assert result.content == "implemented"
    assert not active_plan_path().exists()
    assert plan_history_path(plan.id).exists()


@pytest.mark.asyncio
async def test_blocked_plan_persists_block_reason(tmp_path):
    provider = _Provider("Error: Tool 'exec' require_approval: blocked")
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = _saved_plan(session)
    plan.mark_approved("yes")
    plan.save_to_session(session)

    result = await loop._execute_approved_plan(msg=_msg("yes"), session=session, plan=plan)
    loaded = PlanState.from_session(session)

    assert isinstance(result, OutboundMessage)
    assert loaded is not None
    assert loaded.status == "blocked"
    assert loaded.block_reason == "Error: Tool 'exec' require_approval: blocked"


@pytest.mark.asyncio
async def test_normal_message_does_not_auto_plan(tmp_path):
    provider = _Provider("answer")
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        planning_config=PlanningConfig(mode="auto", min_exploration_steps=0),
    )

    result = await loop._process_message(_msg("fix the bug in the test file"))
    session = loop.sessions.get_or_create("cli:direct")

    assert result is not None
    assert result.content == "answer"
    assert PlanState.from_session(session) is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_code_change_without_exploration_fails_quality(tmp_path):
    provider = _Provider(VALID_PLAN)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        planning_config=PlanningConfig(mode="auto", min_exploration_steps=1, max_rewrites=0),
    )

    result = await _plan(loop)

    assert result is not None
    assert result.content.startswith("计划质量检查未通过")
    assert PlanState.from_session(loop.sessions.get_or_create("cli:direct")) is None


@pytest.mark.asyncio
async def test_missing_structure_triggers_rewrite(tmp_path):
    provider = _Provider(["weak plan", VALID_PLAN])
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        planning_config=PlanningConfig(mode="auto", min_exploration_steps=0, max_rewrites=1),
    )

    result = await _plan(loop)
    plan = PlanState.from_session(loop.sessions.get_or_create("cli:direct"))

    assert result is not None
    assert result.content.startswith("计划已生成")
    assert plan is not None
    assert plan.plan == VALID_PLAN
    assert len(provider.calls) == 2
    assert "Quality failures" in provider.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_rewritten_invalid_plan_is_not_saved(tmp_path):
    provider = _Provider(["weak plan", "still weak"])
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        planning_config=PlanningConfig(mode="auto", min_exploration_steps=0, max_rewrites=1),
    )

    result = await _plan(loop)

    assert result is not None
    assert result.content.startswith("计划质量检查未通过")
    assert PlanState.from_session(loop.sessions.get_or_create("cli:direct")) is None


@pytest.mark.asyncio
async def test_missing_acceptance_criteria_triggers_rewrite(tmp_path):
    weak_plan = VALID_PLAN.replace("## Acceptance Criteria\n- [ ] The focused planning flow test passes.\n- [ ] The change is limited to planning validation behavior.\n\n", "")
    provider = _Provider([weak_plan, VALID_PLAN])
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        planning_config=PlanningConfig(mode="auto", min_exploration_steps=0, max_rewrites=1),
    )

    result = await _plan(loop)
    plan = PlanState.from_session(loop.sessions.get_or_create("cli:direct"))

    assert result is not None
    assert result.content.startswith("计划已生成")
    assert plan is not None
    assert plan.plan == VALID_PLAN
    assert len(provider.calls) == 2
    assert "Acceptance Criteria" in provider.calls[1]["messages"][-1]["content"]


def test_placeholder_acceptance_criteria_fails_quality():
    weak_plan = VALID_PLAN.replace(
        "## Acceptance Criteria\n- [ ] The focused planning flow test passes.\n- [ ] The change is limited to planning validation behavior.",
        "## Acceptance Criteria\n- TBD",
    )

    quality = validate_plan_quality(
        weak_plan,
        task_class="code_change_small",
        tools_used=[],
        min_exploration_steps=0,
    )

    assert not quality.passed
    assert any("Acceptance Criteria" in failure for failure in quality.failures)


@pytest.mark.asyncio
async def test_missing_non_goals_triggers_rewrite(tmp_path):
    weak_plan = VALID_PLAN.replace("## Non-goals / Out of Scope\n- Do not refactor unrelated agent loop behavior.\n\n", "")
    provider = _Provider([weak_plan, VALID_PLAN])
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        planning_config=PlanningConfig(mode="auto", min_exploration_steps=0, max_rewrites=1),
    )

    result = await _plan(loop)
    plan = PlanState.from_session(loop.sessions.get_or_create("cli:direct"))

    assert result is not None
    assert result.content.startswith("计划已生成")
    assert plan is not None
    assert plan.plan == VALID_PLAN
    assert len(provider.calls) == 2
    assert "Non-goals / Out of Scope" in provider.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_planning_loop_uses_read_only_tools_and_project_context(tmp_path):
    provider = _Provider(VALID_PLAN)
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)

    session = loop.sessions.get_or_create("cli:direct")
    session.metadata["active_project"] = "demo"
    session.metadata["active_project_path"] = "/tmp/demo"
    (tmp_path / "memory" / "projects").mkdir(parents=True)
    (tmp_path / "memory" / "projects" / "demo.md").write_text("demo topology", encoding="utf-8")

    await loop._run_planning_loop(
        request="fix tests",
        history=[],
        channel="cli",
        chat_id="direct",
        execution_mode="executable",
        session=session,
    )

    tool_names = {
        definition["function"]["name"]
        for definition in provider.calls[0]["tools"]
    }
    assert {"read_file", "list_dir", "glob", "grep"}.issubset(tool_names)
    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names
    assert "exec" not in tool_names
    assert "spawn" not in tool_names
    assert "message" not in tool_names
    assert 'path="/tmp/demo"' in provider.calls[0]["messages"][0]["content"]
    assert "demo topology" in provider.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_plan_approve_executes_even_with_legacy_research_task_class(tmp_path):
    provider = _Provider("implemented")
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    session.metadata["active_project"] = "demo"
    session.metadata["active_project_path"] = "/tmp/demo"
    (tmp_path / "memory" / "projects").mkdir(parents=True)
    (tmp_path / "memory" / "projects" / "demo.md").write_text("demo topology", encoding="utf-8")
    plan = _saved_plan(session)
    plan.mark_approved("/plan-approve")
    plan.save_to_session(session)
    loop.sessions.save(session)

    result = await loop._execute_approved_plan(msg=_msg("/plan-approve"), session=session, plan=plan)

    assert isinstance(result, OutboundMessage)
    assert result.content == "implemented"
    assert not active_plan_path().exists()
    assert plan_history_path(plan.id).exists()
    system_prompt = provider.calls[0]["messages"][0]["content"]
    assert 'path="/tmp/demo"' in system_prompt
    assert "demo topology" in system_prompt
    prompt = provider.calls[0]["messages"][-1]["content"]
    assert f"已批准计划 `{plan.id}`" in prompt
    assert plan.content_hash in prompt
    assert "# 执行契约" in prompt
    assert "# 上下文优先级" in prompt
    assert "# 完整批准计划参考" in prompt
    assert "以下完整计划仅供参考" in prompt
    assert "Acceptance Criteria" in prompt
    assert "Non-goals / Out of Scope" in prompt

def test_build_execution_contract_extracts_constraints_and_scope():
    plan = PlanState.create("upgrade shop", LARGE_PLAN)
    plan.mark_approved("/plan-approve phase 3")
    scope = {
        "kind": "phases",
        "label": "阶段 3",
        "checklist": [item for item in plan.checklist if "阶段 3" in item["description"]],
    }

    contract = build_execution_contract(plan, scope)

    assert contract["plan_id"] == plan.id
    assert contract["content_hash"] == plan.content_hash
    assert contract["current_scope"] == "阶段 3"
    assert "Implement a staged shop upgrade" in contract["summary"]
    assert any("Do not implement real payment" in item for item in contract["non_negotiable_constraints"])
    assert any("阶段 3" in item["description"] for item in contract["checklist"])
    assert all("阶段 4" not in item["description"] for item in contract["checklist"])
    assert "执行契约和当前范围优先" in contract["conflict_rule"]


@pytest.mark.asyncio
async def test_approved_execution_prompt_is_contract_first_and_history_free(tmp_path):
    provider = _Provider("implemented")
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    session.add_message("user", "old user request")
    session.add_message("assistant", "old assistant plan")
    plan = PlanState.create("upgrade shop", LARGE_PLAN)
    plan.save_to_session(session)
    plan.mark_approved("/plan-approve phase 3")
    scope = {"kind": "phases", "label": "阶段 3", "checklist": [item for item in plan.checklist if "阶段 3" in item["description"]]}

    await loop._execute_approved_plan(msg=_msg("/plan-approve phase 3"), session=session, plan=plan, execution_scope=scope)

    messages = provider.calls[0]["messages"]
    prompt = messages[-1]["content"]
    assert prompt.index("# 执行契约") < prompt.index("# 上下文优先级") < prompt.index("# 当前执行范围") < prompt.index("# 完整批准计划参考")
    assert "以下完整计划仅供参考" in prompt
    assert "最终回复必须包含一段简短的执行契约符合性检查" in prompt
    assert "阶段 4：支付成功扣库存" not in prompt.split("# 验收检查", 1)[0]
    assert all("old user request" not in str(message.get("content")) for message in messages)
    assert all("old assistant plan" not in str(message.get("content")) for message in messages)


def test_overly_granular_plan_fails_quality():
    granular = LARGE_PLAN.replace(
        "- [ ] 阶段 6：新增后台页面。",
        "\n".join(f"- [ ] 阶段 {idx // 7 + 1}：细节步骤 {idx}" for idx in range(1, 36)),
    )

    quality = validate_plan_quality(
        granular,
        task_class="code_change_small",
        tools_used=[],
        min_exploration_steps=0,
    )

    assert not quality.passed
    assert any("too detailed" in failure for failure in quality.failures)
    assert any("3-5 stages" in instruction for instruction in quality.rewrite_instructions)


@pytest.mark.asyncio
async def test_plan_revise_outputs_delta_not_full_plan(tmp_path):
    revised_text = LARGE_PLAN.replace("Admin token default needs documentation.", "Admin token default is dev-admin-token.")
    revised_text = re.sub(r"^- .+6.+$", "", revised_text, flags=re.MULTILINE)
    provider = _Provider(revised_text)
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, planning_config=PlanningConfig(mode="auto", min_exploration_steps=0))
    session = loop.sessions.get_or_create("cli:direct")
    plan = PlanState.create("upgrade shop", LARGE_PLAN)
    plan.save_to_session(session)

    ctx = _ctx(loop, "/plan-revise 后台管理加 admin token")
    ctx.args = "后台管理加 admin token"
    result = await cmd_plan_revise(ctx)

    assert "## 变更摘要" in result.content
    assert "后台管理加 admin token" in result.content
    assert "使用 `/plan-status --full`" in result.content
    assert "## Exploration Evidence" not in result.content


@pytest.mark.asyncio
async def test_plan_approve_large_plan_prompts_for_scope(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider("implemented"), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = PlanState.create("upgrade shop", LARGE_PLAN)
    plan.save_to_session(session)

    result = await cmd_plan_approve(_ctx(loop, "/plan-approve"))

    assert result is not None
    assert "检测到大型计划" in result.content
    assert "/plan-approve all" in result.content


@pytest.mark.asyncio
async def test_plan_approve_phase_passes_scoped_checklist(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider("implemented"), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = PlanState.create("upgrade shop", LARGE_PLAN)
    plan.save_to_session(session)
    captured = {}

    async def fake_execute(**kwargs):
        captured["scope"] = kwargs.get("execution_scope")
        return OutboundMessage(channel="cli", chat_id="direct", content="implemented", metadata={"render_as": "text"})

    loop._execute_approved_plan = fake_execute

    result = await cmd_plan_approve(_ctx(loop, "/plan-approve phase 1"))

    assert result is not None
    assert result.content == "implemented"
    assert captured["scope"]["label"] == "阶段 1"
    assert all("阶段 1" in item["description"] for item in captured["scope"]["checklist"])


@pytest.mark.asyncio
async def test_plan_load_from_history_and_invalid_path(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider(), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = PlanState.create("load me", VALID_PLAN)
    plan.archive(workspace=tmp_path, status="cancelled")

    load_ctx = _ctx(loop, f"/plan-load {plan.id}")
    load_ctx.args = plan.id
    loaded = await cmd_plan_load(load_ctx)
    active = PlanState.from_session(session)
    bad_ctx = _ctx(loop, "/plan-load missing-plan")
    bad_ctx.args = "missing-plan"
    bad = await cmd_plan_load(bad_ctx)

    assert "已将计划加载到当前会话" in loaded.content
    assert active is not None
    assert active.id == plan.id
    assert active.status == "pending"
    assert "无法加载计划" in bad.content


@pytest.mark.asyncio
async def test_plan_approve_freezes_hash_and_executes(tmp_path):
    provider = _Provider("implemented")
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = _saved_plan(session)
    loop.sessions.save(session)

    result = await cmd_plan_approve(_ctx(loop, "/plan-approve"))

    assert isinstance(result, OutboundMessage)
    assert result.content == "implemented"
    assert plan.execution_mode == "executable"
    prompt = provider.calls[0]["messages"][-1]["content"]
    assert "执行模式：executable" in prompt
    assert plan_content_hash(VALID_PLAN) in prompt
    assert "减少工具往返" in prompt
    assert "写入后不要立刻重读" in prompt


@pytest.mark.asyncio
async def test_approved_plan_uses_configured_context_block_limit(tmp_path):
    provider = _Provider("implemented")
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        approved_plan_context_block_limit=12345,
    )
    session = loop.sessions.get_or_create("cli:direct")
    plan = _saved_plan(session)
    plan.mark_approved("yes")
    captured = {}

    async def fake_run_agent_loop(*args, **kwargs):
        captured["context_block_limit"] = kwargs.get("context_block_limit")
        return "implemented", [], kwargs.get("initial_messages", []) if "initial_messages" in kwargs else [], "completed", False, None

    loop._run_agent_loop = fake_run_agent_loop

    result = await loop._execute_approved_plan(msg=_msg("yes"), session=session, plan=plan)

    assert isinstance(result, OutboundMessage)
    assert result.content == "implemented"
    assert captured["context_block_limit"] == 12345


@pytest.mark.asyncio
async def test_stale_hash_refuses_execution(tmp_path):
    provider = _Provider("implemented")
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = _saved_plan(session)
    plan.mark_approved("yes")
    plan.plan += "\nchanged"
    plan.content_hash = plan_content_hash(plan.plan)

    result = await loop._execute_approved_plan(msg=_msg("yes"), session=session, plan=plan)

    assert result is not None
    assert "计划未批准或批准后内容已变化" in result.content
    assert provider.calls == []


@pytest.mark.asyncio
async def test_yes_interceptor_approves_and_no_interceptor_cancels(tmp_path):
    provider = _Provider("implemented")
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    _saved_plan(session)

    approved = await plan_approval_interceptor(_ctx(loop, "yes"))
    assert approved is not None
    assert approved.content == "implemented"
    assert not active_plan_path().exists()

    _saved_plan(session)
    cancelled = await plan_approval_interceptor(_ctx(loop, "no"))
    assert cancelled is not None
    assert "已取消计划" in cancelled.content
    assert PlanState.from_session(session) is None


@pytest.mark.asyncio
async def test_read_only_plan_approval_refuses_execution(tmp_path):
    provider = _Provider("implemented")
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = _saved_plan(session, execution_mode="read_only")
    plan.mark_approved("yes")

    result = await loop._execute_approved_plan(msg=_msg("yes"), session=session, plan=plan)

    assert result is not None
    assert "该计划是只读/不执行模式" in result.content
    assert provider.calls == []


@pytest.mark.asyncio
async def test_old_serialized_plan_defaults_to_executable(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider(), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = PlanState.create("fix tests", VALID_PLAN, task_class="research_only")
    data = plan.to_dict()
    data.pop("execution_mode")
    data["plan"] = data["plan"].replace("## Execution Mode\nexecutable.\n\n", "")
    session.metadata["pending_plan"] = data

    loaded = PlanState.from_session(session)

    assert loaded is not None
    assert loaded.execution_mode == "executable"


@pytest.mark.asyncio
async def test_explicit_read_only_request_creates_read_only_plan(tmp_path):
    provider = _Provider(VALID_PLAN.replace("executable.", "read_only."))
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        planning_config=PlanningConfig(min_exploration_steps=0),
    )

    result = await _plan(loop, "/plan --read-only analyze the project")
    plan = PlanState.from_session(loop.sessions.get_or_create("cli:direct"))

    assert result.content.startswith("计划已生成")
    assert plan is not None
    assert plan.execution_mode == "read_only"
    assert "执行模式：read_only" in result.content


@pytest.mark.asyncio
async def test_plan_revise_replaces_plan_and_records_history(tmp_path):
    provider = _Provider(VALID_PLAN)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        planning_config=PlanningConfig(min_exploration_steps=0),
    )
    session = loop.sessions.get_or_create("cli:direct")
    plan = PlanState.create("fix tests", VALID_PLAN.replace("Update", "Change"), task_class="research_only")
    plan.save_to_session(session)
    loop.sessions.save(session)
    ctx = _ctx(loop, "/plan-revise add verification")
    ctx.args = "add verification"

    result = await cmd_plan_revise(ctx)
    revised = PlanState.from_session(session)

    assert result.content.startswith("计划已更新")
    assert revised is not None
    assert revised.plan == VALID_PLAN
    assert revised.revision_count == 1
    assert revised.history[-1]["status"] == "superseded"


@pytest.mark.asyncio
async def test_plan_history_and_checklist_commands_render(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider(), workspace=tmp_path)
    session = loop.sessions.get_or_create("cli:direct")
    plan = _saved_plan(session)
    plan.add_history_entry("superseded")
    plan.save_to_session(session)

    history = await cmd_plan_history(_ctx(loop, "/plan-history"))
    checklist = await cmd_plan_checklist(_ctx(loop, "/plan-checklist"))

    assert "计划历史" in history.content
    assert "检查清单" in checklist.content
    assert "step-1" in checklist.content


@pytest.mark.asyncio
async def test_use_restores_active_project_path_from_topology(tmp_path):
    loop = AgentLoop(bus=MessageBus(), provider=_Provider(), workspace=tmp_path)
    projects_dir = tmp_path / "memory" / "projects"
    projects_dir.mkdir(parents=True)
    (projects_dir / "demo.md").write_text(
        "# Project Topology: demo\n> Project Path: D:/demo\n> Mode: Aider-style Repo Map\n",
        encoding="utf-8",
    )
    ctx = _ctx(loop, "/use demo")
    ctx.args = "demo"

    result = await cmd_use(ctx)
    session = loop.sessions.get_or_create("cli:direct")

    assert result is not None
    assert "D:/demo" in result.content
    assert session.metadata["active_project"] == "demo"
    assert session.metadata["active_project_path"] == "D:/demo"


@pytest.mark.asyncio
async def test_simple_read_only_request_bypasses_auto_planning(tmp_path):
    provider = _Provider("answer")
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        planning_config=PlanningConfig(mode="auto"),
    )

    result = await loop._process_message(_msg("explain this project"))

    assert result is not None
    assert result.content == "answer"
    assert PlanState.from_session(loop.sessions.get_or_create("cli:direct")) is None

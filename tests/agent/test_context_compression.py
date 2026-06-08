from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghostbot.agent.autocompact import AutoCompact, CheckpointSummary
from ghostbot.agent.context import ContextBuilder
from ghostbot.agent.memory import Consolidator, HybridSearchEngine
from ghostbot.agent.tools.search_memory import SearchMemoryTool
from ghostbot.agent.tools.search import GlobTool, GrepTool
from ghostbot.session.manager import Session, SessionManager


class _DummyProvider:
    generation = SimpleNamespace(max_tokens=4096)

    def get_default_model(self):
        return "test-model"


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "AGENTS.md").write_text("agent rules", encoding="utf-8")
    return tmp_path


def test_context_builder_separates_checkpoint_and_retrieved_memory(workspace):
    builder = ContextBuilder(workspace)
    session = Session(key="cli:test")
    session.touch_file("ghostbot/agent/runner.py")
    session.metadata["last_action"] = "tool_error"

    checkpoint_summary = (
        "Current goal:\n"
        "- fix runner\n\n"
        "Confirmed constraints:\n"
        "- keep exec path unchanged\n\n"
        "Open work / next steps:\n"
        "- inspect tests"
    )
    retrieved_memory = (
        '<record index="1" time_cursor="42">prior finding</record>\n'
        '<record index="2" time_cursor="43">another clue</record>'
    )

    messages = builder.build_messages(
        history=[],
        current_message="fix it",
        checkpoint_summary=checkpoint_summary,
        retrieved_memory=retrieved_memory,
        session_metadata=session.metadata,
        session=session,
        channel="cli",
        chat_id="test",
    )

    runtime = messages[-1]["content"]
    assert "[Working Memory]" in runtime
    assert "[Current goal]" in runtime
    assert "fix runner" in runtime
    assert "keep exec path unchanged" in runtime
    assert "[Retrieved Memory Cards]" in runtime
    assert "[Memory card 1]" in runtime
    assert "prior finding" in runtime

    system_prompt = messages[0]["content"]
    assert "# Working Set" in system_prompt
    assert "ghostbot/agent/runner.py" in system_prompt
    assert "Last action: tool_error" in system_prompt


def test_context_builder_merges_state_into_working_memory(workspace):
    builder = ContextBuilder(workspace)
    session = Session(key="cli:test")
    session.touch_file("ghostbot/agent/loop.py")
    session.metadata.update({
        "active_project": "ghostbot",
        "active_project_path": "C:/workspace/ghostbot",
        "last_action": "edited context builder",
        "pending_plan": {
            "status": "approved",
            "original_request": "finish layered memory rollout",
            "checklist": [
                {"description": "wire project state into working memory", "status": "pending"},
                {"description": "run focused tests", "status": "pending"},
            ],
        },
    })

    messages = builder.build_messages(
        history=[],
        current_message="continue",
        checkpoint_summary="Current goal:\n- refine memory cards",
        session_metadata=session.metadata,
        session=session,
        channel="cli",
        chat_id="test",
    )

    runtime = messages[-1]["content"]
    assert "Active project: ghostbot" in runtime
    assert "Planned request: finish layered memory rollout" in runtime
    assert "Plan status: approved" in runtime
    assert "wire project state into working memory" in runtime
    assert "Project path: C:/workspace/ghostbot" in runtime
    assert "ghostbot/agent/loop.py" in runtime
    assert "edited context builder" in runtime
    assert "refine memory cards" in runtime


def test_context_builder_shows_priority_bucket(workspace):
    builder = ContextBuilder(workspace)
    session = Session(key="cli:test")
    session.touch_file("ghostbot/agent/context.py")

    prompt = builder.build_system_prompt(
        session=session,
        session_metadata={},
        priority_bucket={"ghostbot/agent/loop.py": 9.0, "ghostbot/agent/context.py": 8.0},
    )

    assert "Priority bucket:" in prompt
    assert "ghostbot/agent/loop.py" in prompt


def test_context_builder_shows_graph_neighbors(workspace):
    builder = ContextBuilder(workspace)
    session = Session(key="cli:test")
    session.touch_file("ghostbot/agent/context.py")

    prompt = builder.build_system_prompt(
        session=session,
        session_metadata={},
        graph_neighbors={
            "ghostbot/agent/loop.py": ["referenced by ghostbot/agent/context.py"],
            "ghostbot/agent/memory.py": ["references ghostbot/agent/context.py"],
        },
    )

    assert "Graph neighbors:" in prompt
    assert "ghostbot/agent/loop.py" in prompt
    assert "referenced by ghostbot/agent/context.py" in prompt


def test_context_builder_shows_impacted_files_and_related_tests(workspace):
    builder = ContextBuilder(workspace)
    session = Session(key="cli:test")
    session.touch_file("ghostbot/agent/context.py")

    prompt = builder.build_system_prompt(
        session=session,
        session_metadata={},
        impacted_files={
            "ghostbot/agent/loop.py": ["calls helper in ghostbot/agent/context.py"],
        },
        related_tests={
            "tests/agent/test_context_compression.py": ["name matches ghostbot/agent/context.py"],
        },
    )

    assert "Impacted files:" in prompt
    assert "ghostbot/agent/loop.py" in prompt
    assert "Likely related tests:" in prompt
    assert "tests/agent/test_context_compression.py" in prompt


def test_context_builder_prompt_is_deterministic(workspace):
    builder = ContextBuilder(workspace)
    session = Session(key="cli:test")
    session.metadata["active_project"] = "demo"
    session.touch_file("ghostbot/agent/context.py")

    first = builder.build_system_prompt(session=session, session_metadata=session.metadata)
    second = builder.build_system_prompt(session=session, session_metadata=session.metadata)

    assert first == second


@pytest.mark.asyncio
async def test_grep_content_prioritizes_bucket(tmp_path):
    (tmp_path / "a.py").write_text("needle here\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("needle here\n", encoding="utf-8")

    bucket = {"b.py": 10.0, "a.py": 1.0}
    grep_tool = GrepTool(workspace=tmp_path, priority_bucket_provider=lambda: bucket)

    result = await grep_tool.execute(pattern="needle", path=".", output_mode="content")

    assert result.splitlines()[0].startswith("b.py:")


@pytest.mark.asyncio
async def test_auto_compact_returns_structured_checkpoint_summary(tmp_path):
    sessions = SessionManager(tmp_path)
    provider = _DummyProvider()
    builder = ContextBuilder(tmp_path)
    consolidator = Consolidator(
        store=builder.memory,
        provider=provider,
        model="test-model",
        sessions=sessions,
        context_window_tokens=8000,
        build_messages=builder.build_messages,
        get_tool_definitions=lambda: [],
        max_completion_tokens=1024,
    )
    auto = AutoCompact(sessions=sessions, consolidator=consolidator, session_ttl_minutes=1)

    session = sessions.get_or_create("cli:test")
    session.add_message("user", "need fix")
    session.add_message("assistant", "checking")
    session.updated_at = datetime.now()
    sessions.save(session)

    auto._summaries["cli:test"] = CheckpointSummary(
        text="Current goal: fix\nNext: run tests",
        last_active=datetime.now(),
    )

    reloaded, summary = auto.prepare_session(session, "cli:test")

    assert reloaded is session
    assert isinstance(summary, CheckpointSummary)
    assert summary.text.startswith("Current goal")
    assert "Previous conversation summary" in auto._format_summary(summary)


@pytest.mark.asyncio
async def test_auto_compact_fuse_trips_after_repeated_failures(tmp_path):
    sessions = SessionManager(tmp_path)
    provider = _DummyProvider()
    builder = ContextBuilder(tmp_path)
    consolidator = Consolidator(
        store=builder.memory,
        provider=provider,
        model="test-model",
        sessions=sessions,
        context_window_tokens=8000,
        build_messages=builder.build_messages,
        get_tool_definitions=lambda: [],
        max_completion_tokens=1024,
    )
    auto = AutoCompact(sessions=sessions, consolidator=consolidator, session_ttl_minutes=1)

    session = sessions.get_or_create("cli:test")
    for i in range(12):
        session.add_message("user" if i % 2 == 0 else "assistant", f"m{i}")
    session.updated_at = datetime.now()
    sessions.save(session)

    auto.consolidator.archive = AsyncMock(side_effect=RuntimeError("boom"))

    for _ in range(auto._MAX_CONSECUTIVE_FAILURES):
        await auto._archive("cli:test")

    assert auto._failure_counts["cli:test"] == auto._MAX_CONSECUTIVE_FAILURES

    scheduled = []
    auto.check_expired(lambda coro: scheduled.append(coro))
    assert scheduled == []


def test_consolidator_probe_keeps_new_build_messages_signature(tmp_path):
    sessions = SessionManager(tmp_path)
    provider = _DummyProvider()
    builder = ContextBuilder(tmp_path)
    consolidator = Consolidator(
        store=builder.memory,
        provider=provider,
        model="test-model",
        sessions=sessions,
        context_window_tokens=8000,
        build_messages=builder.build_messages,
        get_tool_definitions=lambda: [],
        max_completion_tokens=1024,
    )
    session = Session(key="cli:test")
    session.add_message("user", "hello")

    tokens, source = consolidator.estimate_session_prompt_tokens(session)

    assert isinstance(tokens, int)
    assert source in {"tiktoken", "none", "provider_counter"}


def test_consolidator_normalizes_unstructured_summary(tmp_path):
    builder = ContextBuilder(tmp_path)
    provider = MagicMock()
    consolidator = Consolidator(
        store=builder.memory,
        provider=provider,
        model="test-model",
        sessions=SessionManager(tmp_path),
        context_window_tokens=8000,
        build_messages=builder.build_messages,
        get_tool_definitions=lambda: [],
        max_completion_tokens=1024,
    )

    normalized = consolidator._normalize_checkpoint_summary(
        "Fix runner context overflow\n"
        "Keep Dream input path unchanged\n"
        "Touched ghostbot/agent/context.py\n"
        "Next: run regression tests"
    )

    assert normalized.startswith("Current goal:")
    assert "Confirmed constraints:\n- Keep Dream input path unchanged" in normalized
    assert "Key files and symbols:\n- Touched ghostbot/agent/context.py" in normalized
    assert "Open work / next steps:\n- Next: run regression tests" in normalized


@pytest.mark.asyncio
async def test_consolidator_archive_strips_thinking_and_persists_normalized_summary(tmp_path):
    sessions = SessionManager(tmp_path)
    builder = ContextBuilder(tmp_path)
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=SimpleNamespace(
        content=(
            "<think>draft</think>\n"
            "Current goal:\n"
            "- Fix compression flow\n\n"
            "Open work / next steps:\n"
            "- Run tests"
        )
    ))
    consolidator = Consolidator(
        store=builder.memory,
        provider=provider,
        model="test-model",
        sessions=sessions,
        context_window_tokens=8000,
        build_messages=builder.build_messages,
        get_tool_definitions=lambda: [],
        max_completion_tokens=1024,
    )

    summary = await consolidator.archive([
        {"role": "user", "content": "fix compression"},
        {"role": "assistant", "content": "checking"},
    ])

    assert summary is not None
    assert "<think>" not in summary
    assert summary.startswith("Current goal:")
    entries = builder.memory.read_unprocessed_history(since_cursor=0)
    assert len(entries) == 1
    assert entries[0]["content"] == summary




@pytest.mark.asyncio
async def test_search_tools_can_prioritize_graph_boosted_bucket(tmp_path):
    (tmp_path / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (tmp_path / "service.py").write_text("def call_helper():\n    return 2\n", encoding="utf-8")

    bucket = {"service.py": 25.0, "app.py": 10.0}
    glob_tool = GlobTool(workspace=tmp_path, priority_bucket_provider=lambda: bucket)

    result = await glob_tool.execute(pattern="*.py", path=".")

    assert result.splitlines()[0] == "service.py"


@pytest.mark.asyncio
async def test_search_tools_prioritize_bucket(tmp_path):
    (tmp_path / "a.py").write_text("needle here\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("needle here\n", encoding="utf-8")

    bucket = {"b.py": 10.0, "a.py": 1.0}
    glob_tool = GlobTool(workspace=tmp_path, priority_bucket_provider=lambda: bucket)
    grep_tool = GrepTool(workspace=tmp_path, priority_bucket_provider=lambda: bucket)

    glob_result = await glob_tool.execute(pattern="*.py", path=".")
    grep_result = await grep_tool.execute(pattern="needle", path=".", output_mode="files_with_matches")

    assert glob_result.splitlines()[0] == "b.py"
    assert grep_result.splitlines()[0] == "b.py"




def test_hybrid_search_engine_returns_typed_memory_records(workspace):
    engine = HybridSearchEngine(workspace)
    engine.index(1, "Must preserve the exec path during pytest runs")
    engine.index(2, "Project eval system uses structured summaries for long tasks")

    records = engine.search_records("exec path pytest", top_k=2)

    assert records
    assert records[0].summary
    assert records[0].record_type in {"instruction", "preference", "decision", "project_fact", "episode"}
    assert records[0].scope in {"global", "project", "branch", "task-cluster"}


@pytest.mark.asyncio
async def test_search_memory_tool_returns_memory_cards(workspace):
    engine = HybridSearchEngine(workspace)
    engine.index(1, "Must preserve the exec path during pytest runs")
    tool = SearchMemoryTool(engine)

    result = await tool.execute("exec path pytest")

    assert "<memory_card" in result
    assert 'type="instruction"' in result
    assert "Summary:" in result
    assert "preserve the exec path" in result

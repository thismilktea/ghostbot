from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghostbot.agent.autocompact import AutoCompact, CheckpointSummary
from ghostbot.agent.context import ContextBuilder
from ghostbot.agent.memory import Consolidator
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

    messages = builder.build_messages(
        history=[],
        current_message="fix it",
        checkpoint_summary="goal: fix runner\nnext: inspect tests",
        retrieved_memory="cursor=42\ncontent: prior finding",
        session_metadata=session.metadata,
        session=session,
        channel="cli",
        chat_id="test",
    )

    runtime = messages[-1]["content"]
    assert "[Checkpoint Summary]" in runtime
    assert "goal: fix runner" in runtime
    assert "[Retrieved Memory]" in runtime
    assert "prior finding" in runtime

    system_prompt = messages[0]["content"]
    assert "# Working Set" in system_prompt
    assert "ghostbot/agent/runner.py" in system_prompt
    assert "Last action: tool_error" in system_prompt


def test_context_builder_limits_recent_history_without_touching_dream_cursor(workspace):
    builder = ContextBuilder(workspace)
    for i in range(80):
        builder.memory.append_history(f"history line {i} " + ("x" * 120))

    prompt = builder.build_system_prompt()

    assert "# Recent History" in prompt
    assert "history line 79" in prompt
    assert len(prompt) < 20000


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


def test_context_builder_system_prompt_is_stable(workspace):
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


def test_session_manager_normalizes_active_files(tmp_path):
    manager = SessionManager(tmp_path)
    session = Session(key="cli:test")
    session.active_files = {".\\ghostbot\\agent\\loop.py": 1.0, "ghostbot/agent/loop.py": 2.0}

    manager.save(session)
    reloaded = manager.get_or_create("cli:test")

    assert reloaded.active_files == {"ghostbot/agent/loop.py": 2.0}

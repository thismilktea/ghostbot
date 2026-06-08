"""Tests for context snapshot, dashboard, and three-tier tool results."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ghostbot.agent.context import (
    CompressionEvent,
    ContextBucketSnapshot,
    ContextBuildSnapshot,
    ContextBuilder,
)
from ghostbot.utils.helpers import (
    ToolResultOutcome,
    build_status_content,
    classify_and_persist_tool_result,
    maybe_persist_tool_result,
)


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "AGENTS.md").write_text("agent rules", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Phase A: Context snapshot foundation
# ---------------------------------------------------------------------------


class TestContextBuildSnapshot:
    def test_build_messages_produces_snapshot_with_all_buckets(self, workspace):
        builder = ContextBuilder(workspace)
        builder.build_messages(
            history=[],
            current_message="hello",
            checkpoint_summary="Current goal:\n- fix runner",
            retrieved_memory='<memory_card index="1" cursor="42">prior finding</memory_card>',
            session_metadata={"active_project": "demo"},
            channel="cli",
            chat_id="test",
        )
        snap = builder.last_snapshot
        assert isinstance(snap, ContextBuildSnapshot)
        names = [b.name for b in snap.buckets]
        assert "working_memory" in names
        assert "retrieved_memory" in names
        assert "static_memory" in names
        assert "project_context" in names
        assert "working_set" in names
        assert "recent_history" in names

    def test_snapshot_records_budget_and_strategy(self, workspace):
        builder = ContextBuilder(workspace)
        builder.build_messages(
            history=[],
            current_message="test",
            session_metadata={},
            channel="cli",
            chat_id="test",
        )
        snap = builder.last_snapshot
        for bucket in snap.buckets:
            assert bucket.budget_chars > 0 or bucket.strategy == "empty"
            assert bucket.final_chars >= 0
            assert bucket.raw_chars >= 0

    def test_snapshot_to_dict_is_serializable(self, workspace):
        builder = ContextBuilder(workspace)
        builder.build_messages(
            history=[],
            current_message="test",
            session_metadata={"active_project": "demo"},
            channel="cli",
            chat_id="test",
        )
        data = builder.last_snapshot.to_dict()
        assert "buckets" in data
        assert "events" in data
        assert "total_final_chars" in data
        assert isinstance(data["buckets"], list)
        assert all(isinstance(b, dict) for b in data["buckets"])


class TestSnapshotProvenance:
    def test_recent_history_records_cut_boundary(self, workspace):
        builder = ContextBuilder(workspace)
        for i in range(5):
            builder.memory.append_history(f"entry {i}")
        builder.build_messages(
            history=[],
            current_message="test",
            session_metadata={},
            channel="cli",
            chat_id="test",
        )
        snap = builder.last_snapshot
        history_bucket = next(b for b in snap.buckets if b.name == "recent_history")
        assert history_bucket.item_count == 5
        assert history_bucket.provenance["total_candidates"] == 5
        assert history_bucket.provenance["cut_reason"] == "all_included"

    def test_retrieved_memory_records_card_cursors(self, workspace):
        builder = ContextBuilder(workspace)
        retrieved = (
            '<memory_card index="1" cursor="10">fact A</memory_card>\n'
            '<memory_card index="2" cursor="20">fact B</memory_card>\n'
            '<memory_card index="3" cursor="30">fact C</memory_card>\n'
            '<memory_card index="4" cursor="40">fact D</memory_card>'
        )
        builder.build_messages(
            history=[],
            current_message="test",
            retrieved_memory=retrieved,
            session_metadata={},
            channel="cli",
            chat_id="test",
        )
        snap = builder.last_snapshot
        mem_bucket = next(b for b in snap.buckets if b.name == "retrieved_memory")
        assert mem_bucket.provenance["total_candidate_cards"] == 4
        assert mem_bucket.provenance["included_cards"] == 3
        assert mem_bucket.dropped_items == 1
        assert "10" in mem_bucket.provenance["card_cursors"]

    def test_working_memory_provenance_includes_sources(self, workspace):
        builder = ContextBuilder(workspace)
        builder.build_messages(
            history=[],
            current_message="test",
            checkpoint_summary="Current goal:\n- refine memory",
            session_metadata={"active_project": "demo", "last_action": "edited file"},
            channel="cli",
            chat_id="test",
        )
        snap = builder.last_snapshot
        wm_bucket = next(b for b in snap.buckets if b.name == "working_memory")
        assert wm_bucket.provenance["has_checkpoint_summary"] is True
        assert wm_bucket.provenance["active_project"] == "demo"
        assert wm_bucket.provenance["last_action"] == "edited file"

    def test_project_context_provenance(self, workspace):
        (workspace / "memory" / "projects").mkdir(parents=True, exist_ok=True)
        (workspace / "memory" / "projects" / "myproj.md").write_text(
            "# MyProj\nA test project.", encoding="utf-8"
        )
        builder = ContextBuilder(workspace)
        builder.build_messages(
            history=[],
            current_message="test",
            session_metadata={"active_project": "myproj", "active_project_path": "/tmp/myproj"},
            channel="cli",
            chat_id="test",
        )
        snap = builder.last_snapshot
        proj_bucket = next(b for b in snap.buckets if b.name == "project_context")
        assert proj_bucket.provenance["active_project"] == "myproj"
        assert proj_bucket.provenance["project_file_exists"] is True
        assert proj_bucket.final_chars > 0


# ---------------------------------------------------------------------------
# Phase B: Context dashboard in /status
# ---------------------------------------------------------------------------


class TestContextDashboard:
    def test_status_includes_bucket_section_when_snapshot_present(self, workspace):
        builder = ContextBuilder(workspace)
        builder.build_messages(
            history=[],
            current_message="hello",
            session_metadata={"active_project": "demo"},
            channel="cli",
            chat_id="test",
        )
        import time

        status = build_status_content(
            version="0.1.0",
            model="test-model",
            start_time=time.time() - 60,
            last_usage={"prompt_tokens": 1000, "completion_tokens": 200},
            context_window_tokens=128000,
            session_msg_count=5,
            context_tokens_estimate=3000,
            context_snapshot=builder.last_snapshot,
        )
        assert "Context Buckets:" in status
        assert "working_memory:" in status
        assert "static_memory:" in status
        assert "Total:" in status

    def test_status_graceful_without_snapshot(self):
        import time

        status = build_status_content(
            version="0.1.0",
            model="test-model",
            start_time=time.time() - 60,
            last_usage={"prompt_tokens": 1000, "completion_tokens": 200},
            context_window_tokens=128000,
            session_msg_count=5,
            context_tokens_estimate=3000,
            context_snapshot=None,
        )
        assert "Context Buckets:" not in status

    def test_status_shows_compression_events(self, workspace):
        builder = ContextBuilder(workspace)
        retrieved = (
            '<memory_card index="1" cursor="10">fact A</memory_card>\n'
            '<memory_card index="2" cursor="20">fact B</memory_card>\n'
            '<memory_card index="3" cursor="30">fact C</memory_card>\n'
            '<memory_card index="4" cursor="40">fact D — this one gets dropped</memory_card>'
        )
        builder.build_messages(
            history=[],
            current_message="hello",
            retrieved_memory=retrieved,
            session_metadata={},
            channel="cli",
            chat_id="test",
        )
        import time

        status = build_status_content(
            version="0.1.0",
            model="test-model",
            start_time=time.time() - 60,
            last_usage={"prompt_tokens": 1000, "completion_tokens": 200},
            context_window_tokens=128000,
            session_msg_count=5,
            context_tokens_estimate=3000,
            context_snapshot=builder.last_snapshot,
        )
        assert "Recent compression:" in status


# ---------------------------------------------------------------------------
# Phase C: Three-tier tool results
# ---------------------------------------------------------------------------


class TestToolResultTiers:
    def test_small_result_stays_inline(self, tmp_path):
        outcome = classify_and_persist_tool_result(
            tmp_path, "sess", "tc1", "short output", max_chars=1000
        )
        assert outcome.tier == "small"
        assert outcome.strategy == "inline"
        assert outcome.display_content == "short output"
        assert outcome.persisted_path is None

    def test_medium_result_persists_with_preview(self, tmp_path):
        content = "x" * 5000
        outcome = classify_and_persist_tool_result(
            tmp_path, "sess", "tc2", content, max_chars=1000, medium_threshold=4000
        )
        assert outcome.tier == "medium"
        assert outcome.strategy == "summary_with_preview"
        assert outcome.persisted_path is not None
        assert Path(outcome.persisted_path).exists()
        assert outcome.original_chars == 5000
        assert "[tool output persisted]" in outcome.display_content

    def test_large_result_reference_only(self, tmp_path):
        content = "y" * 50000
        outcome = classify_and_persist_tool_result(
            tmp_path, "sess", "tc3", content, max_chars=1000, medium_threshold=4000
        )
        assert outcome.tier == "large"
        assert outcome.strategy == "reference_only"
        assert outcome.persisted_path is not None
        assert outcome.preview_chars == 200
        assert "large" in outcome.display_content

    def test_backward_compat_maybe_persist(self, tmp_path):
        small = maybe_persist_tool_result(tmp_path, "sess", "tc4", "tiny", max_chars=1000)
        assert small == "tiny"

        big = maybe_persist_tool_result(tmp_path, "sess", "tc5", "z" * 5000, max_chars=1000)
        assert isinstance(big, str)
        assert "[tool output persisted]" in big

    def test_outcome_to_dict(self, tmp_path):
        outcome = classify_and_persist_tool_result(
            tmp_path, "sess", "tc6", "a" * 5000, max_chars=1000, medium_threshold=4000
        )
        d = outcome.to_dict()
        assert d["tier"] == "medium"
        assert d["original_chars"] == 5000
        assert d["persisted_path"] is not None

    def test_none_workspace_returns_small(self):
        outcome = classify_and_persist_tool_result(
            None, "sess", "tc7", "anything", max_chars=1000
        )
        assert outcome.tier == "small"
        assert outcome.display_content == "anything"

    def test_list_content_persisted_as_json(self, tmp_path):
        content = [{"type": "text", "text": "a" * 5000}]
        outcome = classify_and_persist_tool_result(
            tmp_path, "sess", "tc8", content, max_chars=1000, medium_threshold=4000
        )
        assert outcome.tier == "medium"
        assert outcome.persisted_path is not None
        assert outcome.persisted_path.endswith(".json")


# ---------------------------------------------------------------------------
# Compression event dataclass
# ---------------------------------------------------------------------------


class TestCompressionEvent:
    def test_to_dict_omits_zero_and_none(self):
        ev = CompressionEvent(
            bucket="working_set",
            strategy="bounded_text",
            reason="budget exceeded",
            kept_chars=1200,
            dropped_chars=400,
        )
        d = ev.to_dict()
        assert d["bucket"] == "working_set"
        assert d["kept_chars"] == 1200
        assert d["dropped_chars"] == 400
        assert "note" not in d
        assert "kept_items" not in d

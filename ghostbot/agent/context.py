"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ghostbot.agent.memory import MemoryStore
from ghostbot.agent.skills import SkillsLoader
from ghostbot.utils.helpers import (
    build_assistant_message,
    current_time_str,
    detect_image_mime,
    truncate_text,
)
from ghostbot.utils.prompt_templates import render_template


@dataclass(slots=True)
class WorkingMemorySnapshot:
    goal: list[str]
    constraints: list[str]
    files_and_symbols: list[str]
    errors_and_fixes: list[str]
    next_steps: list[str]
    quoted_details: list[str]

    @classmethod
    def empty(cls) -> "WorkingMemorySnapshot":
        return cls([], [], [], [], [], [])

    def is_empty(self) -> bool:
        return not any((
            self.goal,
            self.constraints,
            self.files_and_symbols,
            self.errors_and_fixes,
            self.next_steps,
            self.quoted_details,
        ))

    def merge(self, other: "WorkingMemorySnapshot") -> "WorkingMemorySnapshot":
        return WorkingMemorySnapshot(
            goal=[*self.goal, *other.goal],
            constraints=[*self.constraints, *other.constraints],
            files_and_symbols=[*self.files_and_symbols, *other.files_and_symbols],
            errors_and_fixes=[*self.errors_and_fixes, *other.errors_and_fixes],
            next_steps=[*self.next_steps, *other.next_steps],
            quoted_details=[*self.quoted_details, *other.quoted_details],
        )


@dataclass(slots=True)
class CompressionEvent:
    bucket: str
    strategy: str
    reason: str
    kept_chars: int
    dropped_chars: int = 0
    kept_items: int = 0
    dropped_items: int = 0
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value not in (None, "", 0)}


@dataclass(slots=True)
class ContextBucketSnapshot:
    name: str
    section: str
    raw_chars: int = 0
    final_chars: int = 0
    budget_chars: int = 0
    item_count: int = 0
    strategy: str = "none"
    dropped_chars: int = 0
    dropped_items: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "section": self.section,
            "raw_chars": self.raw_chars,
            "final_chars": self.final_chars,
            "budget_chars": self.budget_chars,
            "item_count": self.item_count,
            "strategy": self.strategy,
            "dropped_chars": self.dropped_chars,
            "dropped_items": self.dropped_items,
            "provenance": self.provenance,
        }


@dataclass(slots=True)
class ContextBuildSnapshot:
    buckets: list[ContextBucketSnapshot] = field(default_factory=list)
    events: list[CompressionEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_bucket(self, bucket: ContextBucketSnapshot) -> None:
        self.buckets.append(bucket)

    def add_event(self, event: CompressionEvent) -> None:
        self.events.append(event)

    def total_final_chars(self) -> int:
        return sum(bucket.final_chars for bucket in self.buckets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "total_final_chars": self.total_final_chars(),
            "buckets": [bucket.to_dict() for bucket in self.buckets],
            "events": [event.to_dict() for event in self.events],
        }


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"
    _MAX_RECENT_HISTORY = 50
    _MAX_STATIC_MEMORY_CHARS = 4_000
    _MAX_PROJECT_CONTEXT_CHARS = 2_400
    _MAX_WORKING_SET_CHARS = 1_600
    _MAX_WORKING_SET_FILES = 8
    _MAX_PRIORITY_BUCKET_FILES = 6
    _MAX_RECENT_HISTORY_CHARS = 4_000
    _MAX_CHECKPOINT_CHARS = 2_200
    _MAX_RETRIEVED_MEMORY_CHARS = 1_800
    _OPTIONAL_CONTEXT_BUDGET_CHARS = 6_000
    _MAX_WORKING_MEMORY_SECTION_CHARS = 720
    _MAX_RETRIEVED_MEMORY_RECORDS = 3
    _MAX_RETRIEVED_MEMORY_RECORD_CHARS = 420
    _WORKING_MEMORY_SECTIONS = (
        ("Current goal", "goal"),
        ("Confirmed constraints", "constraints"),
        ("Key files and symbols", "files_and_symbols"),
        ("Important errors and fixes", "errors_and_fixes"),
        ("Open work / next steps", "next_steps"),
        ("Recent quoted details", "quoted_details"),
    )

    def __init__(self, workspace: Path, timezone: str | None = None, disabled_skills: list[str] | None = None):
        self.workspace = workspace
        self.timezone = timezone
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace, disabled_skills=set(disabled_skills) if disabled_skills else None)
        self._last_snapshot = ContextBuildSnapshot()

    @staticmethod
    def _bounded_text(text: str | None, max_chars: int) -> str:
        if not text:
            return ""
        return truncate_text(text.strip(), max_chars)

    @classmethod
    def _normalize_bullet_lines(cls, text: str) -> list[str]:
        lines: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ")):
                line = line[2:].strip()
            lines.append(line)
        return lines

    @classmethod
    def _dedupe_items(cls, items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    @classmethod
    def _extract_working_memory_section(cls, summary: str, header: str) -> list[str]:
        marker = f"{header}:"
        start = summary.find(marker)
        if start < 0:
            return []
        start += len(marker)
        end = len(summary)
        for other_header, _ in cls._WORKING_MEMORY_SECTIONS:
            if other_header == header:
                continue
            idx = summary.find(f"{other_header}:", start)
            if idx >= 0:
                end = min(end, idx)
        return cls._normalize_bullet_lines(summary[start:end])

    @classmethod
    def _parse_working_memory(cls, checkpoint_summary: str | None) -> WorkingMemorySnapshot:
        if not checkpoint_summary:
            return WorkingMemorySnapshot.empty()
        data = {
            attr: cls._extract_working_memory_section(checkpoint_summary, header)
            for header, attr in cls._WORKING_MEMORY_SECTIONS
        }
        return WorkingMemorySnapshot(**data)

    @classmethod
    def _snapshot_from_state(
        cls,
        session_metadata: dict[str, Any] | None,
        project: "ProjectState | None",
    ) -> WorkingMemorySnapshot:
        metadata = session_metadata or {}
        snapshot = WorkingMemorySnapshot.empty()

        active_project = metadata.get("active_project")
        active_project_path = metadata.get("active_project_path")
        if active_project:
            snapshot.goal.append(f"Active project: {active_project}")
        if active_project_path:
            snapshot.files_and_symbols.append(f"Project path: {active_project_path}")

        for key, label in (
            ("last_action", "Last action"),
            ("last_exec_status", "Last exec status"),
            ("last_test_status", "Last test status"),
        ):
            value = metadata.get(key)
            if value:
                snapshot.errors_and_fixes.append(f"{label}: {value}")

        pending_plan = metadata.get("pending_plan")
        if isinstance(pending_plan, dict):
            status = pending_plan.get("status")
            if status:
                snapshot.constraints.append(f"Plan status: {status}")
            original_request = str(pending_plan.get("original_request") or "").strip()
            if original_request:
                snapshot.goal.append(f"Planned request: {original_request}")
            checklist = pending_plan.get("checklist")
            if isinstance(checklist, list):
                pending_items = [
                    str(item.get("description") or "").strip()
                    for item in checklist
                    if isinstance(item, dict) and item.get("status") != "completed"
                ]
                snapshot.next_steps.extend(item for item in pending_items[:3] if item)

        if project and getattr(project, "active_files", None):
            active_files = sorted(project.active_files.items(), key=lambda item: item[1], reverse=True)
            snapshot.files_and_symbols.extend(path for path, _ in active_files[: cls._MAX_WORKING_SET_FILES])

        snapshot.goal = cls._dedupe_items(snapshot.goal)
        snapshot.constraints = cls._dedupe_items(snapshot.constraints)
        snapshot.files_and_symbols = cls._dedupe_items(snapshot.files_and_symbols)
        snapshot.errors_and_fixes = cls._dedupe_items(snapshot.errors_and_fixes)
        snapshot.next_steps = cls._dedupe_items(snapshot.next_steps)
        snapshot.quoted_details = cls._dedupe_items(snapshot.quoted_details)
        return snapshot

    @classmethod
    def _compose_working_memory(
        cls,
        checkpoint_summary: str | None,
        session_metadata: dict[str, Any] | None,
        project: "ProjectState | None",
    ) -> WorkingMemorySnapshot:
        parsed = cls._parse_working_memory(checkpoint_summary)
        state_snapshot = cls._snapshot_from_state(session_metadata, project)
        merged = state_snapshot.merge(parsed)
        merged.goal = cls._dedupe_items(merged.goal)
        merged.constraints = cls._dedupe_items(merged.constraints)
        merged.files_and_symbols = cls._dedupe_items(merged.files_and_symbols)
        merged.errors_and_fixes = cls._dedupe_items(merged.errors_and_fixes)
        merged.next_steps = cls._dedupe_items(merged.next_steps)
        merged.quoted_details = cls._dedupe_items(merged.quoted_details)
        return merged

    @staticmethod
    def _clip_provenance_value(value: Any, *, max_chars: int = 240, max_items: int = 8) -> Any:
        if isinstance(value, str):
            return truncate_text(value, max_chars)
        if isinstance(value, list):
            clipped = [ContextBuilder._clip_provenance_value(item, max_chars=max_chars, max_items=max_items) for item in value[:max_items]]
            if len(value) > max_items:
                clipped.append(f"... (+{len(value) - max_items} more)")
            return clipped
        if isinstance(value, dict):
            return {
                str(key): ContextBuilder._clip_provenance_value(item, max_chars=max_chars, max_items=max_items)
                for key, item in list(value.items())[:max_items]
            }
        return value

    @classmethod
    def _record_bucket(
        cls,
        snapshot: ContextBuildSnapshot | None,
        *,
        name: str,
        section: str,
        raw_text: str | None = None,
        final_text: str | None = None,
        budget_chars: int = 0,
        item_count: int = 0,
        strategy: str = "none",
        dropped_items: int = 0,
        provenance: dict[str, Any] | None = None,
        event_reason: str | None = None,
        event_note: str | None = None,
    ) -> None:
        if snapshot is None:
            return
        raw = raw_text or ""
        final = final_text or ""
        bucket = ContextBucketSnapshot(
            name=name,
            section=section,
            raw_chars=len(raw),
            final_chars=len(final),
            budget_chars=budget_chars,
            item_count=item_count,
            strategy=strategy,
            dropped_chars=max(0, len(raw) - len(final)),
            dropped_items=max(0, dropped_items),
            provenance=cls._clip_provenance_value(provenance or {}),
        )
        snapshot.add_bucket(bucket)
        if bucket.dropped_chars or bucket.dropped_items or strategy not in {"none", "full"}:
            snapshot.add_event(CompressionEvent(
                bucket=name,
                strategy=strategy,
                reason=event_reason or "bucket shaping",
                kept_chars=bucket.final_chars,
                dropped_chars=bucket.dropped_chars,
                kept_items=bucket.item_count,
                dropped_items=bucket.dropped_items,
                note=event_note,
            ))

    @staticmethod
    def _parse_retrieved_memory_records(retrieved_memory: str | None) -> list[dict[str, Any]]:
        if not retrieved_memory:
            return []

        records: list[dict[str, Any]] = []
        chunks = retrieved_memory.split("<memory_card ")
        tag = "memory_card"
        closing_tag = "</memory_card>"
        if len(chunks) == 1:
            chunks = retrieved_memory.split("<record ")
            tag = "record"
            closing_tag = "</record>"
        for chunk in chunks[1:]:
            body = chunk.split(">", 1)
            if len(body) != 2:
                continue
            attrs, rest = body
            content = rest.split(closing_tag, 1)[0].strip()
            if not content:
                continue
            metadata = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
            metadata["tag"] = tag
            metadata["content"] = content
            records.append(metadata)
        return records

    @classmethod
    def _working_memory_provenance(
        cls,
        checkpoint_summary: str | None,
        session_metadata: dict[str, Any] | None,
        project: "ProjectState | None",
        merged: WorkingMemorySnapshot,
    ) -> dict[str, Any]:
        metadata = session_metadata or {}
        project_files: list[str] = []
        if project and getattr(project, "active_files", None):
            active_files = sorted(project.active_files.items(), key=lambda item: item[1], reverse=True)
            project_files = [path for path, _ in active_files[: cls._MAX_WORKING_SET_FILES]]
        return {
            "has_checkpoint_summary": bool(checkpoint_summary),
            "active_project": metadata.get("active_project"),
            "active_project_path": metadata.get("active_project_path"),
            "last_action": metadata.get("last_action"),
            "pending_plan_status": ((metadata.get("pending_plan") or {}).get("status") if isinstance(metadata.get("pending_plan"), dict) else None),
            "active_files": project_files,
            "section_counts": {
                "goal": len(merged.goal),
                "constraints": len(merged.constraints),
                "files_and_symbols": len(merged.files_and_symbols),
                "errors_and_fixes": len(merged.errors_and_fixes),
                "next_steps": len(merged.next_steps),
                "quoted_details": len(merged.quoted_details),
            },
        }

    @property
    def last_snapshot(self) -> ContextBuildSnapshot:
        return self._last_snapshot

    @classmethod
    def _format_working_memory(cls, snapshot: WorkingMemorySnapshot) -> str:
        if snapshot.is_empty():
            return ""

        remaining = cls._MAX_CHECKPOINT_CHARS
        sections: list[str] = []
        for header, attr in cls._WORKING_MEMORY_SECTIONS:
            items = getattr(snapshot, attr)
            if not items or remaining <= 0:
                continue
            body = cls._bounded_text(
                "\n".join(f"- {item}" for item in items),
                min(cls._MAX_WORKING_MEMORY_SECTION_CHARS, remaining),
            )
            if not body:
                continue
            block = f"[{header}]\n{body}"
            if len(block) > remaining:
                block = cls._bounded_text(block, remaining)
            sections.append(block)
            remaining = max(0, remaining - len(block))
        return "\n\n".join(sections)

    @classmethod
    def _format_retrieved_memory_cards(cls, retrieved_memory: str | None) -> str:
        if not retrieved_memory:
            return ""

        records: list[str] = []
        chunks = retrieved_memory.split("<memory_card ")
        if len(chunks) == 1:
            chunks = retrieved_memory.split("<record ")
            closing_tag = "</record>"
        else:
            closing_tag = "</memory_card>"
        for chunk in chunks[1:]:
            body = chunk.split(">", 1)
            if len(body) != 2:
                continue
            content = body[1].split(closing_tag, 1)[0].strip()
            if not content:
                continue
            records.append(content)
            if len(records) >= cls._MAX_RETRIEVED_MEMORY_RECORDS:
                break

        if not records:
            return cls._bounded_text(retrieved_memory, cls._MAX_RETRIEVED_MEMORY_CHARS)

        formatted: list[str] = []
        remaining = cls._MAX_RETRIEVED_MEMORY_CHARS
        for idx, record in enumerate(records, start=1):
            if remaining <= 0:
                break
            body = cls._bounded_text(record, min(cls._MAX_RETRIEVED_MEMORY_RECORD_CHARS, remaining))
            block = f"[Memory card {idx}]\n{body}"
            if len(block) > remaining:
                block = cls._bounded_text(block, remaining)
            formatted.append(block)
            remaining = max(0, remaining - len(block))
        return "\n\n".join(formatted)

    def _build_projects_context(
        self,
        active_project_name: str | None,
        active_project_path: str | None = None,
        snapshot: ContextBuildSnapshot | None = None,
    ) -> str:
        """只读取当前 Project 活跃的项目架构。"""
        if not active_project_name:
            self._record_bucket(
                snapshot,
                name="project_context",
                section="system",
                budget_chars=self._MAX_PROJECT_CONTEXT_CHARS,
                strategy="empty",
                provenance={"active_project": None},
                event_reason="project context unavailable",
            )
            return ""

        project_file = self.workspace / "memory" / "projects" / f"{active_project_name}.md"
        raw_content = ""
        final_content = ""
        if project_file.exists():
            try:
                raw_content = project_file.read_text(encoding="utf-8")
                final_content = self._bounded_text(raw_content, self._MAX_PROJECT_CONTEXT_CHARS)
            except Exception:
                raw_content = ""
                final_content = ""

        lines = [f'<active_project name="{active_project_name}"']
        if active_project_path:
            lines[0] += f' path="{active_project_path}"'
        lines[0] += ">"
        if active_project_path:
            lines.append(f"Project path: {active_project_path}")
        if final_content:
            lines.append(final_content)
        lines.append("</active_project>")
        rendered = "\n".join(lines)
        self._record_bucket(
            snapshot,
            name="project_context",
            section="system",
            raw_text=raw_content,
            final_text=rendered,
            budget_chars=self._MAX_PROJECT_CONTEXT_CHARS,
            item_count=1,
            strategy="bounded_text" if raw_content and raw_content != final_content else ("full" if raw_content else "metadata_only"),
            provenance={
                "active_project": active_project_name,
                "active_project_path": active_project_path,
                "project_file": str(project_file),
                "project_file_exists": project_file.exists(),
            },
            event_reason="project context prepared",
        )
        return rendered

    def _build_working_set_context(
        self,
        project: "ProjectState | None",
        session_metadata: dict[str, Any] | None,
        priority_bucket: dict[str, float] | None = None,
        graph_neighbors: dict[str, list[str]] | None = None,
        impacted_files: dict[str, list[str]] | None = None,
        related_tests: dict[str, list[str]] | None = None,
        snapshot: ContextBuildSnapshot | None = None,
    ) -> str:
        lines: list[str] = []
        metadata = session_metadata or {}
        active_file_paths: list[str] = []
        if project and getattr(project, "active_files", None):
            active_files = sorted(project.active_files.items(), key=lambda item: item[1], reverse=True)
            active_file_paths = [path for path, _ in active_files[: self._MAX_WORKING_SET_FILES]]
            if active_file_paths:
                lines.append("Active files:")
                lines.extend(f"- {path}" for path in active_file_paths)

        priority_top: list[str] = []
        if priority_bucket:
            ranked = sorted(priority_bucket.items(), key=lambda item: (-item[1], item[0]))
            priority_top = [path for path, _ in ranked[: self._MAX_PRIORITY_BUCKET_FILES]]
            if priority_top:
                lines.append("Priority bucket:")
                lines.extend(f"- {path}" for path in priority_top)

        graph_top: list[str] = []
        if graph_neighbors:
            ranked_neighbors = sorted(
                graph_neighbors.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
            if ranked_neighbors:
                graph_top = [path for path, _ in ranked_neighbors[: self._MAX_PRIORITY_BUCKET_FILES]]
                lines.append("Graph neighbors:")
                for path, reasons in ranked_neighbors[: self._MAX_PRIORITY_BUCKET_FILES]:
                    reason = "; ".join(reasons[:2])
                    lines.append(f"- {path} — {reason}" if reason else f"- {path}")

        impacted_top: list[str] = []
        if impacted_files:
            ranked_impacted = sorted(
                impacted_files.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
            if ranked_impacted:
                impacted_top = [path for path, _ in ranked_impacted[: self._MAX_PRIORITY_BUCKET_FILES]]
                lines.append("Impacted files:")
                for path, reasons in ranked_impacted[: self._MAX_PRIORITY_BUCKET_FILES]:
                    reason = "; ".join(reasons[:2])
                    lines.append(f"- {path} — {reason}" if reason else f"- {path}")

        tests_top: list[str] = []
        if related_tests:
            ranked_tests = sorted(
                related_tests.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
            if ranked_tests:
                tests_top = [path for path, _ in ranked_tests[: self._MAX_PRIORITY_BUCKET_FILES]]
                lines.append("Likely related tests:")
                for path, reasons in ranked_tests[: self._MAX_PRIORITY_BUCKET_FILES]:
                    reason = "; ".join(reasons[:2])
                    lines.append(f"- {path} — {reason}" if reason else f"- {path}")

        last_action = metadata.get("last_action")
        if last_action:
            lines.append(f"Last action: {last_action}")
        last_exec_status = metadata.get("last_exec_status")
        if last_exec_status:
            lines.append(f"Last exec status: {last_exec_status}")
        last_test_status = metadata.get("last_test_status")
        if last_test_status:
            lines.append(f"Last test status: {last_test_status}")

        raw_text = "\n".join(lines)
        final = self._bounded_text(raw_text, self._MAX_WORKING_SET_CHARS)
        self._record_bucket(
            snapshot,
            name="working_set",
            section="system",
            raw_text=raw_text,
            final_text=final,
            budget_chars=self._MAX_WORKING_SET_CHARS,
            item_count=len(active_file_paths) + len(priority_top) + len(graph_top) + len(impacted_top) + len(tests_top),
            strategy="bounded_text" if len(raw_text) > self._MAX_WORKING_SET_CHARS else "full",
            provenance={
                "active_files": active_file_paths,
                "priority_bucket_top": priority_top,
                "graph_neighbors_top": graph_top,
                "impacted_files_top": impacted_top,
                "related_tests_top": tests_top,
            },
            event_reason="working set assembled",
        )
        return final

    def _build_recent_history_context(self, snapshot: ContextBuildSnapshot | None = None) -> str:
        entries = self.memory.read_unprocessed_history(since_cursor=self.memory.get_last_dream_cursor())
        total_candidates = len(entries)
        if not entries:
            self._record_bucket(
                snapshot,
                name="recent_history",
                section="system",
                budget_chars=self._MAX_RECENT_HISTORY_CHARS,
                strategy="empty",
                provenance={"total_candidates": 0},
                event_reason="no recent history entries",
            )
            return ""

        collected: list[str] = []
        used = 0
        oldest_ts: str | None = None
        newest_ts: str | None = None
        for entry in reversed(entries):
            line = f"- [{entry['timestamp']}] {entry['content']}"
            projected = used + len(line) + (1 if collected else 0)
            if collected and projected > self._MAX_RECENT_HISTORY_CHARS:
                break
            collected.append(line)
            used = projected
            if newest_ts is None:
                newest_ts = entry.get("timestamp")
            oldest_ts = entry.get("timestamp")
            if len(collected) >= self._MAX_RECENT_HISTORY:
                break
        collected.reverse()
        result = "\n".join(collected)
        cut_reason = "max_entries" if len(collected) >= self._MAX_RECENT_HISTORY else ("char_budget" if total_candidates > len(collected) else "all_included")
        self._record_bucket(
            snapshot,
            name="recent_history",
            section="system",
            raw_text=result,
            final_text=result,
            budget_chars=self._MAX_RECENT_HISTORY_CHARS,
            item_count=len(collected),
            strategy="suffix_cut" if total_candidates > len(collected) else "full",
            dropped_items=total_candidates - len(collected),
            provenance={
                "total_candidates": total_candidates,
                "included": len(collected),
                "oldest_included_ts": oldest_ts,
                "newest_included_ts": newest_ts,
                "cut_reason": cut_reason,
            },
            event_reason="recent history assembled",
        )
        return result

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        session_metadata: dict | None = None,
        channel: str | None = None,
        project: "ProjectState | None" = None,
        priority_bucket: dict[str, float] | None = None,
        graph_neighbors: dict[str, list[str]] | None = None,
        impacted_files: dict[str, list[str]] | None = None,
        related_tests: dict[str, list[str]] | None = None,
        session: "ProjectState | None" = None,
        _snapshot: ContextBuildSnapshot | None = None,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, skills, and working-set context."""
        project = project or session
        snap = _snapshot
        parts = [self._get_identity(channel=channel)]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        raw_memory = self.memory.get_memory_context()
        memory = self._bounded_text(raw_memory, self._MAX_STATIC_MEMORY_CHARS)
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        self._record_bucket(
            snap,
            name="static_memory",
            section="system",
            raw_text=raw_memory,
            final_text=memory,
            budget_chars=self._MAX_STATIC_MEMORY_CHARS,
            item_count=1 if memory else 0,
            strategy="bounded_text" if raw_memory and len(raw_memory) > self._MAX_STATIC_MEMORY_CHARS else "full",
            provenance={"source": "memory/MEMORY.md"},
            event_reason="static memory loaded",
        )

        metadata = session_metadata or {}
        active_project = metadata.get("active_project")
        active_project_path = metadata.get("active_project_path")
        projects_context = self._build_projects_context(active_project, active_project_path, snapshot=snap)
        if projects_context:
            parts.append(f"## 🏗️ Current Project Context\n\n{projects_context}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))

        remaining_optional_budget = self._OPTIONAL_CONTEXT_BUDGET_CHARS
        working_set = self._build_working_set_context(
            project,
            session_metadata,
            priority_bucket=priority_bucket,
            graph_neighbors=graph_neighbors,
            impacted_files=impacted_files,
            related_tests=related_tests,
            snapshot=snap,
        )
        if working_set and remaining_optional_budget > 0:
            bounded_working_set = self._bounded_text(
                working_set,
                min(self._MAX_WORKING_SET_CHARS, remaining_optional_budget),
            )
            if bounded_working_set:
                parts.append(f"# Working Set\n\n{bounded_working_set}")
                remaining_optional_budget = max(0, remaining_optional_budget - len(bounded_working_set))

        recent_history = self._build_recent_history_context(snapshot=snap)
        if recent_history and remaining_optional_budget > 256:
            bounded_history = self._bounded_text(
                recent_history,
                min(self._MAX_RECENT_HISTORY_CHARS, remaining_optional_budget),
            )
            if bounded_history:
                parts.append(f"# Recent History\n\n{bounded_history}")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self, channel: str | None = None) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )

    @classmethod
    def _build_runtime_context(
        cls,
        channel: str | None,
        chat_id: str | None,
        timezone: str | None = None,
        *,
        session_summary: str | None = None,
        checkpoint_summary: str | None = None,
        retrieved_memory: str | None = None,
        session_metadata: dict[str, Any] | None = None,
        project: "ProjectState | None" = None,
    ) -> str:
        """Build untrusted runtime metadata blocks for injection before the user message."""
        if session_summary and checkpoint_summary is None and retrieved_memory is None:
            checkpoint_summary = session_summary

        lines = [f"Current Time: {current_time_str(timezone)}"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]

        working_memory = cls._format_working_memory(
            cls._compose_working_memory(checkpoint_summary, session_metadata, project)
        )
        if working_memory:
            lines += ["", "[Working Memory]", working_memory]

        memory_cards = cls._format_retrieved_memory_cards(retrieved_memory)
        if memory_cards:
            lines += ["", "[Retrieved Memory Cards]", memory_cards]
        return cls._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n" + cls._RUNTIME_CONTEXT_END

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        session_summary: str | None = None,
        checkpoint_summary: str | None = None,
        retrieved_memory: str | None = None,
        session_metadata: dict | None = None,
        project: "ProjectState | None" = None,
        priority_bucket: dict[str, float] | None = None,
        graph_neighbors: dict[str, list[str]] | None = None,
        impacted_files: dict[str, list[str]] | None = None,
        related_tests: dict[str, list[str]] | None = None,
        session: "ProjectState | None" = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        project = project or session
        snap = ContextBuildSnapshot(metadata={"channel": channel, "chat_id": chat_id})

        # --- working memory bucket ---
        effective_checkpoint = checkpoint_summary
        if session_summary and checkpoint_summary is None and retrieved_memory is None:
            effective_checkpoint = session_summary
        wm_snapshot = self._compose_working_memory(effective_checkpoint, session_metadata, project)
        wm_text = self._format_working_memory(wm_snapshot)
        wm_provenance = self._working_memory_provenance(effective_checkpoint, session_metadata, project, wm_snapshot)
        self._record_bucket(
            snap,
            name="working_memory",
            section="runtime",
            raw_text=effective_checkpoint or "",
            final_text=wm_text,
            budget_chars=self._MAX_CHECKPOINT_CHARS,
            item_count=sum(len(getattr(wm_snapshot, attr)) for _, attr in self._WORKING_MEMORY_SECTIONS),
            strategy="structured_merge",
            provenance=wm_provenance,
            event_reason="working memory composed",
        )

        # --- retrieved memory bucket ---
        parsed_records = self._parse_retrieved_memory_records(retrieved_memory)
        memory_cards_text = self._format_retrieved_memory_cards(retrieved_memory)
        total_candidate_cards = len(parsed_records)
        included_cards = min(total_candidate_cards, self._MAX_RETRIEVED_MEMORY_RECORDS)
        card_cursors = [rec.get("cursor") or rec.get("time_cursor") for rec in parsed_records[:included_cards]]
        used_fallback = bool(retrieved_memory and not parsed_records)
        self._record_bucket(
            snap,
            name="retrieved_memory",
            section="runtime",
            raw_text=retrieved_memory or "",
            final_text=memory_cards_text,
            budget_chars=self._MAX_RETRIEVED_MEMORY_CHARS,
            item_count=included_cards,
            strategy="card_limit" if total_candidate_cards > included_cards else ("fallback_truncate" if used_fallback else "full"),
            dropped_items=max(0, total_candidate_cards - included_cards),
            provenance={
                "total_candidate_cards": total_candidate_cards,
                "included_cards": included_cards,
                "card_cursors": card_cursors,
                "used_fallback_truncation": used_fallback,
            },
            event_reason="retrieved memory formatted",
        )

        runtime_ctx = self._build_runtime_context(
            channel,
            chat_id,
            self.timezone,
            session_summary=session_summary,
            checkpoint_summary=checkpoint_summary,
            retrieved_memory=retrieved_memory,
            session_metadata=session_metadata,
            project=project,
        )

        user_content = self._build_user_content(current_message, media)
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    skill_names,
                    session_metadata=session_metadata,
                    channel=channel,
                    project=project,
                    priority_bucket=priority_bucket,
                    graph_neighbors=graph_neighbors,
                    impacted_files=impacted_files,
                    related_tests=related_tests,
                    _snapshot=snap,
                ),
            },
            *history,
        ]

        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            self._last_snapshot = snap
            return messages

        messages.append({"role": current_role, "content": merged})
        self._last_snapshot = snap
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages

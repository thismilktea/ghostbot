"""Project-scoped conversation history and runtime metadata."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ghostbot.utils.helpers import ensure_dir, find_legal_message_start, safe_filename

DEFAULT_PROJECT_ID = "default"
_ORIGIN_MAP_FILE = "active_origins.json"


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now()
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.now()


def _normalize_active_file_path(path: str, project_path: str | None, workspace: Path) -> str | None:
    raw = (path or "").strip()
    if not raw:
        return None
    root = Path(project_path).expanduser() if project_path else workspace
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        relative = resolved.relative_to(root.resolve())
    except Exception:
        return None
    normalized = relative.as_posix().strip("/")
    return normalized or None


@dataclass
class ProjectState:
    """Durable state for one active coding project."""

    key: str
    name: str | None = None
    path: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0
    active_files: dict[str, float] = field(default_factory=dict)

    @property
    def project_id(self) -> str:
        return self.key

    def touch_file(self, path: str, max_files: int = 15) -> None:
        self.active_files[path] = time.time()
        sorted_files = sorted(self.active_files.items(), key=lambda x: x[1], reverse=True)
        self.active_files = dict(sorted_files[:max_files])

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:]
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break
        start = find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]
        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        self.messages = []
        self.last_consolidated = 0
        self.active_files.clear()
        self.updated_at = datetime.now()

    def retain_recent_legal_suffix(self, max_messages: int) -> None:
        if max_messages <= 0:
            self.clear()
            return
        if len(self.messages) <= max_messages:
            return
        start_idx = max(0, len(self.messages) - max_messages)
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1
        retained = self.messages[start_idx:]
        start = find_legal_message_start(retained)
        if start:
            retained = retained[start:]
        dropped = len(self.messages) - len(retained)
        self.messages = retained
        self.last_consolidated = max(0, self.last_consolidated - dropped)
        self.updated_at = datetime.now()


class ProjectManager:
    """Manage project-scoped runtime state."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.projects_dir = ensure_dir(self.workspace / "projects")
        self._cache: dict[str, ProjectState] = {}
        self._origin_map_path = self.projects_dir / _ORIGIN_MAP_FILE
        self._origin_map = self._load_origin_map()

    def _project_dir(self, project_id: str) -> Path:
        return ensure_dir(self.projects_dir / safe_filename(project_id))

    def _state_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "state.json"

    def _history_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "history.jsonl"

    def _load_origin_map(self) -> dict[str, str]:
        if not self._origin_map_path.exists():
            return {}
        try:
            data = json.loads(self._origin_map_path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items() if v}
        except Exception as exc:
            logger.warning("Failed to load active project map: {}", exc)
            return {}

    def _save_origin_map(self) -> None:
        self._origin_map_path.write_text(
            json.dumps(self._origin_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_active_for_origin(self, origin_key: str) -> str | None:
        return self._origin_map.get(origin_key) or self._origin_map.get("default")

    def set_active_for_origin(self, origin_key: str, project_id: str) -> None:
        self._origin_map[origin_key] = project_id
        self._origin_map["default"] = project_id
        self._save_origin_map()

    def get_or_create(
        self,
        project_id: str | None,
        *,
        path: str | None = None,
        name: str | None = None,
    ) -> ProjectState:
        key = safe_filename(project_id or DEFAULT_PROJECT_ID)
        if key in self._cache:
            project = self._cache[key]
            changed = False
            if path and project.path != path:
                project.path = path
                project.metadata["active_project_path"] = path
                changed = True
            if name and project.name != name:
                project.name = name
                project.metadata["active_project"] = name
                changed = True
            if changed:
                self.save(project)
            return project
        project = self._load(key)
        if project is None:
            project = ProjectState(key=key, name=name or key, path=path)
            project.metadata.setdefault("active_project", project.name or key)
            if path:
                project.metadata.setdefault("active_project_path", path)
        elif path and project.path != path:
            project.path = path
            project.metadata["active_project_path"] = path
        if name:
            project.name = name
            project.metadata["active_project"] = name
        self._cache[key] = project
        return project

    def _load(self, project_id: str) -> ProjectState | None:
        state_path = self._state_path(project_id)
        history_path = self._history_path(project_id)
        if not state_path.exists() and not history_path.exists():
            return None
        try:
            state = {}
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
            messages = []
            if history_path.exists():
                with open(history_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            messages.append(json.loads(line))
            metadata = state.get("metadata", {})
            project_path = state.get("path") or metadata.get("active_project_path")
            active_files = self._normalize_active_files(state.get("active_files", {}), project_path)
            return ProjectState(
                key=project_id,
                name=state.get("name") or metadata.get("active_project") or project_id,
                path=project_path,
                messages=messages,
                created_at=_parse_dt(state.get("created_at")),
                updated_at=_parse_dt(state.get("updated_at")),
                metadata=metadata,
                last_consolidated=int(state.get("last_consolidated", 0) or 0),
                active_files=active_files,
            )
        except Exception as exc:
            logger.warning("Failed to load project {}: {}", project_id, exc)
            return None

    def _normalize_active_files(
        self,
        active_files: dict[str, float] | None,
        project_path: str | None,
    ) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for path, touched_at in (active_files or {}).items():
            key = _normalize_active_file_path(path, project_path, self.workspace)
            if not key:
                continue
            previous = normalized.get(key)
            if previous is None or touched_at > previous:
                normalized[key] = touched_at
        return normalized

    def save(self, project: ProjectState) -> None:
        project_dir = self._project_dir(project.key)
        history_path = project_dir / "history.jsonl"
        state_path = project_dir / "state.json"
        project.active_files = self._normalize_active_files(project.active_files, project.path)
        project.metadata.setdefault("active_project", project.name or project.key)
        if project.path:
            project.metadata.setdefault("active_project_path", project.path)
        state = {
            "key": project.key,
            "name": project.name or project.key,
            "path": project.path,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
            "metadata": project.metadata,
            "last_consolidated": project.last_consolidated,
            "active_files": project.active_files,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(history_path, "w", encoding="utf-8") as f:
            for msg in project.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._cache[project.key] = project

    def invalidate(self, project_id: str) -> None:
        self._cache.pop(safe_filename(project_id), None)

    def list_projects(self) -> list[dict[str, Any]]:
        projects = []
        for state_path in self.projects_dir.glob("*/state.json"):
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                projects.append({
                    "key": data.get("key") or state_path.parent.name,
                    "name": data.get("name") or state_path.parent.name,
                    "path": data.get("path"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "state_path": str(state_path),
                })
            except Exception:
                continue
        return sorted(projects, key=lambda x: x.get("updated_at", ""), reverse=True)

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.list_projects()

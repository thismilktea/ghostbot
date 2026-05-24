"""Planning workflow helpers for approval-gated coding tasks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from ghostbot.config.paths import get_workspace_path

if TYPE_CHECKING:
    from ghostbot.config.schema import PlanningConfig
    from ghostbot.project import ProjectState


PENDING_PLAN_KEY = "pending_plan"
_PLAN_MEMORY_DIR = Path("memory") / "plans"
_ACTIVE_PLAN_FILE = "active_plan.json"
_HISTORY_DIR = "history"
TaskClass = Literal[
    "simple_explain",
    "research_only",
    "code_change_small",
    "code_change_multi_file",
    "risky_or_destructive",
    "ambiguous",
]
ExecutionMode = Literal["executable", "read_only"]

_PLAN_CHECKLIST_MAX_ITEMS = 30
_PLAN_MAX_STAGES = 5
_PLAN_MAX_ITEMS_PER_STAGE = 8
_PLAN_MAX_LINES = 220


_AFFIRMATIVE = {"yes", "y", "approve", "approved", "go", "go ahead", "proceed", "继续", "同意", "批准", "开始", "执行"}
_NEGATIVE = {"no", "n", "cancel", "stop", "discard", "reject", "取消", "不要", "停止", "拒绝"}
_CODING_CONTEXT = {
    "code", "repo", "file", "test", "bug", "function", "class", "api", "config",
    "cli", "tool", "command", "session", "agent", "代码", "项目", "文件", "测试",
    "函数", "类", "配置", "命令", "工具", "任务", "功能",
}
_REQUIRED_SECTIONS = (
    "Execution Mode",
    "Summary",
    "User Intent",
    "Requirements",
    "Acceptance Criteria",
    "Non-goals / Out of Scope",
    "Exploration Evidence",
    "Proposed Approach",
    "Files Likely to Change",
    "Executable Checklist",
    "Verification Plan",
    "Risks and Open Questions",
)
_EXPLORATION_TOOLS = {"read_file", "list_dir", "glob", "grep", "web_search", "web_fetch"}
_VAGUE_PATTERNS = (
    "update relevant code",
    "modify relevant files",
    "change related code",
    "as needed",
    "相关代码",
    "相关文件",
    "按需",
)
_PLACEHOLDER_SECTION_VALUES = {
    "",
    "tbd",
    "todo",
    "none",
    "n/a",
    "na",
    "not specified",
    "as needed",
    "待定",
    "无",
}
_NO_NONGOALS_PATTERNS = (
    "no additional non-goals identified",
    "no extra out-of-scope items identified",
    "no specific non-goals identified",
    "未识别到额外的非目标",
    "没有额外的非目标",
)
_READ_ONLY_REQUEST_PATTERNS = (
    "--read-only",
    "read only",
    "read-only",
    "no file changes",
    "do not modify",
    "don't modify",
    "do not edit",
    "don't edit",
    "do not execute",
    "don't execute",
    "plan only",
    "只读",
    "不要修改",
    "不要改",
    "不要编辑",
    "不要执行",
    "只分析",
    "只给方案",
    "只规划",
)


@dataclass(slots=True)
class PlanQualityResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    rewrite_instructions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlanState:
    id: str
    status: str
    original_request: str
    plan: str
    created_at: str
    updated_at: str
    revision_count: int = 0
    approved_at: str | None = None
    approval_text: str | None = None
    task_class: TaskClass = "code_change_small"
    execution_mode: ExecutionMode = "executable"
    quality_failures: list[str] = field(default_factory=list)
    checklist: list[dict[str, Any]] = field(default_factory=list)
    content_hash: str = ""
    approved_content_hash: str | None = None
    tools_used: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    block_reason: str | None = None
    completed_at: str | None = None
    active_project: str | None = None
    active_project_path: str | None = None

    @classmethod
    def create(
        cls,
        original_request: str,
        plan: str,
        *,
        task_class: TaskClass = "code_change_small",
        execution_mode: ExecutionMode = "executable",
        quality_failures: list[str] | None = None,
        checklist: list[dict[str, Any]] | None = None,
        tools_used: list[str] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> "PlanState":
        now = datetime.now().isoformat()
        return cls(
            id=f"plan_{uuid4().hex[:12]}",
            status="pending",
            original_request=original_request,
            plan=plan,
            created_at=now,
            updated_at=now,
            task_class=task_class,
            execution_mode=execution_mode,
            quality_failures=quality_failures or [],
            checklist=checklist or extract_plan_checklist(plan),
            content_hash=plan_content_hash(plan),
            tools_used=tools_used or [],
            history=history or [],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlanState | None":
        if not isinstance(data, dict):
            return None
        try:
            plan = str(data["plan"])
            return cls(
                id=str(data["id"]),
                status=str(data.get("status") or "pending"),
                original_request=str(data["original_request"]),
                plan=plan,
                created_at=str(data["created_at"]),
                updated_at=str(data.get("updated_at") or data["created_at"]),
                revision_count=int(data.get("revision_count") or 0),
                approved_at=data.get("approved_at"),
                approval_text=data.get("approval_text"),
                task_class=_coerce_task_class(data.get("task_class")),
                execution_mode=_coerce_execution_mode(data.get("execution_mode"), plan),
                quality_failures=_string_list(data.get("quality_failures")),
                checklist=_dict_list(data.get("checklist")) or extract_plan_checklist(plan),
                content_hash=str(data.get("content_hash") or plan_content_hash(plan)),
                approved_content_hash=data.get("approved_content_hash"),
                tools_used=_string_list(data.get("tools_used")),
                history=_dict_list(data.get("history")),
                block_reason=data.get("block_reason"),
                completed_at=data.get("completed_at"),
                active_project=data.get("active_project"),
                active_project_path=data.get("active_project_path"),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def from_project(cls, project: "ProjectState | None") -> "PlanState | None":
        active = cls.load_active()
        if active is not None:
            return active
        if project is None:
            return None
        return cls.from_dict(project.metadata.get(PENDING_PLAN_KEY))

    @classmethod
    def from_session(cls, session: "ProjectState | None") -> "PlanState | None":
        return cls.from_project(session)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "original_request": self.original_request,
            "plan": self.plan,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision_count": self.revision_count,
            "approved_at": self.approved_at,
            "approval_text": self.approval_text,
            "task_class": self.task_class,
            "execution_mode": self.execution_mode,
            "quality_failures": self.quality_failures,
            "checklist": self.checklist,
            "content_hash": self.content_hash,
            "approved_content_hash": self.approved_content_hash,
            "tools_used": self.tools_used,
            "history": self.history,
            "block_reason": self.block_reason,
            "completed_at": self.completed_at,
            "active_project": self.active_project,
            "active_project_path": self.active_project_path,
        }

    def save_active(self, workspace: Path | None = None) -> None:
        path = active_plan_path(workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_active(cls, workspace: Path | None = None) -> "PlanState | None":
        path = active_plan_path(workspace)
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def clear_active(workspace: Path | None = None) -> None:
        path = active_plan_path(workspace)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def archive(self, workspace: Path | None = None, *, status: str | None = None) -> None:
        if status:
            self.status = status
            self.updated_at = datetime.now().isoformat()
        path = plan_history_path(self.id, workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.clear_active(workspace)

    @classmethod
    def load_from_path(cls, path: str | Path) -> "PlanState | None":
        try:
            data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cls.from_dict(data)

    @classmethod
    def load_history(cls, plan_id: str, workspace: Path | None = None) -> "PlanState | None":
        return cls.load_from_path(plan_history_path(plan_id, workspace))

    def reset_for_loading(self) -> None:
        self.status = "pending"
        self.approved_at = None
        self.approval_text = None
        self.approved_content_hash = None
        self.block_reason = None
        self.completed_at = None
        self.updated_at = datetime.now().isoformat()

    def save_to_project(self, project: "ProjectState") -> None:
        project.metadata[PENDING_PLAN_KEY] = self.to_dict()
        self.save_active()

    def save_to_session(self, session: "ProjectState") -> None:
        self.save_to_project(session)

    @staticmethod
    def clear_from_project(project: "ProjectState") -> None:
        project.metadata.pop(PENDING_PLAN_KEY, None)
        PlanState.clear_active()

    @staticmethod
    def clear_from_session(session: "ProjectState") -> None:
        PlanState.clear_from_project(session)

    def revise(
        self,
        plan: str,
        *,
        task_class: TaskClass | None = None,
        execution_mode: ExecutionMode | None = None,
        quality_failures: list[str] | None = None,
        checklist: list[dict[str, Any]] | None = None,
        tools_used: list[str] | None = None,
        history_limit: int = 10,
    ) -> None:
        self.mark_superseded(history_limit=history_limit)
        self.plan = plan
        self.revision_count += 1
        self.status = "pending"
        self.updated_at = datetime.now().isoformat()
        self.block_reason = None
        self.completed_at = None
        if task_class is not None:
            self.task_class = task_class
        if execution_mode is not None:
            self.execution_mode = execution_mode
        self.quality_failures = quality_failures or []
        self.checklist = checklist or extract_plan_checklist(plan)
        self.content_hash = plan_content_hash(plan)
        self.approved_content_hash = None
        self.approved_at = None
        self.approval_text = None
        self.tools_used = tools_used or []

    def mark_approved(self, approval_text: str | None = None) -> None:
        now = datetime.now().isoformat()
        self.status = "approved"
        self.updated_at = now
        self.approved_at = now
        self.approval_text = approval_text
        self.approved_content_hash = self.content_hash
        self.block_reason = None

    def mark_cancelled(self, *, history_limit: int = 10) -> None:
        self.add_history_entry("cancelled", history_limit=history_limit)
        self.status = "cancelled"
        self.updated_at = datetime.now().isoformat()

    def mark_superseded(self, *, history_limit: int = 10) -> None:
        self.add_history_entry("superseded", history_limit=history_limit)
        self.status = "superseded"
        self.updated_at = datetime.now().isoformat()

    def mark_executing(self) -> None:
        self.status = "executing"
        self.block_reason = None
        self.updated_at = datetime.now().isoformat()

    def mark_blocked(self, reason: str) -> None:
        self.status = "blocked"
        self.block_reason = reason
        self.updated_at = datetime.now().isoformat()

    def mark_completed(self, *, history_limit: int = 10) -> None:
        self.add_history_entry("completed", history_limit=history_limit)
        now = datetime.now().isoformat()
        self.status = "completed"
        self.completed_at = now
        self.updated_at = now

    def mark_executed(self, *, history_limit: int = 10) -> None:
        self.mark_completed(history_limit=history_limit)

    def add_history_entry(
        self,
        status: str,
        *,
        summary: str | None = None,
        quality: dict[str, Any] | None = None,
        history_limit: int = 10,
    ) -> None:
        entry = {
            "id": self.id,
            "status": status,
            "summary": summary or _plan_summary(self.plan),
            "task_class": self.task_class,
            "execution_mode": self.execution_mode,
            "content_hash": self.content_hash,
            "revision_count": self.revision_count,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
        }
        if quality:
            entry["quality"] = quality
        self.history.append(entry)
        self.history = bounded_history(self.history, history_limit)


def plan_memory_dir(workspace: Path | None = None) -> Path:
    root = get_workspace_path(str(workspace)) if workspace is not None else get_workspace_path()
    return root / _PLAN_MEMORY_DIR


def active_plan_path(workspace: Path | None = None) -> Path:
    return plan_memory_dir(workspace) / _ACTIVE_PLAN_FILE


def plan_history_path(plan_id: str, workspace: Path | None = None) -> Path:
    return plan_memory_dir(workspace) / _HISTORY_DIR / f"{plan_id}.json"


def _normalized(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _coerce_task_class(value: Any) -> TaskClass:
    if value in {
        "simple_explain",
        "research_only",
        "code_change_small",
        "code_change_multi_file",
        "risky_or_destructive",
        "ambiguous",
    }:
        return value
    return "code_change_small"


def _coerce_execution_mode(value: Any, plan_text: str = "") -> ExecutionMode:
    if value == "read_only":
        return "read_only"
    if value == "executable":
        return "executable"
    return detect_execution_mode(plan_text)


def detect_execution_mode(text: str) -> ExecutionMode:
    normalized = _normalized(text)
    if any(pattern in normalized for pattern in _READ_ONLY_REQUEST_PATTERNS):
        return "read_only"
    return "executable"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _plan_summary(plan: str) -> str:
    for line in plan.splitlines():
        stripped = line.strip(" #-\t")
        if stripped:
            return stripped[:160]
    return ""


def _section_body(plan_text: str, section: str) -> str:
    match = re.search(rf"^#+\s*{re.escape(section)}\s*$", plan_text or "", re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    next_section = re.search(r"^#+\s+.+$", plan_text[match.end():], re.MULTILINE)
    end = match.end() + next_section.start() if next_section else len(plan_text)
    return plan_text[match.end():end].strip()


def _meaningful_section_body(body: str) -> str:
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^[-*]\s*", "", stripped)
        stripped = re.sub(r"^\[(?: |x|X)\]\s*", "", stripped).strip()
        if stripped:
            lines.append(stripped)
    return " ".join(lines).strip()


def _is_placeholder_body(body: str) -> bool:
    meaningful = _meaningful_section_body(body).lower()
    return meaningful in _PLACEHOLDER_SECTION_VALUES


def _has_bullet_item(body: str) -> bool:
    return bool(re.search(r"^\s*[-*]\s+(?:\[(?: |x|X)\]\s*)?\S", body or "", re.MULTILINE))


def _has_explicit_no_nongoals(body: str) -> bool:
    normalized = _normalized(_meaningful_section_body(body))
    return any(pattern in normalized for pattern in _NO_NONGOALS_PATTERNS)


def is_affirmative_response(text: str) -> bool:
    return _normalized(text) in _AFFIRMATIVE


def is_negative_response(text: str) -> bool:
    return _normalized(text) in _NEGATIVE


def classify_planning_task(text: str, config: "PlanningConfig") -> TaskClass:
    normalized = _normalized(text)
    if not normalized:
        return "ambiguous"
    if any(keyword.lower() in normalized for keyword in getattr(config, "risky_keywords", [])):
        return "risky_or_destructive"
    if any(keyword.lower() in normalized for keyword in getattr(config, "skip_keywords", [])):
        return "simple_explain"
    action = any(keyword.lower() in normalized for keyword in getattr(config, "trigger_keywords", []))
    context = any(token in normalized for token in _CODING_CONTEXT)
    research = any(keyword.lower() in normalized for keyword in getattr(config, "research_keywords", []))
    if action and context:
        multi_markers = ("system", "architecture", "multiple", "across", "workflow", "pipeline", "架构", "流程", "多个")
        return "code_change_multi_file" if any(marker in normalized for marker in multi_markers) else "code_change_small"
    if research or context:
        return "research_only"
    if len(normalized.split()) <= 2:
        return "ambiguous"
    return "simple_explain"


def plan_requires_exploration(task_class: TaskClass) -> bool:
    return task_class in {"research_only", "code_change_small", "code_change_multi_file", "risky_or_destructive"}


def exploration_tools_used(tools_used: list[str]) -> list[str]:
    return [tool for tool in tools_used if tool in _EXPLORATION_TOOLS]


def validate_plan_quality(
    plan_text: str,
    *,
    task_class: TaskClass,
    tools_used: list[str],
    min_exploration_steps: int = 1,
) -> PlanQualityResult:
    failures: list[str] = []
    rewrite: list[str] = []
    text = plan_text or ""
    lower = text.lower()

    for section in _REQUIRED_SECTIONS:
        if not re.search(rf"^#+\s*{re.escape(section)}\s*$", text, re.IGNORECASE | re.MULTILINE):
            failures.append(f"Missing required section: {section}.")
            rewrite.append(f"Add a `{section}` section with concrete content.")

    user_intent = _section_body(text, "User Intent")
    if not user_intent or _is_placeholder_body(user_intent):
        failures.append("User Intent is missing or only placeholder text.")
        rewrite.append("Restate the user's intended outcome in the User Intent section.")

    acceptance = _section_body(text, "Acceptance Criteria")
    if not acceptance or _is_placeholder_body(acceptance) or not _has_bullet_item(acceptance):
        failures.append("Acceptance Criteria must include at least one concrete bullet or checklist item.")
        rewrite.append("Add concrete Acceptance Criteria bullets that define when the task is complete.")

    non_goals = _section_body(text, "Non-goals / Out of Scope")
    if not non_goals or (_is_placeholder_body(non_goals) and not _has_explicit_no_nongoals(non_goals)):
        failures.append("Non-goals / Out of Scope is missing or only placeholder text.")
        rewrite.append("Add concrete Non-goals / Out of Scope bullets, or explicitly state that no additional non-goals were identified.")
    elif not _has_bullet_item(non_goals) and not _has_explicit_no_nongoals(non_goals):
        failures.append("Non-goals / Out of Scope must include a concrete bullet or an explicit no-extra-non-goals statement.")
        rewrite.append("List out-of-scope boundaries as bullets, or explicitly state that no additional non-goals were identified.")

    if plan_requires_exploration(task_class):
        evidence_count = len(exploration_tools_used(tools_used))
        if evidence_count < min_exploration_steps:
            failures.append("Insufficient read-only exploration before planning.")
            rewrite.append("Use read-only tools first and cite the results in Exploration Evidence.")
        if "exploration evidence" not in lower:
            failures.append("Plan does not include exploration evidence.")

    checklist = extract_plan_checklist(text)
    if not checklist:
        failures.append("Plan does not include an executable markdown checklist.")
        rewrite.append("Add an Executable Checklist section with `- [ ]` items.")
    else:
        stage_counts: dict[int, int] = {}
        for item in checklist:
            stage = plan_stage_key(str(item.get("description") or ""))
            if stage:
                stage_counts[stage[0]] = stage_counts.get(stage[0], 0) + 1
        line_count = len([line for line in text.splitlines() if line.strip()])
        overloaded_stages = [stage for stage, count in sorted(stage_counts.items()) if count > _PLAN_MAX_ITEMS_PER_STAGE]
        if len(checklist) > _PLAN_CHECKLIST_MAX_ITEMS:
            failures.append(f"Executable checklist is too detailed: {len(checklist)} items exceeds {_PLAN_CHECKLIST_MAX_ITEMS}.")
            rewrite.append("Collapse the checklist into 3-5 stages and 10-20 user-visible items.")
        if len(stage_counts) > _PLAN_MAX_STAGES:
            failures.append(f"Plan has too many stages: {len(stage_counts)} stages exceeds {_PLAN_MAX_STAGES}.")
            rewrite.append("Combine related phases so the plan has 3-5 major stages.")
        if overloaded_stages:
            failures.append("One or more stages contain too many checklist items: " + ", ".join(f"stage {stage}" for stage in overloaded_stages) + ".")
            rewrite.append("Merge tiny implementation steps under each stage; keep roughly 3-6 bullets per stage and no more than 8 checklist items.")
        if line_count > _PLAN_MAX_LINES:
            failures.append(f"Plan is too long for approval review: {line_count} non-empty lines exceeds {_PLAN_MAX_LINES}.")
            rewrite.append("Shorten the plan by moving hard boundaries into Requirements/Acceptance Criteria instead of repeating micro-steps.")

    if "verification plan" not in lower and "test" not in lower and "pytest" not in lower:
        failures.append("Plan does not include concrete verification steps.")
        rewrite.append("Add focused tests or manual verification steps.")

    if task_class == "risky_or_destructive" and "risk" not in lower and "风险" not in lower:
        failures.append("Risky task does not include an explicit risk discussion.")
        rewrite.append("Add risks, rollback, and approval boundaries.")

    for pattern in _VAGUE_PATTERNS:
        if pattern in lower:
            failures.append(f"Plan contains vague implementation language: {pattern}.")
            rewrite.append("Replace vague steps with file-specific actions.")
            break

    return PlanQualityResult(
        passed=not failures,
        failures=failures,
        rewrite_instructions=rewrite,
    )


def extract_plan_checklist(plan_text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for match in re.finditer(r"^\s*-\s*\[( |x|X)\]\s+(.+?)\s*$", plan_text or "", re.MULTILINE):
        items.append({
            "id": f"step-{len(items) + 1}",
            "description": match.group(2).strip(),
            "status": "completed" if match.group(1).lower() == "x" else "pending",
        })
    return items


def extract_plan_file_paths(plan_text: str) -> list[str]:
    body = _section_body(plan_text, "Files Likely to Change")
    paths: list[str] = []
    for line in body.splitlines():
        match = re.search(r"`([^`]+)`", line)
        candidate = match.group(1) if match else line.strip().lstrip("-* ").split(" — ", 1)[0].split(" - ", 1)[0].strip()
        candidate = candidate.strip()
        if candidate and not candidate.lower().startswith(("none", "n/a", "unknown", "待定")):
            paths.append(candidate)
    return paths


def plan_section(plan_text: str, section: str) -> str:
    return _section_body(plan_text, section)


def plan_section_lines(plan_text: str, section: str, *, limit: int = 5) -> list[str]:
    lines: list[str] = []
    for line in _section_body(plan_text, section).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^[-*]\s*", "", stripped)
        stripped = re.sub(r"^\[(?: |x|X)\]\s*", "", stripped).strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
        if len(lines) >= limit:
            break
    return lines


def plan_stage_key(description: str) -> tuple[int, str] | None:
    match = re.search(r"(?:阶段|phase|stage)\s*([0-9]+)", description or "", re.IGNORECASE)
    if not match:
        return None
    number = int(match.group(1))
    label = f"阶段 {number}"
    tail = re.sub(r"^\s*(?:阶段|phase|stage)\s*[0-9]+\s*[：:：-]?\s*", "", description, flags=re.IGNORECASE).strip()
    if tail:
        label = f"{label}：{tail.split('，', 1)[0].split(',', 1)[0][:40]}"
    return number, label


def aggregate_plan_checklist(checklist: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    if len(checklist) <= limit:
        return [dict(item) for item in checklist]
    grouped: dict[int, dict[str, Any]] = {}
    ungrouped: list[dict[str, Any]] = []
    for item in checklist:
        description = str(item.get("description") or "")
        stage = plan_stage_key(description)
        if stage is None:
            ungrouped.append(dict(item))
            continue
        number, label = stage
        group = grouped.setdefault(number, {"id": f"stage-{number}", "description": label, "status": "pending", "count": 0, "completed": 0})
        group["count"] += 1
        if item.get("status") == "completed":
            group["completed"] += 1
    visible = [grouped[key] for key in sorted(grouped)] + ungrouped
    if len(visible) <= limit:
        return visible
    kept = visible[: max(1, limit - 1)]
    kept.append({
        "id": "more",
        "description": f"其余 {len(visible) - len(kept)} 组/步骤保留在完整计划中",
        "status": "pending",
    })
    return kept


def plan_is_large(plan: "PlanState") -> bool:
    stages = {stage[0] for item in plan.checklist if (stage := plan_stage_key(str(item.get("description") or "")))}
    return len(plan.checklist) > 20 or len(stages) > 3


def build_execution_contract(plan: "PlanState", execution_scope: dict[str, Any] | None = None) -> dict[str, Any]:
    scoped_items = execution_scope.get("checklist") if execution_scope and execution_scope.get("kind") == "phases" else None
    scope_label = str(execution_scope.get("label")) if execution_scope else "全部计划"
    checklist_items = scoped_items or aggregate_plan_checklist(plan.checklist, limit=20)
    summary = plan_section(plan.plan, "Summary") or plan_section(plan.plan, "User Intent") or plan.original_request
    constraints = (
        plan_section_lines(plan.plan, "Requirements", limit=8)
        + plan_section_lines(plan.plan, "Non-goals / Out of Scope", limit=8)
        + plan_section_lines(plan.plan, "Acceptance Criteria", limit=6)
    )
    acceptance_checks = plan_section_lines(plan.plan, "Acceptance Criteria", limit=8)
    if not acceptance_checks:
        acceptance_checks = [str(item.get("description") or "") for item in checklist_items[:8] if item.get("description")]
    return {
        "plan_id": plan.id,
        "content_hash": plan.approved_content_hash or plan.content_hash,
        "execution_mode": plan.execution_mode,
        "summary": summary.strip(),
        "current_scope": scope_label,
        "non_negotiable_constraints": constraints[:14],
        "acceptance_checks": acceptance_checks[:10],
        "checklist": [dict(item) for item in checklist_items],
        "conflict_rule": "执行契约和当前范围优先于旧计划、旧对话、完整计划参考中的冲突内容以及未确认的假设。",
    }


def plan_execution_options(plan: "PlanState") -> list[str]:
    if not plan_is_large(plan):
        return ["回复 `yes` 或使用 `/plan-approve` 开始执行。"]
    stages = sorted({stage[0] for item in plan.checklist if (stage := plan_stage_key(str(item.get("description") or "")))})
    options = ["检测到大型计划，请选择执行范围：", "- `/plan-approve all` — 执行完整计划"]
    if stages:
        first = stages[0]
        options.append(f"- `/plan-approve phase {first}` — 只执行第 {first} 阶段")
        if len(stages) > 1:
            second = stages[min(1, len(stages) - 1)]
            options.append(f"- `/plan-approve phases {stages[0]}-{second}` — 执行第 {stages[0]}-{second} 阶段")
    return options


def parse_plan_execution_scope(text: str, plan: "PlanState") -> dict[str, Any] | None:
    raw = (text or "").strip().lower()
    if not raw or raw in {"/plan-approve", "yes", "y", "approve"}:
        return None
    args = raw.replace("/plan-approve", "", 1).strip()
    if args in {"all", "全部", "所有"}:
        return {"kind": "all", "label": "all"}
    match = re.search(r"(?:phases?|stages?|阶段)\s*([0-9]+)(?:\s*[-~到至]\s*([0-9]+))?", args, re.IGNORECASE)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2) or start)
    if end < start:
        start, end = end, start
    selected = []
    for item in plan.checklist:
        stage = plan_stage_key(str(item.get("description") or ""))
        if stage and start <= stage[0] <= end:
            selected.append(dict(item))
    return {"kind": "phases", "start": start, "end": end, "label": f"阶段 {start}" if start == end else f"阶段 {start}-{end}", "checklist": selected}


def list_archived_plans(workspace: Path | None = None, *, limit: int = 10) -> list[PlanState]:
    history_dir = plan_memory_dir(workspace) / _HISTORY_DIR
    if not history_dir.exists():
        return []
    plans: list[PlanState] = []
    for path in sorted(history_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        plan = PlanState.load_from_path(path)
        if plan is not None:
            plans.append(plan)
        if len(plans) >= limit:
            break
    return plans


def plan_content_hash(plan_text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in (plan_text or "").strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def bounded_history(history: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return history[-limit:]


def should_plan_request(text: str, config: "PlanningConfig") -> bool:
    return False

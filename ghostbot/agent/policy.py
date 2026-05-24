"""Central policy decisions for agent tool execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PolicyDecisionKind = Literal["allow", "deny", "require_approval"]

_WRITE_TOOLS = frozenset({"write_file", "edit_file", "notebook_edit"})
_SHELL_TOOLS = frozenset({"exec"})
_DESTRUCTIVE_COMMAND_PATTERNS = (
    r"\brm\s+-[rf]{1,2}\b",
    r"\bdel\s+/[fq]\b",
    r"\brmdir\s+/s\b",
    r"(?:^|[;&|]\s*)format\b",
    r"\b(mkfs|diskpart)\b",
    r"\b(shutdown|reboot|poweroff)\b",
)
_SAFE_READ_COMMAND_PATTERNS = (
    r"^git\s+(status|diff|log|show|branch)(?:\s|$)",
    r"^python\s+-m\s+py_compile(?:\s|$)",
    r"^python\s+-m\s+pytest(?:\s|$)",
    r"^pytest(?:\s|$)",
)


@dataclass(slots=True)
class PolicyContext:
    approved_plan_id: str | None = None
    approved_content_hash: str | None = None
    allowed_paths: list[str] = field(default_factory=list)
    allowed_roots: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    shell_mode: Literal["disabled", "read_only_commands", "approval_required", "workspace_commands", "unrestricted"] = "workspace_commands"


@dataclass(slots=True)
class PolicyAction:
    tool_name: str
    params: dict[str, Any]
    workspace: Path | None = None
    mode: str = "general"
    session_key: str | None = None
    tool_read_only: bool = False
    side_effect_level: str = "unknown"
    risk_tags: frozenset[str] = field(default_factory=frozenset)
    context: PolicyContext | None = None


@dataclass(slots=True)
class PolicyDecision:
    kind: PolicyDecisionKind
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.kind == "allow"


class PolicyEngine:
    def evaluate(self, action: PolicyAction) -> PolicyDecision:
        if action.tool_read_only and action.side_effect_level == "none":
            return PolicyDecision("allow")
        if action.tool_name in _WRITE_TOOLS:
            return self._evaluate_write(action)
        if action.tool_name in _SHELL_TOOLS:
            return self._evaluate_shell(action)
        if "mcp" in action.risk_tags or action.side_effect_level == "unknown":
            return PolicyDecision("require_approval", "Tool has unknown or external side effects")
        return PolicyDecision("allow")

    def _evaluate_write(self, action: PolicyAction) -> PolicyDecision:
        path = _target_path(action)
        if path is None:
            return PolicyDecision("deny", "Write tool call has no target path")
        if action.workspace is None:
            return PolicyDecision("require_approval", "Write tool has no workspace boundary")
        context = action.context
        roots = [action.workspace, *[Path(root) for root in context.allowed_roots]] if context else [action.workspace]
        try:
            _resolve_under_roots(path, roots)
        except PermissionError as exc:
            return PolicyDecision("deny", str(exc))
        if context and context.approved_plan_id:
            return PolicyDecision("allow")
        return PolicyDecision("allow")

    def _evaluate_shell(self, action: PolicyAction) -> PolicyDecision:
        command = str(action.params.get("command") or "").strip()
        if not command:
            return PolicyDecision("deny", "Shell command is empty")
        lower = command.lower()
        for pattern in _DESTRUCTIVE_COMMAND_PATTERNS:
            if re.search(pattern, lower):
                return PolicyDecision("deny", "Shell command matches a destructive pattern")
        mode = action.context.shell_mode if action.context else "workspace_commands"
        if mode == "disabled":
            return PolicyDecision("deny", "Shell execution is disabled by policy")
        if mode == "unrestricted":
            return PolicyDecision("allow")
        if mode == "approval_required":
            return PolicyDecision("require_approval", "Shell execution requires approval")
        if action.context and command in action.context.verification_commands:
            return PolicyDecision("allow")
        if any(re.search(pattern, lower) for pattern in _SAFE_READ_COMMAND_PATTERNS):
            return PolicyDecision("allow")
        if mode == "read_only_commands":
            return PolicyDecision("require_approval", "Shell command is not in the read-only allowlist")
        if mode == "workspace_commands" and action.workspace is not None:
            if action.context and action.context.approved_plan_id:
                return PolicyDecision("allow")
            return PolicyDecision("require_approval", "Shell command requires an approved executable plan")
        return PolicyDecision("require_approval", "Shell command requires approval")


def _target_path(action: PolicyAction) -> str | None:
    path = action.params.get("path") or action.params.get("file_path") or action.params.get("notebook_path")
    return str(path) if path else None


def _resolve_under_roots(path: str, roots: list[Path | None]) -> Path:
    valid_roots = [root.expanduser().resolve() for root in roots if root is not None]
    if not valid_roots:
        raise PermissionError("No allowed workspace or project boundary is configured")
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = valid_roots[0] / target
    resolved = target.resolve()
    for root in valid_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    allowed = ", ".join(str(root) for root in valid_roots)
    raise PermissionError(f"Path {path} is outside allowed roots: {allowed}")


def _resolve_under_workspace(path: str, workspace: Path) -> Path:
    return _resolve_under_roots(path, [workspace])

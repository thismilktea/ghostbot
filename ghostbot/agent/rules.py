"""User-configurable behavior rules engine.

Rules are loaded from `.ghostbot/rules/*.md` files with YAML frontmatter.
They can block or warn on tool calls matching specified patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass(slots=True)
class Rule:
    name: str
    enabled: bool
    event: str
    tool: str
    action: str
    pattern: str | None = None
    conditions: list[dict[str, str]] = field(default_factory=list)
    message: str = ""

    def matches(self, tool_name: str, params: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        if self.tool != "*" and self.tool != tool_name:
            return False
        if self.pattern:
            text = _extract_matchable_text(tool_name, params)
            if not re.search(self.pattern, text):
                return False
        if self.conditions:
            if not _evaluate_conditions(self.conditions, tool_name, params):
                return False
        return True


@dataclass(slots=True)
class RuleDecision:
    matched: bool
    action: str = "allow"
    rule_name: str = ""
    message: str = ""


class RuleEngine:
    """Loads and evaluates user-defined behavior rules."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._rules: list[Rule] = []
        self.reload()

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)

    def reload(self) -> None:
        self._rules = load_rules(self._workspace)

    def evaluate(self, event: str, tool_name: str, params: dict[str, Any]) -> RuleDecision:
        for rule in self._rules:
            if not rule.enabled:
                continue
            if rule.event not in (event, "all"):
                continue
            if rule.matches(tool_name, params):
                return RuleDecision(
                    matched=True,
                    action=rule.action,
                    rule_name=rule.name,
                    message=rule.message,
                )
        return RuleDecision(matched=False)


def load_rules(workspace: Path) -> list[Rule]:
    """Load all rule files from .ghostbot/rules/ directory."""
    rules_dir = workspace / ".ghostbot" / "rules"
    if not rules_dir.is_dir():
        return []

    rules: list[Rule] = []
    for path in sorted(rules_dir.glob("*.md")):
        try:
            rule = _parse_rule_file(path)
            if rule:
                rules.append(rule)
        except Exception as e:
            logger.warning("Failed to parse rule file {}: {}", path.name, e)
    return rules


def _parse_rule_file(path: Path) -> Rule | None:
    """Parse a single rule markdown file with YAML frontmatter."""
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    frontmatter_text = parts[1].strip()
    body = parts[2].strip()

    frontmatter = _parse_simple_yaml(frontmatter_text)
    if not frontmatter.get("name"):
        return None

    conditions_raw = frontmatter.get("conditions")
    conditions: list[dict[str, str]] = []
    if isinstance(conditions_raw, list):
        conditions = [c for c in conditions_raw if isinstance(c, dict)]

    return Rule(
        name=str(frontmatter.get("name", "")),
        enabled=_parse_bool(frontmatter.get("enabled", "true")),
        event=str(frontmatter.get("event", "before_execute_tools")),
        tool=str(frontmatter.get("tool", "*")),
        action=str(frontmatter.get("action", "warn")),
        pattern=frontmatter.get("pattern"),
        conditions=conditions,
        message=body or f"Rule '{frontmatter.get('name')}' triggered.",
    )


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal YAML-like parser for frontmatter (no dependency on PyYAML)."""
    result: dict[str, Any] = {}
    current_list_key: str | None = None
    current_list: list[Any] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if line.startswith("  - ") and current_list_key:
            item_text = line[4:].strip()
            if ":" in item_text and not item_text.startswith('"'):
                item_dict: dict[str, str] = {}
                for part in re.split(r",\s*", item_text):
                    if ":" in part:
                        k, v = part.split(":", 1)
                        item_dict[k.strip()] = v.strip().strip("'\"")
                current_list.append(item_dict)
            else:
                current_list.append(item_text.strip("'\""))
            continue

        if current_list_key and current_list:
            result[current_list_key] = current_list
            current_list_key = None
            current_list = []

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if not value:
                current_list_key = key
                current_list = []
            else:
                result[key] = value

    if current_list_key and current_list:
        result[current_list_key] = current_list

    return result


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "on")


def _extract_matchable_text(tool_name: str, params: dict[str, Any]) -> str:
    """Build a searchable text from tool parameters."""
    parts: list[str] = [tool_name]
    for key in ("command", "content", "new_content", "path", "file_path", "pattern", "query"):
        val = params.get(key)
        if isinstance(val, str):
            parts.append(val)
    return "\n".join(parts)


def _evaluate_conditions(conditions: list[dict[str, str]], tool_name: str, params: dict[str, Any]) -> bool:
    """Evaluate all conditions (AND logic)."""
    for cond in conditions:
        field_name = cond.get("field", "")
        operator = cond.get("operator", "contains")
        pattern = cond.get("pattern", "")

        if field_name == "tool_name":
            value = tool_name
        else:
            value = str(params.get(field_name, ""))

        if not _check_operator(operator, value, pattern):
            return False
    return True


def _check_operator(operator: str, value: str, pattern: str) -> bool:
    if operator == "contains":
        return pattern in value
    if operator == "not_contains":
        return pattern not in value
    if operator == "equals":
        return value == pattern
    if operator == "starts_with":
        return value.startswith(pattern)
    if operator == "ends_with":
        return value.endswith(pattern)
    if operator == "regex_match":
        return bool(re.search(pattern, value))
    return False

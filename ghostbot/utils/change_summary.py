"""Helpers for user-visible file change summaries."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from ghostbot.utils.path import abbreviate_path

_MAX_DIFF_LINES = 28
_MAX_HUNKS = 2
_MAX_LINE_CHARS = 160
_MAX_TOTAL_CHARS = 3000


def build_change_summary(path: str | Path, before: str | None, after: str | None) -> dict[str, Any] | None:
    if before is None or after is None or before == after:
        return None

    path_text = abbreviate_path(str(path))
    diff_lines = list(difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=str(path),
        tofile=str(path),
        lineterm="",
        n=3,
    ))
    additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removals = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    operation = "Created" if before == "" else "Modified"
    formatted = format_change_summary(path_text, operation, additions, removals, diff_lines)
    return {
        "path": str(path),
        "operation": operation.lower(),
        "additions": additions,
        "removals": removals,
        "diff": "\n".join(diff_lines),
        "formatted": formatted,
    }


def format_change_summary(path: str, operation: str, additions: int, removals: int, diff_lines: list[str]) -> str:
    header = f"{operation} {path} (+{additions} -{removals})"
    body = _trim_diff(diff_lines)
    if not body:
        return header
    return header + "\n" + "\n".join(body)


def _trim_diff(lines: list[str]) -> list[str]:
    trimmed: list[str] = []
    hunks = 0
    truncated = False
    total = 0
    for line in lines:
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            hunks += 1
            if hunks > _MAX_HUNKS:
                truncated = True
                break
        shortened = line if len(line) <= _MAX_LINE_CHARS else line[: _MAX_LINE_CHARS - 1] + "…"
        if len(trimmed) >= _MAX_DIFF_LINES or total + len(shortened) + 1 > _MAX_TOTAL_CHARS:
            truncated = True
            break
        trimmed.append(shortened)
        total += len(shortened) + 1
    if truncated:
        trimmed.append("… diff truncated …")
    return trimmed

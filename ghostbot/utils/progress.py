"""Helpers for extracting user-visible execution progress."""

from __future__ import annotations

import re

_MAX_PROGRESS_CHARS = 150
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_PROGRESS_PATTERNS = [
    re.compile(r"^\s*(?:Step\s+\d+|Stage\s+\d+(?:\s*/\s*\d+)?|Phase\s+\d+|Plan\s+item\s+\d+)\s*[:：-]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\s*(?:Next|Now)\s*[:：-]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\s*(?:第\s*\d+\s*阶段|步骤\s*\d+)\s*[:：-]\s*(.+)$"),
    re.compile(r"^\s*(?:接下来|当前)\s*[:：-]\s*(.+)$"),
    re.compile(r"^\s*-\s*\[[ xX]\]\s*(.+)$"),
]
_PREFIX_RE = re.compile(r"^\s*(?P<prefix>(?:Step\s+\d+|Stage\s+\d+(?:\s*/\s*\d+)?|Phase\s+\d+|Plan\s+item\s+\d+|Next|Now|第\s*\d+\s*阶段|步骤\s*\d+|接下来|当前))\s*[:：-]\s*", re.IGNORECASE)


def extract_plan_progress(text: str | None) -> str | None:
    if not text:
        return None
    clean = _THINK_RE.sub("", text)
    for raw_line in clean.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        for pattern in _PROGRESS_PATTERNS:
            if not pattern.match(line):
                continue
            return _truncate_progress(_normalize_prefix(line))
    return None


def _normalize_prefix(line: str) -> str:
    match = _PREFIX_RE.match(line)
    if not match:
        return line
    prefix = " ".join(match.group("prefix").split())
    rest = line[match.end():].strip()
    return f"{prefix}: {rest}" if rest else prefix


def _truncate_progress(text: str) -> str:
    if len(text) <= _MAX_PROGRESS_CHARS:
        return text
    return text[: _MAX_PROGRESS_CHARS - 1].rstrip() + "…"

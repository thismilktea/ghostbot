from __future__ import annotations

from rich.console import Console

from ghostbot.cli.commands import _render_progress_line
from ghostbot.utils.change_summary import build_change_summary
from ghostbot.utils.progress import extract_plan_progress
from ghostbot.utils.tool_hints import format_tool_hints


class _ToolCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def test_extract_plan_progress_recognizes_english_and_chinese_markers():
    assert extract_plan_progress("Step 2: update runner") == "Step 2: update runner"
    assert extract_plan_progress("第 2 阶段：更新终端输出") == "第 2 阶段: 更新终端输出"
    assert extract_plan_progress("- [ ] wire change summary") == "- [ ] wire change summary"
    assert extract_plan_progress("<think>secret</think>\nNow: run verification") == "Now: run verification"


def test_extract_plan_progress_ignores_plain_paragraphs():
    text = "I inspected the repository and will now make several coordinated changes."
    assert extract_plan_progress(text) is None


def test_tool_hints_describe_active_tool_actions():
    calls = [
        _ToolCall("read_file", {"path": "tests/agent/test_runner.py"}),
        _ToolCall("edit_file", {"path": "ghostbot/agent/runner.py"}),
        _ToolCall("notebook_edit", {"path": "analysis/demo.ipynb"}),
        _ToolCall("exec", {"command": "python -m pytest tests/agent/test_runner.py"}),
    ]

    hints = format_tool_hints(calls)

    assert "Reading tests/agent/test_runner.py" in hints
    assert "Editing ghostbot/agent/runner.py" in hints
    assert "Editing notebook analysis/demo.ipynb" in hints
    assert "Running python -m pytest tests/agent/test_runne…" in hints


def test_change_summary_formats_counts_and_diff():
    summary = build_change_summary("demo.py", "a\nb\n", "a\nc\n")

    assert summary is not None
    assert summary["additions"] == 1
    assert summary["removals"] == 1
    assert summary["formatted"].startswith("Modified demo.py (+1 -1)")
    assert "-b" in summary["formatted"]
    assert "+c" in summary["formatted"]


def test_cli_progress_renderer_indents_multiline_output():
    console = Console(record=True, width=120, color_system=None)

    _render_progress_line(console, "Modified demo.py (+1 -1)\n@@ -1 +1 @@\n-a = 1\n+a = 2")

    output = console.export_text()
    assert "  ↳ Modified demo.py (+1 -1)" in output
    assert "      @@ -1 +1 @@" in output
    assert "      -a = 1" in output
    assert "      +a = 2" in output

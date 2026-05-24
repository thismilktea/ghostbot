from __future__ import annotations

import json

import pytest

from ghostbot.agent.tools.base import ToolResult
from ghostbot.agent.tools.filesystem import EditFileTool, WriteFileTool
from ghostbot.agent.tools.notebook import NotebookEditTool


@pytest.mark.asyncio
async def test_write_file_returns_change_summary_for_created_file(tmp_path):
    tool = WriteFileTool(workspace=tmp_path)

    result = await tool.execute(path="demo.py", content="print('hi')\n")

    assert isinstance(result, ToolResult)
    assert result.content.startswith("Successfully wrote")
    summary = result.metadata["change_summary"]
    assert summary["operation"] == "created"
    assert summary["additions"] == 1
    assert summary["removals"] == 0
    assert "+print('hi')" in summary["formatted"]


@pytest.mark.asyncio
async def test_edit_file_returns_change_summary_for_successful_edit(tmp_path):
    (tmp_path / "demo.py").write_text("a = 1\n", encoding="utf-8")
    tool = EditFileTool(workspace=tmp_path)

    result = await tool.execute(path="demo.py", old_text="a = 1", new_text="a = 2")

    assert isinstance(result, ToolResult)
    summary = result.metadata["change_summary"]
    assert summary["operation"] == "modified"
    assert summary["additions"] == 1
    assert summary["removals"] == 1
    assert "-a = 1" in summary["formatted"]
    assert "+a = 2" in summary["formatted"]


@pytest.mark.asyncio
async def test_failed_edit_does_not_return_change_summary(tmp_path):
    (tmp_path / "demo.py").write_text("a = 1\n", encoding="utf-8")
    tool = EditFileTool(workspace=tmp_path)

    result = await tool.execute(path="demo.py", old_text="missing", new_text="x")

    assert isinstance(result, str)
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_notebook_edit_returns_change_summary(tmp_path):
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [{"cell_type": "code", "source": "a = 1", "metadata": {}, "outputs": [], "execution_count": None}],
    }
    (tmp_path / "demo.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    tool = NotebookEditTool(workspace=tmp_path)

    result = await tool.execute(path="demo.ipynb", cell_index=0, new_source="a = 2")

    assert isinstance(result, ToolResult)
    summary = result.metadata["change_summary"]
    assert summary["operation"] == "modified"
    assert "a = 2" in summary["formatted"]

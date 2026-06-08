from __future__ import annotations

from pathlib import Path
from typing import Any

from ghostbot.agent.tools.base import Tool, tool_parameters
from ghostbot.utils.project_analyzer import ProjectAnalyzer


class _GraphTool(Tool):
    def __init__(self, workspace: Path, project_graph_provider):
        super().__init__()
        self.workspace = workspace
        self._project_graph_provider = project_graph_provider

    @property
    def read_only(self) -> bool:
        return True

    def _analyzer(self) -> ProjectAnalyzer | None:
        info = self._project_graph_provider() or {}
        project_path = info.get("project_path")
        graph_path = info.get("graph_path")
        if not project_path or not graph_path:
            return None
        analyzer = ProjectAnalyzer(Path(project_path), self.workspace / "memory")
        analyzer.graph_file = Path(graph_path)
        return analyzer


@tool_parameters({
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Symbol name, file path, or subsystem keyword to query in the active project graph.",
            "minLength": 1,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (default 10).",
            "minimum": 1,
            "maximum": 50,
        },
    },
    "required": ["query"],
})
class FindSymbolTool(_GraphTool):
    @property
    def name(self) -> str:
        return "find_symbol"

    @property
    def description(self) -> str:
        return "Query the active project graph for symbol definitions instead of grepping blindly."

    async def execute(self, query: str, limit: int = 10, **kwargs: Any) -> str:
        analyzer = self._analyzer()
        if analyzer is None:
            return "No active project graph is available. Run /scan or /use a scanned project first."
        matches = analyzer.find_symbol(query, limit=limit)
        if not matches:
            return f"No symbols matched '{query}'."
        lines = [f"Found {len(matches)} symbol match(es) for '{query}':"]
        for item in matches:
            lines.append(
                f"- {item.get('name')} ({item.get('kind')}) — {item.get('file_path')}"
            )
        return "\n".join(lines)


@tool_parameters({
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Symbol name used to find inbound references in the active project graph.",
            "minLength": 1,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of callers to return (default 10).",
            "minimum": 1,
            "maximum": 50,
        },
    },
    "required": ["query"],
})
class FindCallersTool(_GraphTool):
    @property
    def name(self) -> str:
        return "find_callers"

    @property
    def description(self) -> str:
        return "Use the active project graph to find files that call or import a symbol."

    async def execute(self, query: str, limit: int = 10, **kwargs: Any) -> str:
        analyzer = self._analyzer()
        if analyzer is None:
            return "No active project graph is available. Run /scan or /use a scanned project first."
        matches = analyzer.find_callers(query, limit=limit)
        if not matches:
            return f"No callers found for '{query}'."
        lines = [f"Found {len(matches)} caller(s) for '{query}':"]
        for item in matches:
            lines.append(
                f"- {item.get('source_file')} -> {item.get('target_symbol')} via {item.get('ref_name')}"
            )
        return "\n".join(lines)


@tool_parameters({
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Symbol name used to find outbound references from its defining file.",
            "minLength": 1,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of callees to return (default 10).",
            "minimum": 1,
            "maximum": 50,
        },
    },
    "required": ["query"],
})
class FindCalleesTool(_GraphTool):
    @property
    def name(self) -> str:
        return "find_callees"

    @property
    def description(self) -> str:
        return "Use the active project graph to find symbols referenced from a symbol's defining file."

    async def execute(self, query: str, limit: int = 10, **kwargs: Any) -> str:
        analyzer = self._analyzer()
        if analyzer is None:
            return "No active project graph is available. Run /scan or /use a scanned project first."
        matches = analyzer.find_callees(query, limit=limit)
        if not matches:
            return f"No callees found for '{query}'."
        lines = [f"Found {len(matches)} callee(s) for '{query}':"]
        for item in matches:
            lines.append(
                f"- {item.get('source_file')} -> {item.get('target_symbol')} via {item.get('ref_name')}"
            )
        return "\n".join(lines)


@tool_parameters({
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "File path or symbol name used to find structurally related files.",
            "minLength": 1,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of related files to return (default 10).",
            "minimum": 1,
            "maximum": 50,
        },
    },
    "required": ["query"],
})
class FindRelatedFilesTool(_GraphTool):
    @property
    def name(self) -> str:
        return "find_related_files"

    @property
    def description(self) -> str:
        return "Use the active project graph to find structurally related files for the current symbol or path."

    async def execute(self, query: str, limit: int = 10, **kwargs: Any) -> str:
        analyzer = self._analyzer()
        if analyzer is None:
            return "No active project graph is available. Run /scan or /use a scanned project first."
        matches = analyzer.find_related_files(query, limit=limit)
        if not matches:
            return f"No related files found for '{query}'."
        lines = [f"Found {len(matches)} related file(s) for '{query}':"]
        for item in matches:
            reasons = "; ".join(item.get("reasons", []))
            lines.append(
                f"- {item.get('file_path')} (score={item.get('score', 0):.1f}) — {reasons}"
            )
        return "\n".join(lines)


@tool_parameters({
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "File path or symbol name used to estimate impacted files in the active project graph.",
            "minLength": 1,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of impacted files to return (default 10).",
            "minimum": 1,
            "maximum": 50,
        },
    },
    "required": ["query"],
})
class FindImpactedFilesTool(_GraphTool):
    @property
    def name(self) -> str:
        return "find_impacted_files"

    @property
    def description(self) -> str:
        return "Use the active project graph to estimate likely impacted files for a symbol or file path."

    async def execute(self, query: str, limit: int = 10, **kwargs: Any) -> str:
        analyzer = self._analyzer()
        if analyzer is None:
            return "No active project graph is available. Run /scan or /use a scanned project first."
        matches = analyzer.find_impacted_files(query, limit=limit)
        if not matches:
            return f"No impacted files found for '{query}'."
        lines = [f"Found {len(matches)} impacted file(s) for '{query}':"]
        for item in matches:
            reasons = "; ".join(item.get("reasons", []))
            lines.append(
                f"- {item.get('file_path')} (score={item.get('score', 0):.1f}) — {reasons}"
            )
        return "\n".join(lines)

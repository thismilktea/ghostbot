from ghostbot.agent.tools.base import Tool, tool_parameters
from ghostbot.agent.tools.registry import ToolRegistry
from ghostbot.agent.tools.schema import StringSchema, tool_parameters_schema


@tool_parameters(tool_parameters_schema(value=StringSchema("value")))
class _ReadTool(Tool):
    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return "read"

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs):
        return "ok"


@tool_parameters(tool_parameters_schema(value=StringSchema("value")))
class _WriteTool(Tool):
    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return "write"

    async def execute(self, **kwargs):
        return "ok"


def test_filtered_keeps_matching_tools():
    registry = ToolRegistry()
    registry.register(_ReadTool())
    registry.register(_WriteTool())

    filtered = registry.filtered(lambda tool: tool.name == "read")

    assert filtered.tool_names == ["read"]


def test_read_only_keeps_only_read_only_tools():
    registry = ToolRegistry()
    registry.register(_ReadTool())
    registry.register(_WriteTool())

    filtered = registry.read_only()

    assert filtered.has("read")
    assert not filtered.has("write")


def test_read_only_definitions_exclude_mutating_tools():
    registry = ToolRegistry()
    registry.register(_ReadTool())
    registry.register(_WriteTool())

    definitions = registry.read_only().get_definitions()
    names = [definition["function"]["name"] for definition in definitions]

    assert names == ["read"]

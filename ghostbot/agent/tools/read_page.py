from typing import Any
from ghostbot.agent.tools.base import Tool, tool_parameters
from ghostbot.agent.tools.schema import StringSchema, tool_parameters_schema
from ghostbot.agent.memory import MemoryStore

@tool_parameters(
    tool_parameters_schema(
        pointer=StringSchema("The 8-character memory pointer (e.g., a1b2c3d4)"),
        required=["pointer"],
    )
)
class ReadPageTool(Tool):
    """读取被 Swap 换出的超长文件内容"""
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "read_page"

    @property
    def description(self) -> str:
        return "When you see [指针: xxx] or [Pointer: xxx] in the context, use this tool to read the hidden content."

    async def execute(self, pointer: str, **kwargs: Any) -> str:
        content = self.memory_store.page_in(pointer)
        if content is None:
            return f"Error: Swap page '{pointer}' not found or expired."
        return content
# 文件路径: ghostbot/agent/tools/search_memory.py

import asyncio
from typing import Any

from ghostbot.agent.tools.base import Tool, tool_parameters


@tool_parameters({
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "要搜索的关键词。由于底层是精确代码级检索，请务必优先提取【原始英文报错类名】、【完整变量名】或【特定文件名】（如 'NullPointerException' 或 'VectorStoreManager'），其次才是中文描述。"
        }
    },
    "required": ["query"]
})
class SearchMemoryTool(Tool):
    def __init__(self, search_engine: Any):
        super().__init__()
        self.search_engine = search_engine

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return "当系统自动提供的上下文中缺少你需要的信息时，调用此工具从长期记忆中检索候选记忆卡片，再按需展开，不要凭空捏造。"

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, query: str, **kwargs: Any) -> str:
        try:
            records = await asyncio.to_thread(self.search_engine.search_records, query, 5, 0)

            if not records:
                return f"❌ 记忆库中未找到关于 '{query}' 的记录。请尝试更宽泛的关键词，或换用具体英文变量名重新搜索。"

            lines = [f"🔍 记忆库中检索到关于 '{query}' 的 {len(records)} 条候选记忆卡片：\n"]
            for idx, record in enumerate(records, 1):
                lines.append(
                    f'<memory_card index="{idx}" cursor="{record.cursor}" type="{record.record_type}" scope="{record.scope}">'
                )
                lines.append(f"Summary: {record.summary}")
                lines.append(record.content.strip())
                lines.append("</memory_card>\n")

            lines.append("💡 [系统提示：优先使用 memory card 的摘要与类型；只有确实需要证据时再展开其中细节。]")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 搜索记忆时发生底层错误: {str(e)}"

# 文件路径: ghostbot/agent/tools/search_memory.py

import asyncio
from typing import Any
from ghostbot.agent.tools.base import Tool, tool_parameters


@tool_parameters({
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            # 💡 优化 1：诱导大模型使用极其精确的代码特征词
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
        # 💡 优化 2：明确调用时机
        return "当系统自动提供的上下文中缺少你需要的信息（如用户提及过去的报错、历史配置、或者你忘了某个上下文）时，必须调用此工具去长期记忆库中检索。不要凭空捏造。"

    @property
    def read_only(self) -> bool:
        return True  # 只读操作，允许高并发

    async def execute(self, query: str, **kwargs: Any) -> str:
        try:
            # 💡 优化 3：异步防阻塞！将同步的 sqlite3 查询放入线程池执行
            # 避免搜索时卡死整个 ghostbot 的并发事件循环
            results = await asyncio.to_thread(self.search_engine.search, query, top_k=5)

            if not results:
                return f"❌ 记忆库中未找到关于 '{query}' 的记录。请尝试更换更宽泛的关键词，或换用具体的英文变量名重新搜索。"

            # 💡 优化 4：使用 XML 标签构建“沙盒”，防止大模型产生阅读混乱
            formatted_lines = [f"🔍 记忆库中检索到关于 '{query}' 的 Top-{len(results)} 条历史片段：\n"]

            for idx, (cursor, text) in enumerate(results, 1):
                # 使用结构化的 XML 标签包裹，让 LLM 明确知道这是“历史数据”，而不是“系统指令”
                formatted_lines.append(f'<record index="{idx}" time_cursor="{cursor}">')
                formatted_lines.append(text.strip())
                formatted_lines.append("</record>\n")

            formatted_lines.append("💡 [系统提示：请提取上方 <record> 中的有效信息来回答用户，或基于此继续思考。]")

            return "\n".join(formatted_lines)

        except Exception as e:
            # 捕获异常，防止工具崩溃导致整个 Agent 宕机
            return f"❌ 搜索记忆时发生底层错误: {str(e)}"
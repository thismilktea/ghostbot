import re
from pathlib import Path
from typing import Any
from ghostbot.agent.tools.base import Tool, tool_parameters


@tool_parameters({
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "要搜索的目标文件路径"},
        "regex_pattern": {"type": "string", "description": "用于搜索的正则表达式，例如 'print\\(' 或 'def analyze'"},
        "context_lines": {"type": "integer", "description": "返回匹配行上下几行的上下文，默认 2 行", "default": 2}
    },
    "required": ["file_path", "regex_pattern"]
})
class SearchCodeTool(Tool):
    def __init__(self, workspace: Path):
        super().__init__()
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "search_code"

    @property
    def description(self) -> str:
        return "狙击枪：使用正则精确搜索文件内的代码段。不要用 read_file 找代码，先用本工具定位行号和上下文！"

    async def execute(self, file_path: str, regex_pattern: str, context_lines: int = 2, **kwargs: Any) -> str:
        target_path = self.workspace / file_path
        if not target_path.exists() or not target_path.is_file():
            return f"❌ 错误: 文件 {file_path} 不存在。"

        try:
            content = target_path.read_text(encoding="utf-8").splitlines()
            pattern = re.compile(regex_pattern)

            matches = []
            for i, line in enumerate(content):
                if pattern.search(line):
                    # 计算上下文的起始和结束行 (注意行号从 1 开始)
                    start = max(0, i - context_lines)
                    end = min(len(content), i + context_lines + 1)

                    snippet = []
                    for j in range(start, end):
                        prefix = ">> " if j == i else "   "  # 高亮匹配行
                        snippet.append(f"{j + 1:4d} | {prefix}{content[j]}")

                    matches.append("\n".join(snippet))

            if not matches:
                return f"⚠️ 在 {file_path} 中未找到匹配 '{regex_pattern}' 的代码。"

            # 限制返回数量，防止正则写太宽泛导致依然爆 Token
            result = "\n\n...[分隔]...\n\n".join(matches[:10])
            if len(matches) > 10:
                result += f"\n\n(提示: 还有 {len(matches) - 10} 处匹配被折叠，请使用更精确的正则)"

            return f"🔍 在 {file_path} 中找到以下匹配：\n```python\n{result}\n```"

        except Exception as e:
            return f"❌ 搜索失败: {e}"
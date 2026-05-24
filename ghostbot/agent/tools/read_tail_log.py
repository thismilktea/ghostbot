# ghostbot/agent/tools/read_tail_log.py

import os
from collections import deque
from typing import Any
from ghostbot.agent.tools.base import Tool, tool_parameters


@tool_parameters({
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "日志文件的绝对或相对路径"},
        "lines": {"type": "integer", "description": "要读取的最后行数，默认 50，最大 200"}
    },
    "required": ["file_path"]
})
class ReadTailLogTool(Tool):

    @property
    def name(self) -> str:
        return "read_tail_log"

    @property
    def description(self) -> str:
        return "用于安全地读取大型日志文件的最后 N 行。当用户要求“看报错日志”时必须使用此工具，防止内存溢出。"

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, file_path: str, lines: int = 50, **kwargs: Any) -> str:
        lines = min(lines, 200)  # 强制安全锁

        if not os.path.exists(file_path):
            return f"❌ 找不到日志文件: {file_path}"

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                tail_lines = deque(f, maxlen=lines)

            content = "".join(tail_lines)
            return f"📄 {file_path} 的最后 {len(tail_lines)} 行日志如下:\n\n```text\n{content}\n```"
        except Exception as e:
            return f"❌ 读取日志失败: {str(e)}"
import subprocess
from loguru import logger
from ghostbot.agent.hook import AgentHook, AgentHookContext


class LinterHook(AgentHook):
    def __init__(self, workspace, max_retries=2):
        super().__init__()
        self.workspace = workspace
        self.max_retries = max_retries
        self._retry_count = 0
        self._last_error = None

    async def after_iteration(self, context: AgentHookContext) -> None:
        """每一轮迭代后，如果 AI 修改了代码，立刻进行 Linter 检查"""

        # 1. 检查是否有文件修改
        modify_tools = {"edit_file", "write_file", "replace_file"}
        has_modify = any(tc.name in modify_tools for tc in context.tool_calls)

        if not has_modify:
            self._retry_count = 0  # 没有修改，重置计数器
            return

        # 2. 执行 Linter (推荐使用 ruff，比 flake8 快得多，且提示更智能)
        # 你可以把 'ruff check' 改成 'flake8' 或 'mypy'
        logger.info("🔍 [LinterHook] 正在执行语法检查...")
        try:
            result = subprocess.run(
                ["ruff", "check", "."],  # 假设你项目里装了 ruff
                cwd=self.workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10
            )
        except Exception as e:
            logger.warning(f"Linter 执行失败: {e}")
            return

        # 3. 如果通过了检查 (exit code 0)
        if result.returncode == 0:
            self._retry_count = 0
            return

        # 4. 如果没通过，自动注入错误反馈
        error_msg = result.stdout + result.stderr
        if self._retry_count < self.max_retries:
            self._retry_count += 1
            logger.warning(f"⚠️ [LinterHook] 发现语法错误，第 {self._retry_count} 次自动修复中...")

            # 💡 核心魔法：直接往 Session 里塞一条错误信息
            # 下一轮循环时，大模型会以为这是“用户”发来的错误报告
            error_prompt = f"🚨 代码检查发现错误，请修复以下问题并重新尝试:\n\n{error_msg}"

            # ⚠️ 注意：这里访问 context.session 需要确认你的 hook 注入了 session
            # 如果你的 hook 拿不到 session，可以在 loop 初始化时把 session 传进去
            if hasattr(context, 'session') and context.session:
                context.session.messages.append({"role": "user", "content": error_prompt})
        else:
            logger.error("❌ [LinterHook] 达到最大重试次数，放弃自动修复。")
            self._retry_count = 0
import os
import subprocess
from pathlib import Path
from loguru import logger

from ghostbot.agent.hook import AgentHook, AgentHookContext


class GitCheckpointHook(AgentHook):
    def __init__(self, workspace: Path):
        super().__init__()
        self.workspace = workspace
        self._modified_files = False
        self.active_git_cwd = None  # 动态记录当前大模型正在操作的真实 Git 根目录

    def _get_git_root(self, path: str) -> str | None:
        """向上寻找真正的 Git 根目录 (Toplevel)"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path, capture_output=True, text=True, encoding="utf-8", timeout=3
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    def _project_auto_commit(self, message: str, cwd: str) -> None:
        """针对用户项目的自动提交"""
        try:
            # 增加 timeout 防止某些 Git 钩子卡死
            subprocess.run(["git", "add", "."], cwd=cwd, check=True, timeout=10)
            subprocess.run(["git", "commit", "-m", message], cwd=cwd, check=True, timeout=10)
        except Exception as e:
            logger.warning(f"Project auto-commit failed in {cwd}: {e}")

    def _has_uncommitted_changes(self, cwd: str) -> bool:
        """检查是否有未提交的变动"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd, capture_output=True, text=True, encoding="utf-8", check=True, timeout=5
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _get_git_diff(self, cwd: str) -> str:
        """获取 Diff 用作大模型的输入"""
        try:
            subprocess.run(["git", "add", "."], cwd=cwd, timeout=5)
            result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=5
            )
            return result.stdout[:3000]
        except Exception:
            return ""

    async def _generate_commit_message(self, diff_text: str) -> str:
        # TODO: 未来可在此接入大模型 API 生成语义化的 Commit Message
        return f"GhostBot: AI auto-edit ({len(diff_text)} bytes)"

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        modify_tools = {"edit_file", "write_file", "replace_file", "delete_file"}

        # 1. 动态寻找目标目录：大模型到底要改哪个文件？
        target_file_path = None
        for tc in context.tool_calls:
            if tc.name in modify_tools:
                # 尝试从工具参数里提取路径
                path_arg = tc.parameters.get("path") or tc.parameters.get("file_path")
                if path_arg:
                    target_file_path = path_arg
                    break

        if not target_file_path:
            return

        # 2. 绝对路径解析，确保找对真正的项目目录
        p = Path(target_file_path)
        if not p.is_absolute():
            p = (self.workspace / p).resolve()

        # 3. 寻找该文件所属的 Git 根目录
        # 如果文件不存在，p.parent 一定存在，所以拿 p.parent 去寻根
        git_root = self._get_git_root(str(p.parent))

        # 如果找到了 Git 根目录，说明操作在合法的仓库内
        if git_root:
            # 🟢 关键修复：把真正的根目录赋给 active_git_cwd！
            self.active_git_cwd = git_root
            self._modified_files = True

            # 4. 在根目录里执行 Git 操作（检查 WIP）
            if self._has_uncommitted_changes(self.active_git_cwd):
                logger.info(f"📦 [GitHook] 发现 {self.active_git_cwd} 有未提交修改，正在创建 WIP 存档...")
                self._project_auto_commit("WIP: Save user changes before AI edit", self.active_git_cwd)

    async def after_iteration(self, context: AgentHookContext) -> None:
        # 如果没有标记修改，或者没有捕捉到 Git 根目录，直接跳过
        if not getattr(self, '_modified_files', False) or not self.active_git_cwd:
            return

        if not self._has_uncommitted_changes(self.active_git_cwd):
            self._modified_files = False
            return

        logger.info(f"🤖 [GitHook] AI 修改完毕，正在 {self.active_git_cwd} 生成 Commit...")
        try:
            diff_text = self._get_git_diff(self.active_git_cwd)
            commit_msg = await self._generate_commit_message(diff_text)
            self._project_auto_commit(commit_msg, self.active_git_cwd)
            logger.info(f"✅ [GitHook] 自动提交成功: {commit_msg}")
        except Exception as e:
            logger.error(f"[GitHook] AI 自动提交失败: {e}")
        finally:
            # 清理本轮状态，防止污染下一轮对话
            self._modified_files = False
            self.active_git_cwd = None
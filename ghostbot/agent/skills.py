"""Skills loader for agent capabilities."""

import json
import os
import re
import shutil
from pathlib import Path
from loguru import logger

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Opening ---, YAML body (group 1), closing --- on its own line; supports CRLF.
_STRIP_SKILL_FRONTMATTER = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SkillsLoader:
    """
    进化版 SkillsLoader：支持全量索引缓存与懒加载摘要。
    """

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None,
                 disabled_skills: set[str] | None = None):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self.disabled_skills = disabled_skills or set()

        # 索引文件路径
        self.index_path = self.workspace / "skills_index.json"
        self._index: dict[str, dict] = {}

        # 初始化时加载索引
        self.load_or_rebuild_index()

    # ==========================================
    # 1. 索引与缓存核心逻辑
    # ==========================================

    def load_or_rebuild_index(self):
        """启动时加载索引，如果不存在则重建。"""
        if self.index_path.exists():
            try:
                self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
                logger.debug("Successfully loaded skills index from disk.")
                return
            except Exception as e:
                logger.warning(f"Failed to load skills index: {e}. Rebuilding...")

        self.rebuild_index()

    def rebuild_index(self):
        """全量扫描目录并重建索引。"""
        logger.info("Rebuilding skills index...")
        new_index = {}

        # 扫描逻辑（先扫描内建，再扫描工作区以实现覆盖）
        all_entries = []
        if self.builtin_skills.exists():
            all_entries.extend(self._skill_entries_from_dir(self.builtin_skills, "builtin"))

        all_entries.extend(self._skill_entries_from_dir(self.workspace_skills, "workspace"))

        for entry in all_entries:
            name = entry["name"]
            if name in self.disabled_skills:
                continue

            # 提取元数据用于索引
            meta = self.get_skill_metadata_from_path(Path(entry["path"]))
            new_index[name] = {
                "name": name,
                "path": entry["path"],
                "source": entry["source"],
                "description": meta.get("description", name),
                "requires": self._parse_ghostbot_metadata(meta.get("metadata", "")).get("requires", {}),
                "always": self._parse_ghostbot_metadata(meta.get("metadata", "")).get("always", False)
            }

        self._index = new_index
        self._save_index_to_disk()

    def _save_index_to_disk(self):
        """将内存索引持久化。"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self._index, indent=2, ensure_ascii=False), encoding="utf-8")

    def sync_skill_change(self, name: str, action: str = "update"):
        """当用户通过 CLI 增加/删除技能时，增量更新索引。"""
        if action == "delete":
            self._index.pop(name, None)
        else:
            roots = [self.workspace_skills, self.builtin_skills]
            for root in roots:
                path = root / name / "SKILL.md"
                if path.exists():
                    meta = self.get_skill_metadata_from_path(path)
                    self._index[name] = {
                        "name": name,
                        "path": str(path),
                        "source": "workspace" if "workspace" in str(root) else "builtin",
                        "description": meta.get("description", name),
                        "requires": self._parse_ghostbot_metadata(meta.get("metadata", "")).get("requires", {}),
                        "always": self._parse_ghostbot_metadata(meta.get("metadata", "")).get("always", False)
                    }
                    break
        self._save_index_to_disk()

    # ==========================================
    # 2. 提示词摘要与读取
    # ==========================================

    def build_skills_summary(self) -> str:
        """优化后的摘要生成：仅提供菜单，提示模型自行去读取 location。"""
        if not self._index:
            return ""

        lines = ["<available_skills>", "  "]
        for name, info in self._index.items():
            available = self._check_requirements(info)
            status = "active" if available else "disabled_missing_deps"

            lines.extend([
                f'  <skill name="{name}" status="{status}">',
                f"    <description>{_escape_xml(info['description'])}</description>",
                f"    <location>{info['path']}</location>",
                f"  </skill>"
            ])
        lines.append("</available_skills>")
        return "\n".join(lines)

    def load_skill(self, name: str) -> str | None:
        """为了向下兼容和主动加载使用。"""
        roots = [self.workspace_skills]
        if self.builtin_skills:
            roots.append(self.builtin_skills)
        for root in roots:
            path = root / name / "SKILL.md"
            if path.exists():
                return path.read_text(encoding="utf-8")
        return None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        return "\n\n---\n\n".join(parts)

    def get_always_skills(self) -> list[str]:
        """直接从索引中快速获取常驻技能。"""
        return [name for name, info in self._index.items() if info.get("always") and self._check_requirements(info)]

    # ==========================================
    # 3. 底层辅助解析函数 (从原文件恢复)
    # ==========================================

    def _skill_entries_from_dir(self, base: Path, source: str, *, skip_names: set[str] | None = None) -> list[
        dict[str, str]]:
        if not base.exists():
            return []
        entries: list[dict[str, str]] = []
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = skill_dir.name
            if skip_names is not None and name in skip_names:
                continue
            entries.append({"name": name, "path": str(skill_file), "source": source})
        return entries

    def get_skill_metadata_from_path(self, path: Path) -> dict:
        """从特定路径读取元数据。"""
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        if not content or not content.startswith("---"):
            return {}
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if not match:
            return {}
        metadata: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"\'')
        return metadata

    def _parse_ghostbot_metadata(self, raw: str) -> dict:
        """Parse skill metadata JSON from frontmatter."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(data, dict):
            return {}
        payload = data.get("ghostbot", data.get("openclaw", {}))
        return payload if isinstance(payload, dict) else {}

    def _check_requirements(self, skill_meta: dict) -> bool:
        """Check if skill requirements are met (bins, env vars)."""
        requires = skill_meta.get("requires", {})
        required_bins = requires.get("bins", [])
        required_env_vars = requires.get("env", [])
        return all(shutil.which(cmd) for cmd in required_bins) and all(
            os.environ.get(var) for var in required_env_vars
        )

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if not content.startswith("---"):
            return content
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if match:
            return content[match.end():].strip()
        return content
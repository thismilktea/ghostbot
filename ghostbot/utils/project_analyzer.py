import time
import logging
import re
import threading
import json
from pathlib import Path
from typing import Dict, List, Tuple

# 🚀 引入 tree-sitter 核心引擎
try:
    from tree_sitter_languages import get_language, get_parser

    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

logger = logging.getLogger(__name__)

# 🔍 升级版 TS_QUERIES：不仅提取定义 (@def)，还提取调用/引用 (@ref)
TS_QUERIES = {
    "python": """
        (class_definition name: (identifier) @def.class)
        (function_definition name: (identifier) @def.method)
        (call function: (identifier) @ref.call)
        (call function: (attribute attribute: (identifier) @ref.call))
        (import_from_statement module_name: (dotted_name) @ref.import)
    """,
    "java": """
        (class_declaration name: (identifier) @def.class)
        (method_declaration name: (identifier) @def.method)
        (method_invocation name: (identifier) @ref.call)
    """,
    "go": """
        (type_declaration (type_spec name: (identifier) @def.class))
        (function_declaration name: (identifier) @def.method)
        (call_expression function: (identifier) @ref.call)
        (call_expression function: (selector_expression field: (identifier) @ref.call))
    """,
    "javascript": """
        (class_declaration id: (identifier) @def.class)
        (function_declaration id: (identifier) @def.method)
        (call_expression function: (identifier) @ref.call)
        (call_expression function: (member_expression property: (property_identifier) @ref.call))
    """,
    "typescript": """
        (class_declaration id: (identifier) @def.class)
        (function_declaration id: (identifier) @def.method)
        (method_definition name: (property_identifier) @def.method)
        (call_expression function: (identifier) @ref.call)
        (call_expression function: (member_expression property: (property_identifier) @ref.call))
    """,
}


class ProjectAnalyzer:
    """项目拓扑感知器：基于 Tree-sitter 的全局图谱 (Repo Map) 构建器"""

    _ANCHOR_WEIGHT = 100.0
    _OUTBOUND_WEIGHT = 50.0
    _INBOUND_WEIGHT = 30.0
    _HALF_LIFE_SECONDS = 1800.0

    def __init__(self, workspace: Path, memory_dir: Path, ignore_dirs: set = None):
        self.workspace = workspace
        self.projects_dir = memory_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', workspace.name)
        self.output_md = self.projects_dir / f"{safe_name}.md"
        self.cache_file = self.projects_dir / f"{safe_name}_cache.json"

        self.excluded = {'.git', '.idea', '.vscode', 'target', 'node_modules', 'venv', 'dist', 'build', '__pycache__'}
        if ignore_dirs:
            self.excluded.update(ignore_dirs)

        self.symbol_cache = self._load_cache()
        self._lock = threading.Lock()
        self._is_scanning = False
        self._parsers = {}

    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
        return {}

    def _save_cache(self):
        with self._lock:
            self.cache_file.write_text(json.dumps(self.symbol_cache, ensure_ascii=False), encoding="utf-8")

    def _extract_symbols_ts(self, content: str, lang_id: str) -> Dict[str, List[str]]:
        """使用 Tree-sitter 提取定义和引用"""
        if lang_id not in TS_QUERIES:
            return {"defs": [], "refs": []}

        try:
            if lang_id not in self._parsers:
                self._parsers[lang_id] = (get_language(lang_id), get_parser(lang_id))

            lang, parser = self._parsers[lang_id]
            tree = parser.parse(bytes(content, "utf8"))
            query = lang.query(TS_QUERIES[lang_id])
            captures = query.captures(tree.root_node)

            defs, refs = [], []
            for node, tag in captures:
                name = node.text.decode('utf8')
                if len(name) < 3 or len(name) > 50:
                    continue

                if tag.startswith("def"):
                    prefix = "C:" if "class" in tag else "M:"
                    defs.append(f"{prefix}{name}")
                elif tag.startswith("ref"):
                    refs.append(name)

            return {
                "defs": list(dict.fromkeys(defs))[:20],
                "refs": list(dict.fromkeys(refs))[:50],
            }
        except Exception as e:
            logger.debug(f"Tree-sitter 解析失败 ({lang_id}): {e}")
            return {"defs": [], "refs": []}

    def _extract_symbols_regex(self, content: str) -> Dict[str, List[str]]:
        """最后的兜底：正则表达式提取定义与调用"""
        defs, refs = [], []

        classes = re.findall(r'(?:class|interface|enum)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        for c in classes:
            defs.append(f"C:{c}")

        methods = re.findall(r'(?:def|func|void|public|private)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
        for m in methods:
            if m not in {'if', 'while', 'for', 'switch', 'main'}:
                defs.append(f"M:{m}")

        calls = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
        refs.extend(calls)

        return {
            "defs": list(dict.fromkeys(defs))[:20],
            "refs": list(dict.fromkeys(refs))[:50],
        }

    def _extract_file_data(self, file_path: Path) -> Dict[str, List[str]]:
        """符号提取调度器"""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            ext_map = {'.py': 'python', '.java': 'java', '.go': 'go', '.js': 'javascript', '.ts': 'typescript'}
            lang_id = ext_map.get(file_path.suffix)

            if HAS_TREE_SITTER and lang_id:
                data = self._extract_symbols_ts(content, lang_id)
                if data["defs"] or data["refs"]:
                    return data

            return self._extract_symbols_regex(content)
        except Exception:
            return {"defs": [], "refs": []}

    @staticmethod
    def _normalize_path_key(path: str) -> str:
        return path.replace('\\', '/').strip('/')

    def _normalize_active_files(self, active_files: dict[str, float] | None) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for path, touched_at in (active_files or {}).items():
            key = self._normalize_path_key(path)
            if key:
                normalized[key] = touched_at
        return normalized

    def _build_global_defs(self) -> dict[str, str]:
        global_defs: dict[str, str] = {}
        for file_path, data in self.symbol_cache.items():
            for definition in data.get('defs', []):
                global_defs[definition[2:]] = file_path
        return global_defs

    def _scan_directory(self, current_dir: Path, max_depth: int, current_depth: int = 0) -> bool:
        """纯净扫描，只写 Cache，不再直接组装 Markdown"""
        if current_depth > max_depth:
            return False

        any_changed = False
        try:
            items = list(current_dir.iterdir())
        except PermissionError:
            return False

        for f in items:
            if f.name in self.excluded:
                continue

            if f.is_dir():
                if self._scan_directory(f, max_depth, current_depth + 1):
                    any_changed = True
            elif f.is_file():
                if f.suffix not in {'.py', '.java', '.kt', '.go', '.ts', '.js'}:
                    continue

                path_key = self._normalize_path_key(f.relative_to(self.workspace).as_posix())
                mtime = f.stat().st_mtime

                if path_key not in self.symbol_cache or self.symbol_cache[path_key].get('mtime') < mtime:
                    data = self._extract_file_data(f)
                    self.symbol_cache[path_key] = {
                        "mtime": mtime,
                        "defs": data["defs"],
                        "refs": data["refs"],
                    }
                    any_changed = True

        return any_changed

    def build_ranked_file_scores(
        self,
        active_files: dict[str, float] | None,
        limit: int | None = None,
    ) -> list[tuple[str, float]]:
        if not self.symbol_cache:
            return []

        normalized_active_files = self._normalize_active_files(active_files)
        current_time = time.time()
        global_defs = self._build_global_defs()
        scores = {file_path: 0.0 for file_path in self.symbol_cache.keys()}

        if normalized_active_files:
            for file_path, data in self.symbol_cache.items():
                if file_path in normalized_active_files:
                    age_seconds = max(0.0, current_time - normalized_active_files[file_path])
                    decay = 0.5 ** (age_seconds / self._HALF_LIFE_SECONDS)
                    scores[file_path] += self._ANCHOR_WEIGHT * decay

                    for ref in data.get('refs', []):
                        target_file = global_defs.get(ref)
                        if target_file and target_file != file_path:
                            scores[target_file] += self._OUTBOUND_WEIGHT * decay
                else:
                    for ref in data.get('refs', []):
                        target_file = global_defs.get(ref)
                        if not target_file or target_file not in normalized_active_files:
                            continue
                        age_seconds = max(0.0, current_time - normalized_active_files[target_file])
                        decay = 0.5 ** (age_seconds / self._HALF_LIFE_SECONDS)
                        scores[file_path] += self._INBOUND_WEIGHT * decay

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if normalized_active_files:
            ranked = [(path, score) for path, score in ranked if score > 0]
        if limit is not None:
            ranked = ranked[:limit]
        return ranked

    def build_priority_bucket(
        self,
        active_files: dict[str, float] | None,
        limit: int | None = None,
    ) -> dict[str, float]:
        return dict(self.build_ranked_file_scores(active_files, limit=limit))

    def rank_paths(self, paths: list[str], active_files: dict[str, float] | None) -> list[str]:
        bucket = self.build_priority_bucket(active_files)
        normalized_paths = {path: self._normalize_path_key(path) for path in paths}
        return sorted(
            paths,
            key=lambda path: (-bucket.get(normalized_paths[path], 0.0), normalized_paths[path]),
        )

    def get_repo_map(self, active_files: dict[str, float] | None = None, max_output_files: int = 40) -> str:
        """
        🚀 核心魔法：动态计算 PageRank，生成上下文感知的 Repo Map
        :param active_files: 锚点文件（当前正在编辑或用户提到的文件列表）
        """
        if not self.symbol_cache:
            return "(Repo Map is empty)"

        normalized_active_files = self._normalize_active_files(active_files)
        ranked_files = self.build_ranked_file_scores(normalized_active_files)
        if normalized_active_files:
            top_ranked = ranked_files[:max_output_files]
            top_files = [path for path, _ in top_ranked]
            score_map = dict(top_ranked)
        else:
            top_files = sorted(self.symbol_cache.keys())[:max_output_files]
            score_map = {path: 0.0 for path in top_files}

        lines = []
        for file_path in top_files:
            score = score_map.get(file_path, 0.0)
            data = self.symbol_cache[file_path]
            defs = data.get('defs', [])

            tag = " 🌟(Active)" if score >= self._ANCHOR_WEIGHT else (" 🔗(Dep)" if score > 0 else "")
            lines.append(f"📄 {file_path}{tag}")

            for definition in defs:
                icon = "📦" if definition.startswith("C:") else "⚡"
                lines.append(f"   {icon} {definition[2:]}")

        total_ranked = len(ranked_files) if normalized_active_files else len(self.symbol_cache)
        if total_ranked > max_output_files:
            lines.append(f"\n... (已折叠 {total_ranked - max_output_files} 个不相关文件) ...")

        return "\n".join(lines)

    def _perform_scan(self, max_depth: int):
        logger.info(f"🔍 启动后台依赖图谱扫描: {self.workspace.name}")
        try:
            any_changed = self._scan_directory(self.workspace, max_depth)
            if any_changed:
                self._save_cache()

            repo_map_text = self.get_repo_map()
            header = (
                f"# 🏗️ Project Topology: {self.workspace.name}\n"
                f"> Project Path: {self.workspace.resolve()}\n"
                f"> Mode: Aider-style Repo Map\n"
                f"> Last Sync: {time.ctime()}\n\n"
            )
            self.output_md.write_text(header + "```text\n" + repo_map_text + "\n```", encoding="utf-8")

            logger.info(f"✅ 图谱已更新: {self.output_md.name}")
        except Exception as e:
            logger.error(f"❌ 扫描失败: {e}")

    def async_sync(self, max_depth: int = 4, force: bool = False):
        if self._is_scanning:
            return
        if not force and self.output_md.exists():
            if (time.time() - self.output_md.stat().st_mtime) < 300:
                return

        self._is_scanning = True
        threading.Thread(target=self._safe_perform_scan, args=(max_depth,), daemon=True).start()

    def _safe_perform_scan(self, max_depth: int):
        try:
            self._perform_scan(max_depth)
        finally:
            self._is_scanning = False


def sync_project_structure(target_workspace: Path, memory_dir: Path, max_depth: int = 4, force: bool = False,
                           ignore_dirs: set = None) -> Tuple[bool, Path]:
    analyzer = ProjectAnalyzer(target_workspace, memory_dir, ignore_dirs=ignore_dirs)
    if force:
        analyzer._safe_perform_scan(max_depth)
    else:
        analyzer.async_sync(max_depth, force)
    return True, analyzer.output_md

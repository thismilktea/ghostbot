"""Memory system: pure file I/O store, lightweight Consolidator, and Dream processor."""
from __future__ import annotations


import asyncio
import json
import re
import weakref
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from ghostbot.utils.prompt_templates import render_template
from ghostbot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain, strip_think

from ghostbot.agent.runner import AgentRunSpec, AgentRunner
from ghostbot.agent.tools.registry import ToolRegistry
from ghostbot.utils.gitstore import GitStore

if TYPE_CHECKING:
    from ghostbot.providers.base import LLMProvider
    from ghostbot.session.manager import Session, SessionManager

STOP_WORDS = {
    "的", "了", "和", "是", "就", "都", "而", "及", "与", "着",
    "这", "那", "我", "你", "他", "它", "啊", "嗯", "哦", "好的",
    "请", "帮", "一下", "怎么", "什么", "为什么",
    "def", "class", "import", "return", "if", "else" # 高频但无检索区分度的代码词
}


@dataclass(slots=True)
class MemoryRecord:
    cursor: str
    summary: str
    content: str
    scope: str = "global"
    record_type: str = "episode"


import sqlite3
import re
from pathlib import Path
from loguru import logger


# ghostbot/agent/memory.py

class QueryEnhancer:
    """LLM-backed query expansion for retrieval evaluation and memory search."""

    def __init__(self, provider: LLMProvider, model: str):
        self.provider = provider
        self.model = model

    async def expand(self, query: str) -> str:
        if not query.strip():
            return ""

        prompt = (
            "Expand this retrieval query with concise related keywords, identifiers, and likely synonyms. "
            "Return only the expanded query text, no explanation.\n\n"
            f"Query: {query}"
        )
        try:
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            expanded = (response.content if response else "").strip()
        except Exception as e:
            logger.warning(f"⚠️ 查询扩展失败: {e}")
            return query
        return f"{query} {expanded}" if expanded and query not in expanded else (expanded or query)


class ResultReranker:
    """精排器：利用小模型对粗召回结果进行语义校准"""

    def __init__(self, provider: LLMProvider, model: str):
        self.provider = provider
        self.model = model

    async def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        query: 用户原始提问
        candidates: [(cursor, text), ...] 粗召回的结果
        """
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates

        from ghostbot.utils.prompt_templates import render_template

        # 1. 构造候选列表字符串
        context_items = []
        for i, (cursor, text) in enumerate(candidates):
            context_items.append(f"[ID: {i}] (Cursor: {cursor}) 内容: {text}")

        candidates_str = "\n".join(context_items)

        # 2. 渲染你写好的精排提示词 (假设文件名为 agent/rerank.md)
        prompt = render_template("agent/rerank.md", query=query, candidates=candidates_str)

        try:
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )

            # 必须检查 content 是否存在
            if not response or not response.content:
                logger.warning("⚠️ 模型返回内容为空，维持粗召回排序")
                return candidates

            match = re.search(r'\[ID:\s*(\d+)\]', response.content)
            if match:
                best_index = int(match.group(1))
                if 0 <= best_index < len(candidates):
                    winner = candidates.pop(best_index)
                    # 💡 增加一个 Debug 标记，让我们知道精排器确实干活了
                    logger.success(f"🎯 精排器介入：选中了 ID {best_index} (Cursor {winner[0]})")
                    return [winner] + candidates

            return candidates
        except Exception as e:
            logger.warning(f"⚠️ 精排失败: {e}")
            return candidates

class HybridSearchEngine:
    def __init__(self, workspace: Path):
        self.db_path = workspace / "memory" / "index.sqlite"

        # 1. 连接数据库
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        # ⚠️ 注意：因为我们要换 Tokenizer，必须重建索引表
        # 这会清空你之前的搜索索引，但这很安全，因为历史记录在 history.jsonl 里
        self.conn.execute("DROP TABLE IF EXISTS history_search")

        # 2. 使用 'trigram' 分词器
        self.conn.execute("""
            CREATE VIRTUAL TABLE history_search USING fts5(
                cursor UNINDEXED, 
                content, 
                raw_content UNINDEXED,
                tokenize='trigram' 
            );
        """)

    def get_raw_scores(self, keywords: str, top_k: int = 30) -> dict[int, float]:
        """纯净版 FTS5 检索：只查 Cursor 和 BM25 分数，极其轻量 (耗时 < 5ms)"""
        if not keywords.strip():
            return {}

        fts_query = self._build_fts_query(keywords)  # 你的双轨制构造函数
        try:
            cursor = self.conn.execute("""
                SELECT cursor, abs(bm25(history_search)) as score 
                FROM history_search WHERE content MATCH ? LIMIT ?
            """, (fts_query, top_k))
            return {int(row[0]): float(row[1]) for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"FTS5 查询失败: {e}")
            return {}

    def fetch_and_decay(self, score_map: dict[int, float], temporal_intent: str, current_cursor: int) -> list[
        tuple[float, str, str]]:
        """组装相邻上下文，并动态应用时间衰减"""
        if not score_map:
            return []

        # 1. 收集目标 Cursor，并包含下一行 (相邻上下文扩展)
        target_cursors = set()
        for c in score_map.keys():
            target_cursors.add(c)
            target_cursors.add(c + 1)

        placeholders = ",".join("?" * len(target_cursors))
        content_cursor = self.conn.execute(
            f"SELECT cursor, raw_content FROM history_search WHERE cursor IN ({placeholders})",
            list(target_cursors)
        )
        content_map = {int(r[0]): r[1] for r in content_cursor.fetchall()}

        # 2. 根据意图动态调整时间惩罚系数
        decay_multiplier = 0.05  # default 'low'
        if temporal_intent == 'high':
            decay_multiplier = 0.2  # 陡峭衰减，旧记忆分数暴跌
        elif temporal_intent == 'none':
            decay_multiplier = 0.0  # 关闭时间衰减，纯靠 BM25 拼硬核匹配

        ranked_results = []
        for mem_cursor, bm25_score in score_map.items():
            curr_text = content_map.get(mem_cursor, "")
            next_text = content_map.get(mem_cursor + 1, "")
            combined_text = f"{curr_text}\n{next_text}".strip()

            distance = max(0, current_cursor - mem_cursor)
            decay_factor = 1.0 / (1.0 + distance * decay_multiplier)

            final_score = bm25_score * decay_factor
            ranked_results.append((final_score, str(mem_cursor), combined_text))

        # 降序排列
        ranked_results.sort(key=lambda x: x[0], reverse=True)
        return ranked_results

    def _build_fts_query(self, user_query: str, expanded_query: str = "") -> str:
        """
        权重双轨制查询：(原话高压匹配) OR (扩展词兜底匹配)
        注：你需要稍微修改调用逻辑，把原话和扩展后的话一起传进来。
        """

        def clean_and_quote(text):
            cleaned = re.sub(r'[^\w\s\u4e00-\u9fa5]', ' ', text).split()
            return [f'"{t}"' for t in cleaned if len(t) >= 2]

        user_tokens = clean_and_quote(user_query)
        expanded_tokens = clean_and_quote(expanded_query)

        if not user_tokens and not expanded_tokens:
            return ""

        queries = []

        # 策略 A：紧凑度优先（用 NEAR 锁定用户原话中的连续短语，极大提高精确度）
        if len(user_tokens) >= 2:
            # 要求前两三个核心词距离不超过 5
            near_clause = f"NEAR({' '.join(user_tokens[:3])}, 5)"
            queries.append(f"({near_clause})")

        # 策略 B：用户原话 ALL IN（要求至少同时包含用户的几个核心词）
        if user_tokens:
            queries.append(f"({' AND '.join(user_tokens[:3])})")

        # 策略 C：扩展词大杂烩（兜底召回）
        all_tokens = list(set(user_tokens + expanded_tokens))
        if all_tokens:
            queries.append(f"({' OR '.join(all_tokens)})")

        # 最终组装：只要命中高权重组，BM25 分数会成倍飙升
        return " OR ".join(queries)

    def index(self, cursor: int, text: str):
        """完全废弃 Jieba，存入原始文本"""
        if not text.strip():
            return

        # 直接存 raw text，让 sqlite 的 trigram 自动分词
        self.conn.execute(
            "INSERT INTO history_search (cursor, content, raw_content) VALUES (?, ?, ?)",
            (cursor, text, text)
        )
        self.conn.commit()

    def search(self, query: str, top_k: int = 15, current_cursor: int = 0):
        """兼容层：供 SearchMemoryTool 主动调用（默认采用 'low' 一般排查意图）"""
        if not query.strip():
            return []

        print(f"\n🔍 [主动搜索] 触发内部 FTS5 检索: '{query}'")

        raw_scores = self.get_raw_scores(query, top_k=top_k * 2)
        decayed_results = self.fetch_and_decay(raw_scores, temporal_intent="low", current_cursor=current_cursor)
        return [(item[1], item[2]) for item in decayed_results[:top_k]]

    @staticmethod
    def _infer_record_type(text: str) -> str:
        lowered = text.casefold()
        if any(token in lowered for token in ("must", "should", "不要", "必须", "约束", "constraint")):
            return "instruction"
        if any(token in lowered for token in ("prefer", "偏好", "喜欢", "习惯")):
            return "preference"
        if any(token in lowered for token in ("decision", "决定", "结论", "agreed")):
            return "decision"
        if any(token in lowered for token in ("project", "仓库", "模块", "评测", "release")):
            return "project_fact"
        return "episode"

    @staticmethod
    def _infer_scope(text: str) -> str:
        lowered = text.casefold()
        if any(token in lowered for token in ("branch", "分支", "pr", "pull request")):
            return "branch"
        if any(token in lowered for token in ("project", "仓库", "repo", "模块")):
            return "project"
        if any(token in lowered for token in ("task", "下一步", "todo", "checklist")):
            return "task-cluster"
        return "global"

    @staticmethod
    def _summarize_record(text: str, max_chars: int = 180) -> str:
        first_nonempty = next((line.strip("-*• \t") for line in text.splitlines() if line.strip()), "")
        return first_nonempty[:max_chars] if first_nonempty else text[:max_chars]

    def search_records(self, query: str, top_k: int = 5, current_cursor: int = 0) -> list[MemoryRecord]:
        return [
            MemoryRecord(
                cursor=cursor,
                summary=self._summarize_record(text),
                content=text,
                scope=self._infer_scope(text),
                record_type=self._infer_record_type(text),
            )
            for cursor, text in self.search(query, top_k=top_k, current_cursor=current_cursor)
        ]

# ---------------------------------------------------------------------------
# MemoryStore — pure file I/O layer
# ---------------------------------------------------------------------------

class MemoryStore:
    """Pure file I/O for memory files: MEMORY.md, history.jsonl, SOUL.md, USER.md."""

    _DEFAULT_MAX_HISTORY = 1000
    _LEGACY_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*")
    _LEGACY_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*")
    _LEGACY_RAW_MESSAGE_RE = re.compile(
        r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+[A-Z][A-Z0-9_]*(?:\s+\[tools:\s*[^\]]+\])?:"
    )

    def __init__(self, workspace: Path, max_history_entries: int = _DEFAULT_MAX_HISTORY):
        self.workspace = workspace
        self.max_history_entries = max_history_entries
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.legacy_history_file = self.memory_dir / "HISTORY.md"
        self.soul_file = workspace / "SOUL.md"
        self.user_file = workspace / "USER.md"
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"
        self._git = GitStore(workspace, tracked_files=[
            "SOUL.md", "USER.md", "memory/MEMORY.md",
        ])
        self._maybe_migrate_legacy_history()
        self.search_engine = HybridSearchEngine(workspace)
        self.swap_dir = ensure_dir(self.memory_dir / "paging_swap")

    def page_out(self, content: str) -> str:
        """将巨量文本写入 Swap 分区，返回 8 位 Hash 指针"""
        import hashlib
        pointer = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        file_path = self.swap_dir / f"{pointer}.txt"

        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")
        return pointer

    def page_in(self, pointer: str) -> str | None:
        """通过指针从 Swap 分区换入数据"""
        # 基础防注入校验
        if not pointer.isalnum() or len(pointer) > 64:
            return None

        file_path = self.swap_dir / f"{pointer}.txt"
        if file_path.exists():
            return self.read_file(file_path)
        return None

    def delete_page(self, pointer: str) -> bool:
        """从 Swap 分区删除指定指针的数据 (垃圾回收)"""
        # 基础防注入校验
        if not pointer.isalnum() or len(pointer) > 64:
            return False

        file_path = self.swap_dir / f"{pointer}.txt"
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"🗑️ [GC] Swap memory freed for pointer: {pointer}")
                return True
        except Exception as e:
            logger.warning(f"Failed to delete swap file {pointer}: {e}")
        return False
    @property
    def git(self) -> GitStore:
        return self._git

    # -- generic helpers -----------------------------------------------------

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _maybe_migrate_legacy_history(self) -> None:
        """One-time upgrade from legacy HISTORY.md to history.jsonl.

        The migration is best-effort and prioritizes preserving as much content
        as possible over perfect parsing.
        """
        if not self.legacy_history_file.exists():
            return
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return

        try:
            legacy_text = self.legacy_history_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            logger.exception("Failed to read legacy HISTORY.md for migration")
            return

        entries = self._parse_legacy_history(legacy_text)
        try:
            if entries:
                self._write_entries(entries)
                last_cursor = entries[-1]["cursor"]
                self._cursor_file.write_text(str(last_cursor), encoding="utf-8")
                # Default to "already processed" so upgrades do not replay the
                # user's entire historical archive into Dream on first start.
                self._dream_cursor_file.write_text(str(last_cursor), encoding="utf-8")

            backup_path = self._next_legacy_backup_path()
            self.legacy_history_file.replace(backup_path)
            logger.info(
                "Migrated legacy HISTORY.md to history.jsonl ({} entries)",
                len(entries),
            )
        except Exception:
            logger.exception("Failed to migrate legacy HISTORY.md")

    def _parse_legacy_history(self, text: str) -> list[dict[str, Any]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        fallback_timestamp = self._legacy_fallback_timestamp()
        entries: list[dict[str, Any]] = []
        chunks = self._split_legacy_history_chunks(normalized)

        for cursor, chunk in enumerate(chunks, start=1):
            timestamp = fallback_timestamp
            content = chunk
            match = self._LEGACY_TIMESTAMP_RE.match(chunk)
            if match:
                timestamp = match.group(1)
                remainder = chunk[match.end():].lstrip()
                if remainder:
                    content = remainder

            entries.append({
                "cursor": cursor,
                "timestamp": timestamp,
                "content": content,
            })
        return entries

    def _split_legacy_history_chunks(self, text: str) -> list[str]:
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        saw_blank_separator = False

        for line in lines:
            if saw_blank_separator and line.strip() and current:
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            if self._should_start_new_legacy_chunk(line, current):
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            current.append(line)
            saw_blank_separator = not line.strip()

        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _should_start_new_legacy_chunk(self, line: str, current: list[str]) -> bool:
        if not current:
            return False
        if not self._LEGACY_ENTRY_START_RE.match(line):
            return False
        if self._is_raw_legacy_chunk(current) and self._LEGACY_RAW_MESSAGE_RE.match(line):
            return False
        return True

    def _is_raw_legacy_chunk(self, lines: list[str]) -> bool:
        first_nonempty = next((line for line in lines if line.strip()), "")
        match = self._LEGACY_TIMESTAMP_RE.match(first_nonempty)
        if not match:
            return False
        return first_nonempty[match.end():].lstrip().startswith("[RAW]")

    def _legacy_fallback_timestamp(self) -> str:
        try:
            return datetime.fromtimestamp(
                self.legacy_history_file.stat().st_mtime,
            ).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _next_legacy_backup_path(self) -> Path:
        candidate = self.memory_dir / "HISTORY.md.bak"
        suffix = 2
        while candidate.exists():
            candidate = self.memory_dir / f"HISTORY.md.bak.{suffix}"
            suffix += 1
        return candidate

    # -- MEMORY.md (long-term facts) -----------------------------------------

    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    # -- SOUL.md -------------------------------------------------------------

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")

    # -- USER.md -------------------------------------------------------------

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    # -- context injection (used by context.py) ------------------------------

    def get_memory_context(self) -> str:
        long_term = self.read_memory()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # -- history.jsonl — append-only, JSONL format ---------------------------

    def append_history(self, entry: str) -> int:
        """Append *entry* to history.jsonl and return its auto-incrementing cursor."""
        cursor = self._next_cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        record = {"cursor": cursor, "timestamp": ts, "content": strip_think(entry.rstrip()) or entry.rstrip()}
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cursor_file.write_text(str(cursor), encoding="utf-8")
        self.search_engine.index(cursor, entry)
        return cursor

    def _next_cursor(self) -> int:
        """Read the current cursor counter and return next value."""
        if self._cursor_file.exists():
            try:
                return int(self._cursor_file.read_text(encoding="utf-8").strip()) + 1
            except (ValueError, OSError):
                pass
        # Fallback: read last line's cursor from the JSONL file.
        last = self._read_last_entry()
        if last:
            return last["cursor"] + 1
        return 1

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        """Return history entries with cursor > *since_cursor*."""
        return [e for e in self._read_entries() if e["cursor"] > since_cursor]

    def compact_history(self) -> None:
        """Drop oldest entries if the file exceeds *max_history_entries*."""
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
        kept = entries[-self.max_history_entries:]
        self._write_entries(kept)

    # -- JSONL helpers -------------------------------------------------------

    def _read_entries(self) -> list[dict[str, Any]]:
        """Read all entries from history.jsonl."""
        entries: list[dict[str, Any]] = []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        return entries

    def _read_last_entry(self) -> dict[str, Any] | None:
        """Read the last entry from the JSONL file efficiently."""
        try:
            with open(self.history_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size, 4096)
                f.seek(size - read_size)
                data = f.read().decode("utf-8")
                lines = [l for l in data.split("\n") if l.strip()]
                if not lines:
                    return None
                return json.loads(lines[-1])
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        """Overwrite history.jsonl with the given entries."""
        with open(self.history_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- dream cursor --------------------------------------------------------

    def get_last_dream_cursor(self) -> int:
        if self._dream_cursor_file.exists():
            try:
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self._dream_cursor_file.write_text(str(cursor), encoding="utf-8")

    # -- message formatting utility ------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    def raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to history.jsonl without LLM summarization."""
        self.append_history(
            f"[RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )



# ---------------------------------------------------------------------------
# Consolidator — lightweight token-budget triggered consolidation
# ---------------------------------------------------------------------------


class Consolidator:
    """Lightweight consolidation: summarizes evicted messages into history.jsonl."""

    _MAX_CONSOLIDATION_ROUNDS = 5
    _MAX_CHUNK_MESSAGES = 500  # hard cap per consolidation round

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift
    _CHECKPOINT_SECTIONS = (
        "Current goal:",
        "Confirmed constraints:",
        "Key files and symbols:",
        "Important errors and fixes:",
        "Open work / next steps:",
        "Recent quoted details:",
    )
    _FILE_SYMBOL_RE = re.compile(r"(?:[\\/][\w.-]+)+|\b[\w.-]+\.(?:py|ts|tsx|js|jsx|md|json|yml|yaml|toml|sh)\b")
    _ERROR_RE = re.compile(r"(?:error|exception|traceback|failed|failure|bug|fix|boom|错误|报错|失败|修复)", re.IGNORECASE)
    _NEXT_STEP_RE = re.compile(r"(?:\bnext\b|\btodo\b|\bblocker\b|\bremaining\b|\bfollow-up\b|下一步|待办|阻塞|后续)", re.IGNORECASE)
    _CONSTRAINT_RE = re.compile(r"(?:\bmust\b|\bshould\b|\bcannot\b|\bcan'?t\b|\bdon't\b|\bavoid\b|\bkeep\b|\bpreserve\b|\bwithout\b|必须|不要|不能|保留|保持)", re.IGNORECASE)
    _QUOTE_RE = re.compile(r'^["“].+["”]$')

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    def pick_consolidation_boundary(
            self,
            session: Session,
            tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None

        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            # 1. 先累加当前消息的 Token
            removed_tokens += estimate_message_tokens(message)

            # 2. 如果当前消息是一个 User 消息，记录为一个“合法的停靠站”
            if message.get("role") == "user":
                last_boundary = (idx, removed_tokens)

                # 3. 只要累加够了，立刻在当前的 User 边界返回
                if removed_tokens >= tokens_to_remove:
                    return last_boundary

        # 4. 如果循环结束都没达到目标，也要返回最后一个找到的 User 边界
        return last_boundary

    def _cap_consolidation_boundary(
        self,
        session: Session,
        end_idx: int,
    ) -> int | None:
        """Clamp the chunk size without breaking the user-turn boundary."""
        start = session.last_consolidated
        if end_idx - start <= self._MAX_CHUNK_MESSAGES:
            return end_idx

        capped_end = start + self._MAX_CHUNK_MESSAGES
        for idx in range(capped_end, start, -1):
            if session.messages[idx].get("role") == "user":
                return idx
        return None

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel = None
        chat_id = None
        metadata = getattr(session, "metadata", {}) or {}
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
            session_metadata=metadata,
            project=session,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    @classmethod
    def _extract_section_body(cls, text: str, section: str) -> str:
        start = text.find(section)
        if start < 0:
            return ""
        start += len(section)
        end = len(text)
        for other in cls._CHECKPOINT_SECTIONS:
            if other == section:
                continue
            idx = text.find(other, start)
            if idx >= 0:
                end = min(end, idx)
        return text[start:end].strip()

    @classmethod
    def _normalize_section_lines(cls, body: str) -> list[str]:
        if not body:
            return []
        normalized: list[str] = []
        for raw in body.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ")):
                normalized.append(f"- {line[2:].strip()}")
            else:
                normalized.append(f"- {line}")
        return normalized

    @classmethod
    def _dedupe_preserve_order(cls, lines: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for line in lines:
            key = line.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(line)
        return deduped

    @classmethod
    def _heuristic_checkpoint_sections(cls, text: str) -> dict[str, list[str]]:
        lines = [line.strip("-•* \t") for line in text.splitlines() if line.strip()]
        if not lines:
            return {section: [] for section in cls._CHECKPOINT_SECTIONS}

        sections = {section: [] for section in cls._CHECKPOINT_SECTIONS}
        fallback: list[str] = []
        for line in lines:
            if cls._QUOTE_RE.match(line):
                sections["Recent quoted details:"].append(f"- {line}")
                continue
            if cls._FILE_SYMBOL_RE.search(line):
                sections["Key files and symbols:"].append(f"- {line}")
            if cls._ERROR_RE.search(line):
                sections["Important errors and fixes:"].append(f"- {line}")
            if cls._NEXT_STEP_RE.search(line):
                sections["Open work / next steps:"].append(f"- {line}")
            if cls._CONSTRAINT_RE.search(line):
                sections["Confirmed constraints:"].append(f"- {line}")
            fallback.append(f"- {line}")

        sections["Current goal:"] = [fallback[0]] if fallback else []
        if len(fallback) > 1 and not sections["Open work / next steps:"]:
            sections["Open work / next steps:"] = [fallback[1]]
        if len(fallback) > 2 and not sections["Important errors and fixes:"]:
            sections["Important errors and fixes:"] = [fallback[2]]
        return {
            section: cls._dedupe_preserve_order(lines)
            for section, lines in sections.items()
        }

    @classmethod
    def _normalize_checkpoint_summary(cls, summary: str) -> str:
        text = strip_think(summary or "").strip()
        if not text:
            return "[no summary]"

        if all(section in text for section in cls._CHECKPOINT_SECTIONS):
            extracted = {
                section: cls._normalize_section_lines(cls._extract_section_body(text, section))
                for section in cls._CHECKPOINT_SECTIONS
            }
        else:
            extracted = cls._heuristic_checkpoint_sections(text)

        if not any(extracted.values()):
            fallback = [f"- {line.strip()}" for line in text.splitlines() if line.strip()]
            extracted["Current goal:"] = fallback[:1] or ["- [no summary]"]

        sections: list[str] = []
        for section in cls._CHECKPOINT_SECTIONS:
            body = extracted.get(section) or ["- (none)"]
            sections.append(f"{section}\n" + "\n".join(body))
        return "\n\n".join(sections)

    async def archive(self, messages: list[dict]) -> str | None:
        """Summarize messages via LLM and append to history.jsonl.

        Returns the summary text on success, None if nothing to archive.
        """
        if not messages:
            return None
        try:
            formatted = MemoryStore._format_messages(messages)
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template(
                            "agent/consolidator_archive.md",
                            strip=True,
                        ),
                    },
                    {"role": "user", "content": formatted},
                ],
                tools=None,
                tool_choice=None,
            )
            summary = self._normalize_checkpoint_summary(response.content or "[no summary]")
            self.store.append_history(summary)
            return summary
        except Exception:
            logger.warning("Consolidation LLM call failed, raw-dumping to history")
            self.store.raw_archive(messages)
            return None

    def _garbage_collect_swap(self, chunk: list[dict[str, Any]]) -> None:
        """
        扫描即将被淘汰的消息，提取其中的 Swap 指针，并将其物理文件删除。
        """
        import re

        # 定义我们在之前逻辑中写入指针时的固定格式，例如：
        # "[name 的长结果已被系统内核压缩至 Swap 分区 (Paged Out)。\n内存指针: a1b2c3d4...]"
        # 这里用正则宽松匹配 "内存指针: " 或 "Pointer: " 后面的 8 位字母数字
        pointer_pattern = re.compile(r'(?:内存指针|指针|Pointer):\s*([a-f0-9]{8})', re.IGNORECASE)

        deleted_count = 0
        for msg in chunk:
            content = msg.get("content")
            if not isinstance(content, str):
                continue

            # 扫描消息内容中的所有指针
            pointers = pointer_pattern.findall(content)
            for pointer in pointers:
                if self.store.delete_page(pointer):
                    deleted_count += 1

        if deleted_count > 0:
            logger.info(f"🧹 [Garbage Collection] Reclaimed {deleted_count} swap files during consolidation.")

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            try:
                estimated, source = self.estimate_session_prompt_tokens(session)
            except Exception:
                logger.exception("Token estimation failed for {}", session.key)
                estimated, source = 0, "error"
            if estimated <= 0:
                return
            if estimated < budget:
                unconsolidated_count = len(session.messages) - session.last_consolidated
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}, msgs={}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    unconsolidated_count,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    breakpoint()
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                end_idx = self._cap_consolidation_boundary(session, end_idx)
                if end_idx is None:
                    logger.debug(
                        "Token consolidation: no capped boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return
                self._garbage_collect_swap(chunk)  # <--- 新增的调用

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.archive(chunk):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                try:
                    estimated, source = self.estimate_session_prompt_tokens(session)
                except Exception:
                    logger.exception("Token estimation failed for {}", session.key)
                    estimated, source = 0, "error"
                if estimated <= 0:
                    return



# ---------------------------------------------------------------------------
# Dream — heavyweight cron-scheduled memory consolidation
# ---------------------------------------------------------------------------


class Dream:
    """Two-phase memory processor: analyze history.jsonl, then edit files via AgentRunner.

    Phase 1 produces an analysis summary (plain LLM call).
    Phase 2 delegates to AgentRunner with read_file / edit_file tools so the
    LLM can make targeted, incremental edits instead of replacing entire files.
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self._runner = AgentRunner(provider)
        self._tools = self._build_tools()

    # -- tool registry -------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Build a minimal tool registry for the Dream agent."""
        from ghostbot.agent.skills import BUILTIN_SKILLS_DIR
        from ghostbot.agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool

        tools = ToolRegistry()
        workspace = self.store.workspace
        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        tools.register(ReadFileTool(
            workspace=workspace,
            allowed_dir=workspace,
            extra_allowed_dirs=extra_read,
        ))
        tools.register(EditFileTool(workspace=workspace, allowed_dir=workspace))
        skills_dir = workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        tools.register(WriteFileTool(workspace=workspace, allowed_dir=skills_dir))
        return tools

    # -- skill listing --------------------------------------------------------

    def _list_existing_skills(self) -> list[str]:
        """List existing skills as 'name — description' for dedup context."""
        import re as _re

        from ghostbot.agent.skills import BUILTIN_SKILLS_DIR

        _DESC_RE = _re.compile(r"^description:\s*(.+)$", _re.MULTILINE | _re.IGNORECASE)
        entries: dict[str, str] = {}
        for base in (self.store.workspace / "skills", BUILTIN_SKILLS_DIR):
            if not base.exists():
                continue
            for d in base.iterdir():
                if not d.is_dir():
                    continue
                skill_md = d / "SKILL.md"
                if not skill_md.exists():
                    continue
                if d.name in entries and base == BUILTIN_SKILLS_DIR:
                    continue
                content = skill_md.read_text(encoding="utf-8")[:500]
                m = _DESC_RE.search(content)
                desc = m.group(1).strip() if m else "(no description)"
                entries[d.name] = desc
        return [f"{name} — {desc}" for name, desc in sorted(entries.items())]

    # -- main entry ----------------------------------------------------------

    async def run(self) -> bool:
        """Process unprocessed history entries. Returns True if work was done."""
        from ghostbot.agent.skills import BUILTIN_SKILLS_DIR

        last_cursor = self.store.get_last_dream_cursor()
        entries = self.store.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return False

        batch = entries[: self.max_batch_size]
        logger.info(
            "Dream: processing {} entries (cursor {}→{}), batch={}",
            len(entries), last_cursor, batch[-1]["cursor"], len(batch),
        )

        history_text = "\n".join(
            f"[{e['timestamp']}] {e['content']}" for e in batch
        )

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_memory = self.store.read_memory() or "(empty)"
        current_soul = self.store.read_soul() or "(empty)"
        current_user = self.store.read_user() or "(empty)"

        file_context = (
            f"## Current Date\n{current_date}\n\n"
            f"## Current MEMORY.md ({len(current_memory)} chars)\n{current_memory}\n\n"
            f"## Current SOUL.md ({len(current_soul)} chars)\n{current_soul}\n\n"
            f"## Current USER.md ({len(current_user)} chars)\n{current_user}"
        )

        phase1_prompt = (
            f"## Conversation History\n{history_text}\n\n{file_context}"
        )

        try:
            phase1_response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template("agent/dream_phase1.md", strip=True),
                    },
                    {"role": "user", "content": phase1_prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            analysis = phase1_response.content or ""
            logger.debug("Dream Phase 1 analysis ({} chars): {}", len(analysis), analysis[:500])
        except Exception:
            logger.exception("Dream Phase 1 failed")
            return False

        existing_skills = self._list_existing_skills()
        skills_section = ""
        if existing_skills:
            skills_section = (
                "\n\n## Existing Skills\n"
                + "\n".join(f"- {s}" for s in existing_skills)
            )
        phase2_prompt = f"## Analysis Result\n{analysis}\n\n{file_context}{skills_section}"

        tools = self._tools
        skill_creator_path = BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md"
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": render_template(
                    "agent/dream_phase2.md",
                    strip=True,
                    skill_creator_path=str(skill_creator_path),
                ),
            },
            {"role": "user", "content": phase2_prompt},
        ]

        try:
            result = await self._runner.run(AgentRunSpec(
                initial_messages=messages,
                tools=tools,
                model=self.model,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                fail_on_tool_error=False,
            ))
            logger.debug(
                "Dream Phase 2 complete: stop_reason={}, tool_events={}",
                result.stop_reason, len(result.tool_events),
            )
            for ev in (result.tool_events or []):
                logger.info(
                    "Dream tool_event: name={}, status={}, detail={}",
                    ev.get("name"),
                    ev.get("status"),
                    ev.get("detail", "")[:200],
                )
        except Exception:
            logger.exception("Dream Phase 2 failed")
            result = None

        changelog: list[str] = []
        if result and result.tool_events:
            for event in result.tool_events:
                if event["status"] == "ok":
                    changelog.append(f"{event['name']}: {event['detail']}")

        new_cursor = batch[-1]["cursor"]
        self.store.set_last_dream_cursor(new_cursor)
        self.store.compact_history()

        if result and result.stop_reason == "completed":
            logger.info(
                "Dream done: {} change(s), cursor advanced to {}",
                len(changelog), new_cursor,
            )
        else:
            reason = result.stop_reason if result else "exception"
            logger.warning(
                "Dream incomplete ({}): cursor advanced to {}",
                reason, new_cursor,
            )

        if changelog and self.store.git.is_initialized():
            ts = batch[-1]["timestamp"]
            sha = self.store.git.auto_commit(f"dream: {ts}, {len(changelog)} change(s)")
            if sha:
                logger.info("Dream commit: {}", sha)

        return True


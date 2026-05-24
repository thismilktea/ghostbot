import re
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from pathlib import Path
from loguru import logger
try:
    from tree_sitter_languages import get_language, get_parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

class BaseTruncator(ABC):
    """内容截断策略的抽象基类"""

    def __init__(self, max_output_lines: int = 100):
        self.max_output_lines = max_output_lines

    @abstractmethod
    def truncate(self, content: str, filename: str, page_out_fn: Callable[[str], str]) -> str:
        """
        执行截断逻辑
        :param content: 原始超长文本
        :param filename: 文件名（用于辅助嗅探）
        :param page_out_fn: Swap 换出函数，接收字符串，返回 8位 Hash 指针
        """
        pass


class LogTruncator(BaseTruncator):
    """日志文件专用截断器：正则提纯 ERROR/Traceback"""

    def __init__(self, max_output_lines: int = 100):
        super().__init__(max_output_lines)
        # 匹配异常信息的正则
        self.error_pattern = re.compile(
            r'(?i)(error|exception|traceback|fatal|fail|warn|oom|caused by|at\s+[a-zA-Z0-9_$.]+)')

    def truncate(self, content: str, filename: str, page_out_fn: Callable[[str], str]) -> str:
        lines = content.splitlines()
        total_lines = len(lines)

        if total_lines <= self.max_output_lines:
            return content

        # 头部和尾部保留行数
        head_keep = 20
        tail_keep = 50

        head_lines = lines[:head_keep]
        tail_lines = lines[-tail_keep:]
        middle_lines = lines[head_keep:-tail_keep]

        # 对中间层进行异常提纯
        extracted_middle = []
        hidden_lines_chunk = []

        for i, line in enumerate(middle_lines):
            if self.error_pattern.search(line):
                # 发现错误行，先结算之前被隐藏的无用日志
                if hidden_lines_chunk:
                    chunk_text = "\n".join(hidden_lines_chunk)
                    ptr = page_out_fn(chunk_text)
                    extracted_middle.append(
                        f"\n... [隐藏了 {len(hidden_lines_chunk)} 行 INFO 日志 (指针: {ptr})] ...\n")
                    hidden_lines_chunk.clear()
                # 保留错误行
                extracted_middle.append(line)
            else:
                hidden_lines_chunk.append(line)

        # 结算最后一块隐藏区域
        if hidden_lines_chunk:
            chunk_text = "\n".join(hidden_lines_chunk)
            ptr = page_out_fn(chunk_text)
            extracted_middle.append(f"\n... [隐藏了 {len(hidden_lines_chunk)} 行 INFO 日志 (指针: {ptr})] ...\n")

        # 拼接最终骨架
        result = (
                f"[系统提示: {filename} 过长，已触发智能日志提纯，无关日志已存入 Swap]\n"
                + "\n".join(head_lines) + "\n"
                + "\n".join(extracted_middle) + "\n"
                + "\n".join(tail_lines)
        )
        return result


class DefaultTruncator(BaseTruncator):
    """默认兜底截断器：双端保留，中间换出"""

    def truncate(self, content: str, filename: str, page_out_fn: Callable[[str], str]) -> str:
        lines = content.splitlines()
        if len(lines) <= self.max_output_lines:
            return content

        half = self.max_output_lines // 2
        head = "\n".join(lines[:half])
        tail = "\n".join(lines[-half:])
        middle = "\n".join(lines[half:-half])

        ptr = page_out_fn(middle)
        return f"{head}\n\n... [中间 {len(lines) - self.max_output_lines} 行已换出至 Swap (指针: {ptr})] ...\n\n{tail}"


class CodeTruncator(BaseTruncator):
    """代码截断器：基于 AST 的智能方法体折叠"""

    # 针对不同语言，专门提取“方法体/块 (body/block)”的查询语句
    BODY_QUERIES = {
        ".py": "(function_definition body: (block) @body)",
        ".java": "(method_declaration body: (block) @body) (constructor_declaration body: (block) @body)",
        ".js": "(function_declaration body: (statement_block) @body) (method_definition body: (statement_block) @body)",
        ".ts": "(function_declaration body: (statement_block) @body) (method_definition body: (statement_block) @body)",
        ".go": "(function_declaration body: (block) @body) (method_declaration body: (block) @body)",
    }

    LANG_MAP = {
        ".py": "python", ".java": "java", ".js": "javascript",
        ".ts": "typescript", ".go": "go"
    }

    def truncate(self, content: str, filename: str, page_out_fn: Callable[[str], str]) -> str:
        lines = content.splitlines()
        # 如果本来就不长，直接放行
        if len(lines) <= self.max_output_lines:
            return content

        ext = Path(filename).suffix.lower()

        # 如果环境没有 Tree-sitter，或者是不支持的语言后缀，直接走默认硬截断兜底
        if not HAS_TREE_SITTER or ext not in self.BODY_QUERIES:
            return DefaultTruncator(self.max_output_lines).truncate(content, filename, page_out_fn)

        try:
            lang_id = self.LANG_MAP[ext]
            lang = get_language(lang_id)
            parser = get_parser(lang_id)

            # ⚠️ Tree-sitter 需要操作字节
            content_bytes = content.encode('utf8')
            tree = parser.parse(content_bytes)

            query = lang.query(self.BODY_QUERIES[ext])
            captures = query.captures(tree.root_node)

            # 1. 提取所有 body 节点的起始和结束字节位置
            spans = []
            for node, tag in captures:
                # 过滤掉太短的方法体（比如只写了 pass 或者只有 2 行代码的，没必要折叠，直接让大模型看）
                if node.end_byte - node.start_byte > 150:
                    spans.append((node.start_byte, node.end_byte))

            # 2. 💣 核心技巧：必须从后往前替换！
            # 如果从前往后替换，一旦前面插入了文本，后面的 start_byte 和 end_byte 偏移量就全错位了。
            spans.sort(key=lambda x: x[0], reverse=True)

            result_bytes = content_bytes
            for start, end in spans:
                # 切割出被折叠的方法体源码
                body_bytes = result_bytes[start:end]
                # 存入 Swap 获得指针
                ptr = page_out_fn(body_bytes.decode('utf8', errors='ignore'))

                # 构造骨架提示词 (替换原来的方法体)
                replacement = f"\n    ... [内部逻辑已折叠至 Swap, 阅读细节请调用 read_page(pointer: \"{ptr}\")] ...\n".encode('utf8')

                # 字节拼接：保留头部 -> 塞入补丁 -> 保留尾部
                result_bytes = result_bytes[:start] + replacement + result_bytes[end:]

            final_text = result_bytes.decode('utf8', errors='ignore')

            # 3. 终极防爆破兜底
            # 如果即使折叠了所有方法，这个文件还是长得离谱（比如有几千行的顶级全局变量数组），再用硬截断切一刀
            if len(final_text.splitlines()) > self.max_output_lines * 2:
                return DefaultTruncator(self.max_output_lines).truncate(final_text, filename, page_out_fn)

            return f"[系统提示: {filename} 过长，已基于 AST 提取代码骨架，方法细节已存入 Swap]\n" + final_text

        except Exception as e:
            logger.warning(f"AST 折叠失败 ({filename}): {e}，回退到默认截断")
            return DefaultTruncator(self.max_output_lines).truncate(content, filename, page_out_fn)


class ContentRouter:
    """内容嗅探与策略分发中心"""

    def __init__(self):
        # 初始化策略实例
        self.log_strategy = LogTruncator()
        self.code_strategy = CodeTruncator()
        self.default_strategy = DefaultTruncator()

        # 后缀映射路由表
        self.ext_routes = {
            '.log': self.log_strategy,
            '.out': self.log_strategy,
            '.err': self.log_strategy,
            '.py': self.code_strategy,
            '.java': self.code_strategy,
            '.go': self.code_strategy,
            '.js': self.code_strategy,
            '.ts': self.code_strategy,
        }

    def route_and_truncate(self, content: str, filename: str, page_out_fn: Callable[[str], str]) -> str:
        """入口函数：嗅探 -> 分发 -> 执行"""

        # 1. 强制类型嗅探 (根据文件名后缀)
        ext = Path(filename).suffix.lower()
        strategy = self.ext_routes.get(ext)

        # 2. 启发式内容嗅探 (针对没有后缀的终端输出，如 shell tool 的 stdout)
        if not strategy:
            # 如果内容中包含大量的日期时间戳或 INFO/DEBUG 字样，推测为日志
            if re.search(r'(?i)(\[\d{4}-\d{2}-\d{2}|\bINFO\b|\bDEBUG\b)', content[:1000]):
                strategy = self.log_strategy
                logger.debug(f"内容嗅探: {filename} 判定为 Log 类型")

        # 3. 兜底策略
        if not strategy:
            strategy = self.default_strategy

        # 执行截断
        return strategy.truncate(content, filename, page_out_fn)
import re
from pathlib import Path


def extract_error_context(error_text: str, workspace_path: str | Path, window: int = 10) -> str:
    """提取报错信息周围的源码上下文"""
    java_pattern = r"\((.*?\.java):(\d+)\)"
    python_pattern = r"File [\"'](.*?\.[a-zA-Z]+)[\"'], line (\d+)"

    match = re.search(java_pattern, error_text) or re.search(python_pattern, error_text)
    if not match:
        return ""

    filename = match.group(1).split('/')[-1].split('\\')[-1]
    line_num = int(match.group(2))

    target_file = None
    # 在当前工作区递归寻找该文件
    for path in Path(workspace_path).rglob(filename):
        target_file = path
        break

    if not target_file:
        return ""

    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start = max(0, line_num - window - 1)
        end = min(len(lines), line_num + window)
        snippet = "".join(lines[start:end])

        return f"\n\n[System Context] Source code around {target_file.name}:{line_num}:\n```\n{snippet}\n```"
    except Exception:
        return ""
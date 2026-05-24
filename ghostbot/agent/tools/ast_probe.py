import os
import re


def extract_code_context(error_msg: str, cwd: str) -> str:
    """
    动态上下文增强引擎 (RAG):
    从报错信息中嗅探文件路径和行号，并提取报错行前后的源码。
    """
    # 策略 1: 匹配 Python 报错栈 -> File "C:\path\main.py", line 15
    py_pattern = r'[Ff]ile "([^"]+)", line (\d+)'

    # 策略 2: 匹配 Java/Maven/通用 报错 -> C:\path\main.java:[15,20]
    general_pattern = r'([a-zA-Z]:[\\/][^\s]+?\.(?:java|py|js|ts|cpp|cs|go))[^\d]+(\d+)'

    # 尝试提取所有匹配项
    matches = re.findall(py_pattern, error_msg)
    if not matches:
        matches = re.findall(general_pattern, error_msg)

    if not matches:
        return ""  # 没嗅探到文件路径，安全退出

    # 逆向遍历匹配项：通常报错栈最底部的（即最后一个匹配的）是用户自己的业务代码，而不是系统底层库
    for filepath, linenum in reversed(matches):
        try:
            linenum = int(linenum)

            # 处理相对路径
            if not os.path.isabs(filepath) and cwd:
                filepath = os.path.join(cwd, filepath)

            # 如果文件确实存在于硬盘上
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 提取报错行上下各 10 行的上下文
                start = max(0, linenum - 10)
                end = min(len(lines), linenum + 10)

                # 给报错的那一行打上高亮标记 '>>>'
                snippet_lines = []
                for i in range(start, end):
                    prefix = ">>> " if i == linenum - 1 else "    "
                    snippet_lines.append(f"{i + 1:4d} {prefix}{lines[i]}")

                snippet = "".join(snippet_lines)

                # 组装上下文提示词
                return (
                    f"\n\n=========================================\n"
                    f"🕵️ 幽灵探针已激活：提取到引发报错的源码上下文\n"
                    f"📁 文件: {filepath} (第 {linenum} 行)\n"
                    f"-----------------------------------------\n"
                    f"```\n{snippet}```\n"
                    f"=========================================\n"
                )
        except Exception as e:
            print(f"提取源码上下文失败: {e}")
            continue  # 尝试下一个匹配项

    return ""
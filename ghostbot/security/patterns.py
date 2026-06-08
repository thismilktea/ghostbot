"""Security pattern scanner — regex-based instant detection of known-dangerous code patterns."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SecurityFinding:
    rule_name: str
    severity: str
    message: str
    file_path: str
    matched_text: str = ""
    line_number: int = 0


def _ext_filter(*exts: str):
    def _check(path: str) -> bool:
        return any(path.endswith(e) for e in exts)
    return _check


_PYTHON_EXTS = _ext_filter(".py", ".pyw")
_JS_TS_EXTS = _ext_filter(".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs")
_HTML_EXTS = _ext_filter(".html", ".htm", ".vue", ".svelte")
_YAML_EXTS = _ext_filter(".yml", ".yaml")
_ALL = lambda path: True


SECURITY_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "python_eval",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["eval("],
        "regex": r"\beval\s*\([^)]*\b(?:input|request|user|arg|param|data|query)",
        "severity": "high",
        "message": "eval() with user-controlled input is a code injection risk.",
    },
    {
        "name": "python_exec",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["exec("],
        "regex": r"\bexec\s*\([^)]*\b(?:input|request|user|arg|param|data|query)",
        "severity": "high",
        "message": "exec() with user-controlled input is a code injection risk.",
    },
    {
        "name": "python_pickle_load",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["pickle.load", "pickle.loads"],
        "regex": r"\bpickle\.loads?\s*\(",
        "severity": "high",
        "message": "pickle.load on untrusted data enables arbitrary code execution.",
    },
    {
        "name": "python_yaml_load",
        "path_filter": _YAML_EXTS.__func__ if hasattr(_YAML_EXTS, '__func__') else _PYTHON_EXTS,
        "substrings": ["yaml.load("],
        "regex": r"\byaml\.load\s*\([^)]*(?!\bLoader\s*=\s*yaml\.SafeLoader)",
        "severity": "medium",
        "message": "yaml.load() without SafeLoader can execute arbitrary Python objects.",
    },
    {
        "name": "python_os_system",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["os.system("],
        "regex": r"\bos\.system\s*\(",
        "severity": "medium",
        "message": "os.system() is vulnerable to shell injection. Use subprocess with shell=False.",
    },
    {
        "name": "python_subprocess_shell",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["shell=True"],
        "regex": r"\bsubprocess\.\w+\([^)]*shell\s*=\s*True",
        "severity": "medium",
        "message": "subprocess with shell=True is vulnerable to shell injection.",
    },
    {
        "name": "python_sql_format",
        "path_filter": _PYTHON_EXTS,
        "substrings": [".format(", "f\"", "f'"],
        "regex": r"(?:execute|cursor\.execute|\.query)\s*\(\s*(?:f['\"]|['\"].*\.format\()",
        "severity": "high",
        "message": "SQL query built with string formatting is vulnerable to SQL injection. Use parameterized queries.",
    },
    {
        "name": "python_torch_load_unsafe",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["torch.load("],
        "regex": r"\btorch\.load\s*\([^)]*weights_only\s*=\s*False",
        "severity": "medium",
        "message": "torch.load with weights_only=False can execute arbitrary code from model files.",
    },
    {
        "name": "python_hardcoded_secret",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["password", "secret", "api_key", "token"],
        "regex": r"(?:password|secret|api_key|token|private_key)\s*=\s*['\"][^'\"]{8,}['\"]",
        "severity": "high",
        "message": "Hardcoded secret detected. Use environment variables or a secrets manager.",
    },
    {
        "name": "python_tempfile_insecure",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["mktemp("],
        "regex": r"\btempfile\.mktemp\s*\(",
        "severity": "low",
        "message": "tempfile.mktemp() is insecure (race condition). Use mkstemp() or NamedTemporaryFile().",
    },
    {
        "name": "js_innerhtml",
        "path_filter": _JS_TS_EXTS,
        "substrings": ["innerHTML"],
        "regex": r"\.innerHTML\s*=\s*(?!['\"]\s*['\"]\s*;)",
        "severity": "high",
        "message": "Setting innerHTML with dynamic content is an XSS risk. Use textContent or a sanitizer.",
    },
    {
        "name": "js_eval",
        "path_filter": _JS_TS_EXTS,
        "substrings": ["eval("],
        "regex": r"\beval\s*\(",
        "severity": "high",
        "message": "eval() can execute arbitrary code. Avoid it entirely or use safer alternatives.",
    },
    {
        "name": "js_dangerously_set_html",
        "path_filter": _JS_TS_EXTS,
        "substrings": ["dangerouslySetInnerHTML"],
        "regex": r"dangerouslySetInnerHTML\s*=\s*\{",
        "severity": "medium",
        "message": "dangerouslySetInnerHTML bypasses React's XSS protection. Ensure content is sanitized.",
    },
    {
        "name": "js_document_write",
        "path_filter": _JS_TS_EXTS,
        "substrings": ["document.write("],
        "regex": r"\bdocument\.write\s*\(",
        "severity": "medium",
        "message": "document.write() with dynamic content is an XSS risk.",
    },
    {
        "name": "js_sql_template",
        "path_filter": _JS_TS_EXTS,
        "substrings": ["query(", "execute("],
        "regex": r"(?:query|execute)\s*\(\s*`[^`]*\$\{",
        "severity": "high",
        "message": "SQL query built with template literals is vulnerable to injection. Use parameterized queries.",
    },
    {
        "name": "js_hardcoded_secret",
        "path_filter": _JS_TS_EXTS,
        "substrings": ["password", "secret", "apiKey", "token"],
        "regex": r"(?:password|secret|apiKey|token|privateKey)\s*[:=]\s*['\"][^'\"]{8,}['\"]",
        "severity": "high",
        "message": "Hardcoded secret detected. Use environment variables.",
    },
    {
        "name": "html_script_src_http",
        "path_filter": _HTML_EXTS,
        "substrings": ["<script", "http://"],
        "regex": r"<script[^>]+src\s*=\s*['\"]http://",
        "severity": "medium",
        "message": "Loading scripts over HTTP is vulnerable to MITM attacks. Use HTTPS.",
    },
    {
        "name": "html_onclick_inline",
        "path_filter": _HTML_EXTS,
        "substrings": ["onclick=", "onerror=", "onload="],
        "regex": r"\bon(?:click|error|load|mouseover)\s*=\s*['\"](?!return\s+false)",
        "severity": "low",
        "message": "Inline event handlers can be an XSS vector. Use addEventListener instead.",
    },
    {
        "name": "generic_private_key",
        "path_filter": _ALL,
        "substrings": ["-----BEGIN", "PRIVATE KEY"],
        "regex": r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
        "severity": "high",
        "message": "Private key embedded in source code. Store keys in secure key management.",
    },
    {
        "name": "generic_aws_key",
        "path_filter": _ALL,
        "substrings": ["AKIA"],
        "regex": r"AKIA[0-9A-Z]{16}",
        "severity": "high",
        "message": "AWS access key ID detected in source code.",
    },
    {
        "name": "generic_jwt_hardcoded",
        "path_filter": _ALL,
        "substrings": ["eyJ"],
        "regex": r"eyJ[A-Za-z0-9_-]{20,}\.eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}",
        "severity": "medium",
        "message": "Hardcoded JWT token detected. Tokens should be dynamically generated.",
    },
    {
        "name": "python_path_traversal",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["../", "..\\"],
        "regex": r"(?:open|Path)\s*\([^)]*(?:\.\./|\.\.\\)",
        "severity": "medium",
        "message": "Potential path traversal. Validate and sanitize file paths.",
    },
    {
        "name": "python_ssrf_requests",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["requests.get(", "requests.post(", "httpx.get("],
        "regex": r"(?:requests|httpx)\.(?:get|post|put|delete|patch)\s*\([^)]*\b(?:url|user|input|param|arg)",
        "severity": "medium",
        "message": "HTTP request with user-controlled URL is an SSRF risk. Validate against an allowlist.",
    },
    {
        "name": "python_debug_breakpoint",
        "path_filter": _PYTHON_EXTS,
        "substrings": ["breakpoint(", "pdb.set_trace("],
        "regex": r"\b(?:breakpoint|pdb\.set_trace)\s*\(\s*\)",
        "severity": "low",
        "message": "Debug breakpoint left in code. Remove before production.",
    },
    {
        "name": "generic_todo_security",
        "path_filter": _ALL,
        "substrings": ["TODO", "FIXME", "HACK"],
        "regex": r"(?:TODO|FIXME|HACK)\s*:?\s*.*(?:security|auth|permission|credential|secret|vuln)",
        "severity": "low",
        "message": "Security-related TODO/FIXME found. Address before shipping.",
    },
]


def check_patterns(file_path: str, content: str) -> list[SecurityFinding]:
    """Scan content against all security patterns and return findings."""
    findings: list[SecurityFinding] = []
    for pattern in SECURITY_PATTERNS:
        path_filter = pattern.get("path_filter", _ALL)
        if not path_filter(file_path):
            continue

        substrings = pattern.get("substrings", [])
        if substrings and not any(sub in content for sub in substrings):
            continue

        regex = pattern.get("regex")
        if not regex:
            continue

        for match in re.finditer(regex, content):
            line_number = content[:match.start()].count("\n") + 1
            findings.append(SecurityFinding(
                rule_name=pattern["name"],
                severity=pattern.get("severity", "medium"),
                message=pattern.get("message", "Security pattern matched."),
                file_path=file_path,
                matched_text=match.group(0)[:120],
                line_number=line_number,
            ))
            break

    return findings

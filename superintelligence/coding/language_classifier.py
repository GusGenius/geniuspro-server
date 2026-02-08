"""
GeniusPro Superintelligence — Coding language classifier

Heuristically detects the dominant programming language from recent messages.
This is used only to improve CODE routing defaults for Cursor-focused workflows.
"""

from __future__ import annotations

import re
from enum import Enum


class CodingLanguage(str, Enum):
    UNKNOWN = "unknown"
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SQL = "sql"
    HTML = "html"
    CSS = "css"


_FENCE_LANG_RE = re.compile(r"```(\w+)?\n", re.IGNORECASE)

_EXT_LANGUAGE: list[tuple[re.Pattern[str], CodingLanguage]] = [
    (re.compile(r"\.(py|pyi)\b", re.IGNORECASE), CodingLanguage.PYTHON),
    (re.compile(r"\.(ts|tsx)\b", re.IGNORECASE), CodingLanguage.TYPESCRIPT),
    (re.compile(r"\.(js|jsx|mjs|cjs)\b", re.IGNORECASE), CodingLanguage.JAVASCRIPT),
    (re.compile(r"\.(java)\b", re.IGNORECASE), CodingLanguage.JAVA),
    (re.compile(r"\.(cs)\b", re.IGNORECASE), CodingLanguage.CSHARP),
    (re.compile(r"\.(cpp|cc|cxx|hpp|hh|hxx)\b", re.IGNORECASE), CodingLanguage.CPP),
    (re.compile(r"\.(go)\b", re.IGNORECASE), CodingLanguage.GO),
    (re.compile(r"\.(rs)\b", re.IGNORECASE), CodingLanguage.RUST),
    (re.compile(r"\.(php)\b", re.IGNORECASE), CodingLanguage.PHP),
    (re.compile(r"\.(rb)\b", re.IGNORECASE), CodingLanguage.RUBY),
    (re.compile(r"\.(swift)\b", re.IGNORECASE), CodingLanguage.SWIFT),
    (re.compile(r"\.(kt|kts)\b", re.IGNORECASE), CodingLanguage.KOTLIN),
    (re.compile(r"\.(sql)\b", re.IGNORECASE), CodingLanguage.SQL),
    (re.compile(r"\.(html|htm)\b", re.IGNORECASE), CodingLanguage.HTML),
    (re.compile(r"\.(css)\b", re.IGNORECASE), CodingLanguage.CSS),
]

_KEYWORD_HINTS: list[tuple[re.Pattern[str], CodingLanguage]] = [
    (re.compile(r"\bfrom\s+\w+\s+import\b|\bdef\s+\w+\(", re.IGNORECASE), CodingLanguage.PYTHON),
    (re.compile(r"\binterface\s+\w+|\btype\s+\w+\s*=", re.IGNORECASE), CodingLanguage.TYPESCRIPT),
    (re.compile(r"\bconsole\.log\b|\bmodule\.exports\b|\brequire\(", re.IGNORECASE), CodingLanguage.JAVASCRIPT),
    (re.compile(r"\bpublic\s+class\b|\bSystem\.out\.println\b", re.IGNORECASE), CodingLanguage.JAVA),
    (re.compile(r"\busing\s+System\b|\bnamespace\s+\w+", re.IGNORECASE), CodingLanguage.CSHARP),
    (re.compile(r"#include\s*<|\bstd::\w+", re.IGNORECASE), CodingLanguage.CPP),
    (re.compile(r"\bpackage\s+main\b|\bfmt\.Print", re.IGNORECASE), CodingLanguage.GO),
    (re.compile(r"\bfn\s+\w+\b|\bprintln!\b", re.IGNORECASE), CodingLanguage.RUST),
    (re.compile(r"\bSELECT\b.+\bFROM\b|\bCREATE\s+TABLE\b", re.IGNORECASE | re.DOTALL), CodingLanguage.SQL),
]

_FENCE_LANG_MAP: dict[str, CodingLanguage] = {
    "py": CodingLanguage.PYTHON,
    "python": CodingLanguage.PYTHON,
    "ts": CodingLanguage.TYPESCRIPT,
    "tsx": CodingLanguage.TYPESCRIPT,
    "typescript": CodingLanguage.TYPESCRIPT,
    "js": CodingLanguage.JAVASCRIPT,
    "jsx": CodingLanguage.JAVASCRIPT,
    "javascript": CodingLanguage.JAVASCRIPT,
    "java": CodingLanguage.JAVA,
    "csharp": CodingLanguage.CSHARP,
    "cs": CodingLanguage.CSHARP,
    "cpp": CodingLanguage.CPP,
    "cxx": CodingLanguage.CPP,
    "cc": CodingLanguage.CPP,
    "go": CodingLanguage.GO,
    "golang": CodingLanguage.GO,
    "rust": CodingLanguage.RUST,
    "rs": CodingLanguage.RUST,
    "php": CodingLanguage.PHP,
    "ruby": CodingLanguage.RUBY,
    "rb": CodingLanguage.RUBY,
    "swift": CodingLanguage.SWIFT,
    "kotlin": CodingLanguage.KOTLIN,
    "kt": CodingLanguage.KOTLIN,
    "sql": CodingLanguage.SQL,
    "html": CodingLanguage.HTML,
    "css": CodingLanguage.CSS,
}


def detect_coding_language(messages: list[dict]) -> CodingLanguage:
    """
    Detect a likely coding language from the last few messages.
    Returns UNKNOWN if no strong signal is present.
    """
    if not messages:
        return CodingLanguage.UNKNOWN

    # Focus on last 3 messages (user + assistant context).
    parts: list[str] = []
    for msg in messages[-3:]:
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            parts.append(content)
    text = "\n".join(parts)
    if not text.strip():
        return CodingLanguage.UNKNOWN

    # Code fence language tag wins when present.
    for m in _FENCE_LANG_RE.finditer(text):
        raw = (m.group(1) or "").strip().lower()
        if not raw:
            continue
        lang = _FENCE_LANG_MAP.get(raw)
        if lang:
            return lang

    # File extensions are a strong signal.
    for pattern, lang in _EXT_LANGUAGE:
        if pattern.search(text):
            return lang

    # Keyword hints as a fallback.
    for pattern, lang in _KEYWORD_HINTS:
        if pattern.search(text):
            return lang

    return CodingLanguage.UNKNOWN


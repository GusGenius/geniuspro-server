"""
GeniusPro Superintelligence — Coding summarization helpers
"""

from __future__ import annotations

from typing import Literal

from superintelligence.coding.prompts import (
    SUMMARIZE_DIFF_SYSTEM_PROMPT,
    SUMMARIZE_FILE_OR_FOLDER_SYSTEM_PROMPT,
    SUMMARIZE_SELECTION_SYSTEM_PROMPT,
    SUMMARIZE_SESSION_SYSTEM_PROMPT,
)

SummarizeType = Literal["diff", "session", "file_or_folder", "selection"]


def get_summarize_system_prompt(summary_type: SummarizeType) -> str:
    if summary_type == "diff":
        return SUMMARIZE_DIFF_SYSTEM_PROMPT
    if summary_type == "session":
        return SUMMARIZE_SESSION_SYSTEM_PROMPT
    if summary_type == "file_or_folder":
        return SUMMARIZE_FILE_OR_FOLDER_SYSTEM_PROMPT
    if summary_type == "selection":
        return SUMMARIZE_SELECTION_SYSTEM_PROMPT
    # Literal should prevent this at type-check time, but keep runtime safety.
    raise ValueError(f"Unknown summary type: {summary_type}")


def build_summarize_messages(
    *,
    summary_type: SummarizeType,
    content: str,
    instructions: str | None = None,
) -> list[dict]:
    system_prompt = get_summarize_system_prompt(summary_type)
    user_parts = [content]
    if instructions:
        user_parts.append("")
        user_parts.append("Extra instructions:")
        user_parts.append(instructions)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts).strip()},
    ]


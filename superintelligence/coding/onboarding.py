"""
GeniusPro Superintelligence — Coding onboarding

Provides the first-turn "Quick start" question for Cursor coding sessions.
"""

from __future__ import annotations

import time


def onboarding_question_text() -> str:
    return (
        "Quick start — what are we doing today?\n\n"
        "a) Start a brand-new project (scaffold from scratch)\n"
        "b) Work on an existing project (fix/feature/refactor in a repo)\n"
        "c) Planning only (architecture/task breakdown — no code changes yet)\n"
        "d) Something else (tell me in one sentence)\n\n"
        "My recommendation: **b)** if you already have a repo/workspace; otherwise **a)**."
    )


def build_onboarding_response(*, model_name: str) -> dict:
    start = time.time()
    return {
        "id": f"si-{int(start * 1000)}",
        "object": "chat.completion",
        "created": int(start),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": onboarding_question_text()},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


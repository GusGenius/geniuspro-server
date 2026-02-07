"""
GeniusPro Superintelligence v1 — Task Classifier

Classifies incoming prompts into task types so the router can pick
the best expert model. Uses a hybrid approach:

1. Rule-based fast path (keywords, patterns) — instant, no model needed
2. Embedding + kNN (future) — learned from benchmark + live data

The rule-based classifier ships with v1. kNN layer is added as the
agent swarm collects performance data.
"""

import logging
import re
from enum import Enum

logger = logging.getLogger("superintelligence.classifier")


class TaskType(Enum):
    """Task categories for routing."""

    CODE = "code"
    MATH = "math"
    REASONING = "reasoning"
    CREATIVE = "creative"
    GENERAL = "general"
    MULTIMODAL = "multimodal"


# ── Keyword patterns for rule-based classification ───────────────────────────

CODE_PATTERNS = [
    r"\b(def|class|function|import|const|let|var|return|async|await)\b",
    r"\b(python|javascript|typescript|rust|java|golang|cpp|sql|html|css)\b",
    r"\b(bug|debug|error|fix|refactor|compile|runtime|syntax)\b",
    r"\b(api|endpoint|http|rest|graphql|websocket|cors)\b",
    r"\b(git|commit|branch|merge|pull request|deploy)\b",
    r"\b(docker|kubernetes|nginx|server|database|redis)\b",
    r"\b(react|next\.?js|fastapi|django|flask|express)\b",
    r"```",  # Code blocks
    r"[{}\[\]();].*[{}\[\]();]",  # Code-like punctuation clusters
]

MATH_PATTERNS = [
    r"\b(solve|equation|integral|derivative|matrix|vector|proof)\b",
    r"\b(calculate|compute|evaluate|simplify|factor)\b",
    r"\b(theorem|lemma|corollary|conjecture|axiom)\b",
    r"\b(algebra|calculus|geometry|topology|statistics)\b",
    r"\b(probability|distribution|regression|hypothesis)\b",
    r"[∫∑∏√∞∂∇≈≠≤≥]",  # Math symbols
    r"\$.*\\(frac|sqrt|int|sum|prod|lim)\b",  # LaTeX math
]

REASONING_PATTERNS = [
    r"\b(analyze|evaluate|compare|contrast|assess)\b",
    r"\b(explain why|reason|logic|argument|evidence)\b",
    r"\b(pros and cons|trade.?offs?|advantages|disadvantages)\b",
    r"\b(strategy|plan|approach|framework|methodology)\b",
    r"\b(research|study|paper|journal|findings)\b",
    r"\b(implications?|consequences?|impact|effect)\b",
]

CREATIVE_PATTERNS = [
    r"\b(write|story|poem|essay|blog|article|narrative)\b",
    r"\b(creative|imagine|fiction|character|plot|scene)\b",
    r"\b(rewrite|rephrase|paraphrase|tone|style)\b",
    r"\b(slogan|tagline|headline|copy|marketing)\b",
    r"\b(song|lyrics|script|dialogue|monologue)\b",
]


def _count_matches(text: str, patterns: list[str]) -> int:
    """Count how many patterns match in the text."""
    count = 0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            count += 1
    return count


class TaskClassifier:
    """
    Classifies prompts into task types using rule-based pattern matching.

    Scoring: count keyword matches per category, highest score wins.
    Ties go to GENERAL. Minimum threshold of 2 matches to classify
    as anything other than GENERAL.
    """

    MIN_THRESHOLD = 2  # Minimum matches to classify as non-general

    def classify(self, prompt: str) -> TaskType:
        """Classify a prompt into a TaskType."""
        scores = {
            TaskType.CODE: _count_matches(prompt, CODE_PATTERNS),
            TaskType.MATH: _count_matches(prompt, MATH_PATTERNS),
            TaskType.REASONING: _count_matches(prompt, REASONING_PATTERNS),
            TaskType.CREATIVE: _count_matches(prompt, CREATIVE_PATTERNS),
        }

        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]

        if best_score < self.MIN_THRESHOLD:
            logger.debug("Classification: GENERAL (best was %s=%d, below threshold %d)",
                         best_type.value, best_score, self.MIN_THRESHOLD)
            return TaskType.GENERAL

        logger.info("Classification: %s (score=%d) — scores: %s",
                     best_type.value, best_score,
                     ", ".join(f"{k.value}={v}" for k, v in scores.items() if v > 0))
        return best_type

    def classify_messages(self, messages: list[dict]) -> TaskType:
        """Classify based on the full message history (focuses on last user message)."""
        # Concatenate the last few messages for context
        text_parts: list[str] = []
        for msg in messages[-3:]:  # Last 3 messages for context
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)

        combined = " ".join(text_parts)
        return self.classify(combined)

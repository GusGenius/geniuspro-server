"""
GeniusPro Superintelligence v1 — Router

The core routing engine. Takes a classified task type and selects
the best provider based on benchmark-seeded performance data.

Each provider has a score per task type, seeded from public benchmarks.
The router picks the provider with the highest score for the given task.
Includes fallback chain if the primary provider fails.
"""

import logging
import random
from typing import Optional

from superintelligence.providers.base import BaseProvider
from superintelligence.routing.classifier import TaskClassifier, TaskType
from superintelligence.coding.language_classifier import CodingLanguage

logger = logging.getLogger("superintelligence.router")


# ── Benchmark-seeded scores (0-100) per provider per task ────────────────────
# Sourced from independent benchmarks as of Feb 7, 2026:
#   CODE:      Terminal-Bench 2.0, SWE-Bench Pro, HumanEval
#   MATH:      AIME 2025, MATH benchmark
#   REASONING: Humanity's Last Exam (HLE), GPQA Diamond
#   CREATIVE:  LMArena (human preference), Artificial Analysis Index
#   GENERAL:   Artificial Analysis Intelligence Index v4.0
#   MULTIMODAL: Provider capability (text+image+video+audio support)
#
# The Agent Swarm updates these scores continuously from live data.

DEFAULT_SCORES: dict[str, dict[TaskType, float]] = {
    "openai": {
        # GPT-5.3-Codex: Terminal-Bench 2.0 #1 (75.1%), GPQA 92.4%
        # AIME 100% (GPT-5.2), AI Index 51 (#2)
        TaskType.CODE: 98,       # Terminal-Bench 2.0 #1
        TaskType.MATH: 95,       # AIME 2025 100% (GPT-5.2)
        TaskType.REASONING: 92,  # GPQA Diamond 92.4%
        TaskType.CREATIVE: 88,   # Strong but not #1
        TaskType.GENERAL: 93,    # AI Index 51 (#2)
        TaskType.MULTIMODAL: 85, # Supports image input
    },
    "anthropic": {
        # Opus 4.6: Terminal-Bench 2.0 #2 (69.9%), SWE-Bench Pro #1 (Sonnet 4.5 82%)
        # AI Index 53 (#1), HLE strong, GPQA ~85%
        TaskType.CODE: 94,       # Terminal-Bench 69.9% (#2), SWE-Bench family leads
        TaskType.MATH: 88,       # AIME 96.3% (Haiku 4.5), solid
        TaskType.REASONING: 95,  # AI Index #1 (53), HLE top tier
        TaskType.CREATIVE: 96,   # LMArena #1 for writing quality
        TaskType.GENERAL: 96,    # AI Index 53 — highest overall
        TaskType.MULTIMODAL: 78, # Image input only, no video/audio
    },
    "deepseek": {
        # V3.2: HumanEval 96.1%, IMO gold, IOI gold, AIME near-perfect
        # Very cheap ($0.28/1M), weak on creative/multimodal
        TaskType.CODE: 93,       # HumanEval 96.1%, IOI gold
        TaskType.MATH: 98,       # IMO gold medal, AIME near-perfect
        TaskType.REASONING: 88,  # HLE 71.5% (DeepSeek R1)
        TaskType.CREATIVE: 72,   # Not optimized for creative
        TaskType.GENERAL: 82,    # Good but not frontier on general
        TaskType.MULTIMODAL: 40, # Text only
    },
    "openrouter-deepseek": {
        # DeepSeek V3.2 via OpenRouter: Same capabilities as direct DeepSeek
        # HumanEval 96.1%, IMO gold, IOI gold, AIME near-perfect
        TaskType.CODE: 93,       # HumanEval 96.1%, IOI gold
        TaskType.MATH: 98,       # IMO gold medal, AIME near-perfect
        TaskType.REASONING: 88,  # HLE 71.5% (DeepSeek R1)
        TaskType.CREATIVE: 72,   # Not optimized for creative
        TaskType.GENERAL: 82,    # Good but not frontier on general
        TaskType.MULTIMODAL: 40, # Text only
    },
    "openrouter": {
        # Kimi K2 Thinking: AIME 99.1%, HLE 44.9% (#2)
        # Strong math/reasoning, available via OpenRouter
        TaskType.CODE: 82,       # Decent but not frontier
        TaskType.MATH: 92,       # Kimi K2 AIME 99.1%
        TaskType.REASONING: 86,  # HLE 44.9% (#2 on some benchmarks)
        TaskType.CREATIVE: 78,   # Average
        TaskType.GENERAL: 84,    # Good all-rounder
        TaskType.MULTIMODAL: 65, # Limited
    },
    "google": {
        # Gemini 3 Pro: HLE 74.2% (#1), GPQA 91.9% (#1), AIME 100%
        # Best multimodal (text+image+video+audio+PDF), 1M context
        TaskType.CODE: 86,       # Good but not SWE-Bench leader
        TaskType.MATH: 96,       # AIME 2025 100%, GPQA 91.9%
        TaskType.REASONING: 97,  # HLE 74.2% #1, GPQA Diamond #1
        TaskType.CREATIVE: 84,   # Decent
        TaskType.GENERAL: 92,    # Strong across the board
        TaskType.MULTIMODAL: 99, # Best — text/image/video/audio/PDF
    },
    "mistral": {
        # Large 3: 675B MoE, Apache 2.0, strong multilingual
        # Budget-friendly, open-source, MMLU 81%
        TaskType.CODE: 80,       # Codestral decent but not frontier
        TaskType.MATH: 76,       # Below frontier models
        TaskType.REASONING: 78,  # Below frontier models
        TaskType.CREATIVE: 82,   # Good multilingual creative
        TaskType.GENERAL: 81,    # MMLU 81%
        TaskType.MULTIMODAL: 55, # Limited multimodal support
    },
    "openrouter-mistral": {
        # Mistral Large 3 via OpenRouter: Same capabilities as direct Mistral
        # 675B MoE, Apache 2.0, strong multilingual
        TaskType.CODE: 80,       # Codestral decent but not frontier
        TaskType.MATH: 76,       # Below frontier models
        TaskType.REASONING: 78,  # Below frontier models
        TaskType.CREATIVE: 82,   # Good multilingual creative
        TaskType.GENERAL: 81,    # MMLU 81%
        TaskType.MULTIMODAL: 55, # Limited multimodal support
    },
    "openrouter-google": {
        # Gemini 2.5 Pro via OpenRouter: HLE 74.2% (#1), GPQA 91.9% (#1)
        # Best multimodal (text+image+video+audio+PDF), 1M context
        TaskType.CODE: 86,       # Good but not SWE-Bench leader
        TaskType.MATH: 96,       # AIME 2025 100%, GPQA 91.9%
        TaskType.REASONING: 97,  # HLE 74.2% #1, GPQA Diamond #1
        TaskType.CREATIVE: 84,   # Decent
        TaskType.GENERAL: 92,    # Strong across the board
        TaskType.MULTIMODAL: 99, # Best — text/image/video/audio/PDF
    },
}

# ── Cursor coding: language-aware routing (heuristic) ─────────────────────────
#
# Public benchmarks are still mostly Python-heavy (SWE-bench, LiveCodeBench) and
# aggregate across languages (Aider Polyglot). Multi-LCB (ICLR 2026) shows
# substantial disparities across languages, so we encode a small heuristic
# profile for Cursor coding flows. These values can be tuned over time.
#
# If a provider isn't listed for a language, we fall back to DEFAULT_SCORES[CODE].
CODE_LANGUAGE_PROVIDER_SCORES: dict[CodingLanguage, dict[str, float]] = {
    CodingLanguage.PYTHON: {
        "anthropic": 99.0,
        "openai": 97.0,
        "deepseek": 95.0,
        "openrouter-deepseek": 95.0,
        "google": 93.0,
        "openrouter-google": 93.0,
    },
    CodingLanguage.TYPESCRIPT: {
        "openai": 99.0,
        "anthropic": 97.0,
        "google": 95.0,
        "openrouter-google": 95.0,
        "deepseek": 92.0,
        "openrouter-deepseek": 92.0,
    },
    CodingLanguage.JAVASCRIPT: {
        "openai": 99.0,
        "anthropic": 97.0,
        "google": 95.0,
        "openrouter-google": 95.0,
        "deepseek": 92.0,
        "openrouter-deepseek": 92.0,
    },
    CodingLanguage.RUST: {
        "deepseek": 97.0,
        "openrouter-deepseek": 97.0,
        "openai": 96.0,
        "anthropic": 92.0,
        "google": 90.0,
        "openrouter-google": 90.0,
    },
    CodingLanguage.CPP: {
        "deepseek": 97.0,
        "openrouter-deepseek": 97.0,
        "openai": 96.0,
        "anthropic": 92.0,
        "google": 90.0,
        "openrouter-google": 90.0,
    },
    CodingLanguage.GO: {
        "deepseek": 96.0,
        "openrouter-deepseek": 96.0,
        "openai": 95.0,
        "anthropic": 92.0,
        "google": 91.0,
        "openrouter-google": 91.0,
    },
    CodingLanguage.JAVA: {
        "openai": 97.0,
        "google": 96.0,
        "openrouter-google": 96.0,
        "anthropic": 94.0,
        "deepseek": 92.0,
        "openrouter-deepseek": 92.0,
    },
    CodingLanguage.CSHARP: {
        "openai": 97.0,
        "google": 95.0,
        "openrouter-google": 95.0,
        "anthropic": 94.0,
        "deepseek": 92.0,
        "openrouter-deepseek": 92.0,
    },
    CodingLanguage.SQL: {
        "openai": 97.0,
        "deepseek": 96.0,
        "openrouter-deepseek": 96.0,
        "anthropic": 94.0,
        "google": 92.0,
        "openrouter-google": 92.0,
    },
}


class SuperintelligenceRouter:
    """
    Routes requests to the best provider based on task classification.

    Flow:
    1. Classify the prompt into a TaskType
    2. Look up scores for all available providers
    3. Pick the highest-scoring provider
    4. If it fails, fall back to the next best
    """

    def __init__(self) -> None:
        self.classifier = TaskClassifier()
        self.providers: dict[str, BaseProvider] = {}
        self.scores: dict[str, dict[TaskType, float]] = dict(DEFAULT_SCORES)
        self._provider_health: dict[str, bool] = {}

    def register_provider(self, provider: BaseProvider) -> None:
        """Register a provider with the router."""
        self.providers[provider.name] = provider
        self._provider_health[provider.name] = True
        logger.info("Registered provider: %s (model=%s)", provider.name, provider.default_model)

    def classify(self, messages: list[dict]) -> TaskType:
        """Classify the task type from messages."""
        return self.classifier.classify_messages(messages)

    def select_provider(
        self,
        task_type: TaskType,
        exclude: Optional[set[str]] = None,
        code_language: Optional[CodingLanguage] = None,
    ) -> Optional[BaseProvider]:
        """
        Select the best provider for a task type.

        Args:
            task_type: The classified task type
            exclude: Provider names to skip (for fallback chain)

        Returns:
            The best available provider, or None if all excluded/unavailable
        """
        exclude = exclude or set()
        candidates: list[tuple[str, float]] = []

        for name, provider in self.providers.items():
            if name in exclude:
                continue
            if not self._provider_health.get(name, True):
                continue

            score = self.scores.get(name, {}).get(task_type, 50.0)
            if task_type == TaskType.CODE and code_language and code_language != CodingLanguage.UNKNOWN:
                lang_scores = CODE_LANGUAGE_PROVIDER_SCORES.get(code_language)
                if lang_scores and name in lang_scores:
                    score = lang_scores[name]
            # Add small random jitter to break ties
            candidates.append((name, score + random.uniform(0, 0.1)))

        if not candidates:
            logger.warning("No candidates for task_type=%s (exclude=%s)", task_type.value, exclude)
            return None

        # Sort by score descending, pick the best
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_name = candidates[0][0]
        top3 = ", ".join(f"{n}={s:.0f}" for n, s in candidates[:3])
        logger.info("Selected provider %s for %s (top: %s)", best_name, task_type.value, top3)
        return self.providers[best_name]

    def get_fallback_chain(
        self,
        task_type: TaskType,
        max_fallbacks: int = 3,
        code_language: Optional[CodingLanguage] = None,
    ) -> list[BaseProvider]:
        """
        Get an ordered list of providers for a task type (best first).
        Used for automatic fallback if primary fails.
        """
        chain: list[BaseProvider] = []
        exclude: set[str] = set()

        for _ in range(max_fallbacks):
            provider = self.select_provider(task_type, exclude=exclude, code_language=code_language)
            if provider is None:
                break
            chain.append(provider)
            exclude.add(provider.name)

        return chain

    def mark_unhealthy(self, provider_name: str) -> None:
        """Mark a provider as unhealthy (called by Latency Agent)."""
        self._provider_health[provider_name] = False
        logger.warning("Provider marked unhealthy: %s", provider_name)

    def mark_healthy(self, provider_name: str) -> None:
        """Mark a provider as healthy again."""
        self._provider_health[provider_name] = True
        logger.info("Provider marked healthy: %s", provider_name)

    def update_score(
        self, provider_name: str, task_type: TaskType, new_score: float
    ) -> None:
        """Update a provider's score for a task type (called by Agent Swarm)."""
        if provider_name not in self.scores:
            self.scores[provider_name] = {}
        self.scores[provider_name][task_type] = new_score

    def get_routing_info(self, task_type: TaskType) -> dict:
        """Get debug info about routing decision (internal use only)."""
        ranked: list[dict] = []
        for name in self.providers:
            score = self.scores.get(name, {}).get(task_type, 50.0)
            healthy = self._provider_health.get(name, True)
            ranked.append({"provider": name, "score": score, "healthy": healthy})

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return {
            "task_type": task_type.value,
            "candidates": ranked,
        }

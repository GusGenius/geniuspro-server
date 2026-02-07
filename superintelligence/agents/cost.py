"""
GeniusPro Superintelligence v1 — Cost Agent

Tracks provider pricing and updates the routing cost matrix.
Runs every hour to check for pricing changes.

In v1, pricing is hardcoded from known provider rates.
Future: scrape provider pricing pages or APIs automatically.
"""

import asyncio
from typing import Optional

from superintelligence.routing.classifier import TaskType


# ── Known pricing as of Feb 2026 (per 1M tokens) ────────────────────────────

PROVIDER_PRICING: dict[str, dict[str, float]] = {
    "openai": {
        "input": 5.00,     # GPT-5.3-Codex (restricted, estimated)
        "output": 25.00,
    },
    "anthropic": {
        "input": 5.00,     # Claude Opus 4.6
        "output": 25.00,
    },
    "deepseek": {
        "input": 0.28,     # DeepSeek V3.2
        "output": 1.10,
    },
    "openrouter": {
        "input": 2.00,     # Varies by model, average estimate
        "output": 8.00,
    },
    "google": {
        "input": 1.25,     # Gemini 3 Pro
        "output": 5.00,
    },
    "mistral": {
        "input": 2.00,     # Mistral Large 3
        "output": 6.00,
    },
}

# Our pricing to clients
SUPERINTELLIGENCE_PRICING = {
    "input": 4.00,   # $4 per 1M input tokens
    "output": 24.00,  # $24 per 1M output tokens
}


class CostAgent:
    """Monitors and reports provider costs."""

    def __init__(self, interval: int = 3600) -> None:
        self.interval = interval  # 1 hour default
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.pricing = dict(PROVIDER_PRICING)

    async def start(self) -> None:
        """Start the cost monitoring loop."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        print("  Cost Agent started")

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Main loop — checks pricing periodically."""
        while self._running:
            try:
                await self._update_pricing()
            except Exception as e:
                print(f"  Cost Agent error: {e}")
            await asyncio.sleep(self.interval)

    async def _update_pricing(self) -> None:
        """
        Update provider pricing.

        v1: Uses hardcoded values.
        Future: Scrape provider pricing APIs or pages.
        """
        # In v1, pricing is static. This method is a placeholder for
        # future dynamic pricing updates.
        pass

    def get_cost_per_request(
        self, provider_name: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate the cost of a request to a specific provider."""
        pricing = self.pricing.get(provider_name, {"input": 5.0, "output": 25.0})
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def get_revenue_per_request(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate revenue from a request at our pricing."""
        input_rev = (prompt_tokens / 1_000_000) * SUPERINTELLIGENCE_PRICING["input"]
        output_rev = (completion_tokens / 1_000_000) * SUPERINTELLIGENCE_PRICING["output"]
        return input_rev + output_rev

    def get_margin(
        self, provider_name: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate margin for a request."""
        cost = self.get_cost_per_request(provider_name, prompt_tokens, completion_tokens)
        revenue = self.get_revenue_per_request(prompt_tokens, completion_tokens)
        return revenue - cost

    def get_pricing_summary(self) -> dict:
        """Get a summary of all provider pricing."""
        return {
            "our_pricing": SUPERINTELLIGENCE_PRICING,
            "provider_pricing": self.pricing,
        }

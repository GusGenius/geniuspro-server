"""
GeniusPro Superintelligence v1 — Latency Agent

Monitors all provider endpoints every 60 seconds. If a provider
fails health checks, marks it unhealthy in the router so requests
are automatically rerouted. When it recovers, marks it healthy again.
"""

import asyncio
import time
from typing import Optional

from superintelligence.routing.router import SuperintelligenceRouter


class LatencyAgent:
    """Monitors provider health and latency."""

    def __init__(self, router: SuperintelligenceRouter, interval: int = 60) -> None:
        self.router = router
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Track latency history per provider: name -> list of (timestamp, latency_ms)
        self.latency_history: dict[str, list[tuple[float, float]]] = {}

    async def start(self) -> None:
        """Start the latency monitoring loop."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        print("  Latency Agent started")

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("  Latency Agent stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop — runs every `interval` seconds."""
        while self._running:
            try:
                await self._check_all_providers()
            except Exception as e:
                print(f"  Latency Agent error: {e}")
            await asyncio.sleep(self.interval)

    async def _check_all_providers(self) -> None:
        """Check health of all registered providers in parallel."""
        tasks = {}
        for name, provider in self.router.providers.items():
            tasks[name] = asyncio.create_task(self._check_provider(name, provider))

        for name, task in tasks.items():
            try:
                is_healthy, latency_ms = await task
                was_healthy = self.router._provider_health.get(name, True)

                if is_healthy and not was_healthy:
                    self.router.mark_healthy(name)
                    print(f"  Latency Agent: {name} recovered ({latency_ms:.0f}ms)")
                elif not is_healthy and was_healthy:
                    self.router.mark_unhealthy(name)
                    print(f"  Latency Agent: {name} is DOWN")

                # Record latency
                if name not in self.latency_history:
                    self.latency_history[name] = []
                self.latency_history[name].append((time.time(), latency_ms))

                # Keep only last 60 readings (~1 hour at 60s interval)
                if len(self.latency_history[name]) > 60:
                    self.latency_history[name] = self.latency_history[name][-60:]

            except Exception as e:
                print(f"  Latency Agent: error checking {name}: {e}")

    async def _check_provider(self, name: str, provider) -> tuple[bool, float]:
        """Check a single provider. Returns (is_healthy, latency_ms)."""
        start = time.time()
        try:
            is_healthy = await provider.health_check()
            latency_ms = (time.time() - start) * 1000
            return is_healthy, latency_ms
        except Exception:
            latency_ms = (time.time() - start) * 1000
            return False, latency_ms

    def get_stats(self) -> dict[str, dict]:
        """Get latency stats for all providers."""
        stats = {}
        for name, history in self.latency_history.items():
            if not history:
                continue
            latencies = [h[1] for h in history]
            stats[name] = {
                "healthy": self.router._provider_health.get(name, True),
                "avg_ms": sum(latencies) / len(latencies),
                "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
                "last_ms": latencies[-1],
                "checks": len(latencies),
            }
        return stats

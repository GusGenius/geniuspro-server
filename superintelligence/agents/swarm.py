"""
GeniusPro Superintelligence v1 — Agent Swarm Orchestrator

Manages the lifecycle of all agents. Starts them on app startup,
stops them on shutdown. Future agents (Benchmark, Discovery, Drift,
Quality) are added here as they're built.
"""

import asyncio
from typing import Optional

from superintelligence.agents.latency import LatencyAgent
from superintelligence.agents.cost import CostAgent
from superintelligence.agents.benchmark import BenchmarkAgent
from superintelligence.routing.router import SuperintelligenceRouter


class AgentSwarm:
    """
    Orchestrates all background agents.

    Active agents:
      - Latency Agent: health checks every 60s
      - Cost Agent: pricing tracker every 1hr
      - Benchmark Agent: full eval suite every 24hr + on-demand

    Future agents (Phase 2+):
      - Discovery Agent: new model detection
      - Drift Agent: model change detection
      - Quality Agent: routing accuracy scoring
    """

    def __init__(self, router: SuperintelligenceRouter) -> None:
        self.router = router
        self.latency_agent = LatencyAgent(router, interval=60)
        self.cost_agent = CostAgent(interval=3600)
        self.benchmark_agent = BenchmarkAgent(router, interval=86400)

    async def start(self) -> None:
        """Start all agents."""
        print("Starting Agent Swarm...")
        await self.latency_agent.start()
        await self.cost_agent.start()
        await self.benchmark_agent.start()
        print("Agent Swarm active (3 agents)")

    async def stop(self) -> None:
        """Stop all agents."""
        print("Stopping Agent Swarm...")
        await self.latency_agent.stop()
        await self.cost_agent.stop()
        await self.benchmark_agent.stop()
        print("Agent Swarm stopped")

    def get_status(self) -> dict:
        """Get status of all agents."""
        return {
            "latency_agent": {
                "running": self.latency_agent._running,
                "stats": self.latency_agent.get_stats(),
            },
            "cost_agent": {
                "running": self.cost_agent._running,
                "pricing": self.cost_agent.get_pricing_summary(),
            },
            "benchmark_agent": {
                "running": self.benchmark_agent._running,
                "results": self.benchmark_agent.get_results(),
            },
            # Future agents
            "discovery_agent": {"running": False, "status": "not_implemented"},
            "drift_agent": {"running": False, "status": "not_implemented"},
            "quality_agent": {"running": False, "status": "not_implemented"},
        }

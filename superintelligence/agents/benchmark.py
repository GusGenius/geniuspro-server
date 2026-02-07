"""
GeniusPro Superintelligence v1 — Benchmark Agent

Runs our own evaluation suite against all providers daily and on
new model detection. Updates the router's scores with real performance
data instead of relying on external leaderboards.

Test suite covers all task types:
  - CODE: coding challenges (HumanEval-style)
  - MATH: competition math problems
  - REASONING: complex multi-step reasoning
  - CREATIVE: writing quality assessment
  - GENERAL: broad knowledge questions
"""

import asyncio
import json
import time
import traceback
from typing import Optional

from superintelligence.providers.base import BaseProvider, ChatRequest, ChatMessage
from superintelligence.routing.classifier import TaskType
from superintelligence.routing.router import SuperintelligenceRouter


# ── Evaluation prompts per task type ─────────────────────────────────────────
# Each test has a prompt, expected answer pattern, and scoring criteria.

EVAL_SUITE: dict[TaskType, list[dict]] = {
    TaskType.CODE: [
        {
            "prompt": "Write a Python function called `merge_sorted` that takes two sorted lists and returns a single sorted list. Do not use the built-in sort. Return only the function, no explanation.",
            "check": "def merge_sorted",
            "verify_code": True,
        },
        {
            "prompt": "Write a Python function `is_balanced` that checks if a string of parentheses (), [], {} is balanced. Return True or False. Return only the function.",
            "check": "def is_balanced",
            "verify_code": True,
        },
        {
            "prompt": "Write a Python function `flatten` that takes a nested list of any depth and returns a flat list. Example: flatten([1, [2, [3, 4], 5]]) -> [1, 2, 3, 4, 5]. Return only the function.",
            "check": "def flatten",
            "verify_code": True,
        },
        {
            "prompt": "Write a SQL query to find the top 3 customers by total order amount from tables `customers` (id, name) and `orders` (id, customer_id, amount). Return only the SQL.",
            "check": "SELECT",
            "verify_code": False,
        },
        {
            "prompt": "Write a Python function `lru_cache` that implements a least recently used cache with a max size. It should support get(key) and put(key, value). Return only the class.",
            "check": "class",
            "verify_code": True,
        },
    ],
    TaskType.MATH: [
        {
            "prompt": "What is the sum of the first 100 positive integers? Answer with just the number.",
            "answer": "5050",
        },
        {
            "prompt": "Solve: If f(x) = 3x^2 - 2x + 1, what is f(4)? Answer with just the number.",
            "answer": "41",
        },
        {
            "prompt": "What is the derivative of x^3 * sin(x)? Give the simplified expression.",
            "check": "3x^2",
        },
        {
            "prompt": "How many ways can you arrange the letters in the word MISSISSIPPI? Answer with just the number.",
            "answer": "34650",
        },
        {
            "prompt": "What is the GCD of 252 and 198? Answer with just the number.",
            "answer": "18",
        },
    ],
    TaskType.REASONING: [
        {
            "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Answer with just the number.",
            "answer": "9",
        },
        {
            "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Answer in minutes with just the number.",
            "answer": "5",
        },
        {
            "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer in dollars.",
            "answer": "0.05",
        },
        {
            "prompt": "Three people check into a hotel room that costs $30. They each pay $10. The manager realizes the room is only $25, so he sends a bellboy with $5. The bellboy keeps $2 and gives each person $1 back. Now each person paid $9 (total $27) and the bellboy has $2 (total $29). Where did the missing dollar go? Explain briefly.",
            "check": "no missing dollar",
        },
        {
            "prompt": "If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons? List the steps briefly.",
            "check": "4",
        },
    ],
    TaskType.CREATIVE: [
        {
            "prompt": "Write a haiku about artificial intelligence.",
            "min_length": 10,
            "max_length": 200,
        },
        {
            "prompt": "Write a one-paragraph product description for a smart water bottle that tracks hydration and glows when you need to drink.",
            "min_length": 50,
            "max_length": 500,
        },
        {
            "prompt": "Rewrite this sentence in a more engaging way: 'The software update includes several bug fixes and performance improvements.'",
            "min_length": 20,
            "max_length": 300,
        },
    ],
    TaskType.GENERAL: [
        {
            "prompt": "What is the capital of Australia? Answer with just the city name.",
            "answer": "Canberra",
        },
        {
            "prompt": "Who wrote the novel '1984'? Answer with just the author's name.",
            "check": "Orwell",
        },
        {
            "prompt": "What is the speed of light in meters per second? Give the approximate value.",
            "check": "300",
        },
        {
            "prompt": "Explain what an API is in one sentence, suitable for a non-technical person.",
            "min_length": 20,
            "max_length": 300,
        },
    ],
}


def _score_response(test: dict, response: str) -> float:
    """
    Score a response against a test case. Returns 0.0 to 1.0.

    Scoring methods:
      - "answer": exact match (case-insensitive, stripped)
      - "check": substring match
      - "verify_code": checks for function/class definition
      - "min_length"/"max_length": length within bounds
    """
    text = response.strip()

    if "answer" in test:
        expected = test["answer"].strip().lower()
        # Check if the answer appears in the response
        if expected in text.lower():
            return 1.0
        return 0.0

    if "check" in test:
        if test["check"].lower() in text.lower():
            return 1.0
        return 0.0

    if test.get("verify_code"):
        check = test.get("check", "def ")
        if check.lower() in text.lower():
            return 1.0
        return 0.0

    # Length-based scoring (for creative tasks)
    score = 1.0
    if "min_length" in test and len(text) < test["min_length"]:
        score *= 0.3
    if "max_length" in test and len(text) > test["max_length"]:
        score *= 0.7
    if len(text) == 0:
        return 0.0
    return score


class BenchmarkAgent:
    """
    Runs evaluation suite against all providers and updates router scores.

    Schedule:
      - Daily at a configurable hour
      - On-demand when triggered by Discovery Agent (new model detected)
    """

    def __init__(
        self,
        router: SuperintelligenceRouter,
        interval: int = 86400,  # 24 hours
    ) -> None:
        self.router = router
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Store last eval results: provider -> task_type -> score
        self.last_results: dict[str, dict[str, float]] = {}
        self.last_run_time: Optional[float] = None

    async def start(self) -> None:
        """Start the benchmark loop."""
        self._running = True
        self._task = asyncio.create_task(self._benchmark_loop())
        print("  Benchmark Agent started (first run in 60s)")

    async def stop(self) -> None:
        """Stop the benchmark loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _benchmark_loop(self) -> None:
        """Main loop — runs evals periodically."""
        # Wait 60s before first run to let everything initialize
        await asyncio.sleep(60)

        while self._running:
            try:
                await self.run_full_eval()
            except Exception as e:
                print(f"  Benchmark Agent error: {e}")
                traceback.print_exc()
            await asyncio.sleep(self.interval)

    async def run_full_eval(self) -> dict[str, dict[str, float]]:
        """Run the full evaluation suite against all providers."""
        print("  Benchmark Agent: Starting full evaluation...")
        start = time.time()
        results: dict[str, dict[str, float]] = {}

        for name, provider in self.router.providers.items():
            print(f"    Evaluating {name}...")
            provider_scores: dict[str, float] = {}

            for task_type, tests in EVAL_SUITE.items():
                score = await self._eval_provider(provider, tests)
                provider_scores[task_type.value] = score

                # Update the router score (normalize to 0-100 scale)
                new_score = score * 100
                old_score = self.router.scores.get(name, {}).get(task_type, 50.0)
                # Blend: 70% new benchmark, 30% old score (smooth transitions)
                blended = (new_score * 0.7) + (old_score * 0.3)
                self.router.update_score(name, task_type, blended)

            results[name] = provider_scores
            print(f"    {name}: {provider_scores}")

        self.last_results = results
        self.last_run_time = time.time()
        elapsed = time.time() - start
        print(f"  Benchmark Agent: Evaluation complete ({elapsed:.1f}s)")
        return results

    async def _eval_provider(
        self, provider: BaseProvider, tests: list[dict]
    ) -> float:
        """Evaluate a provider on a set of tests. Returns average score 0.0-1.0."""
        scores: list[float] = []

        for test in tests:
            try:
                request = ChatRequest(
                    messages=[ChatMessage(role="user", content=test["prompt"])],
                    temperature=0.0,  # Deterministic for benchmarks
                    max_tokens=1024,
                    stream=False,
                )
                request.model = provider.default_model
                result = await asyncio.wait_for(
                    provider.chat_completion(request),
                    timeout=60,
                )
                score = _score_response(test, result.content)
                scores.append(score)

            except asyncio.TimeoutError:
                scores.append(0.0)
            except Exception as e:
                print(f"      Test failed for {provider.name}: {e}")
                scores.append(0.0)

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    async def trigger_eval(self, provider_name: Optional[str] = None) -> None:
        """Trigger an on-demand evaluation (e.g., new model detected)."""
        if provider_name and provider_name in self.router.providers:
            provider = self.router.providers[provider_name]
            print(f"  Benchmark Agent: On-demand eval for {provider_name}")
            for task_type, tests in EVAL_SUITE.items():
                score = await self._eval_provider(provider, tests)
                new_score = score * 100
                old_score = self.router.scores.get(provider_name, {}).get(task_type, 50.0)
                blended = (new_score * 0.7) + (old_score * 0.3)
                self.router.update_score(provider_name, task_type, blended)
        else:
            await self.run_full_eval()

    def get_results(self) -> dict:
        """Get the latest benchmark results."""
        return {
            "last_run": self.last_run_time,
            "results": self.last_results,
        }

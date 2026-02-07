"""
GeniusPro Superintelligence v1 — Anthropic Provider

Handles the Anthropic Messages API format (Claude Opus 4.6, Sonnet, etc.).
Translates between OpenAI-style requests and Anthropic's format.
"""

import json
from typing import AsyncIterator

import aiohttp

from superintelligence.providers.base import (
    BaseProvider,
    ChatRequest,
    ProviderResponse,
    StreamChunk,
)

ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Provider client for the Anthropic Messages API."""

    def _auth_headers(self) -> dict[str, str]:
        """Anthropic uses x-api-key instead of Bearer token."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
        }

    async def chat_completion(self, request: ChatRequest) -> ProviderResponse:
        """Send a non-streaming chat completion request."""
        model = self._get_model(request)
        payload = self._build_payload(request, model, stream=False)

        async with self.session.post(
            f"{self.base_url}/v1/messages",
            headers=self._auth_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"Provider {self.name} returned {resp.status}: {error_text}"
                )
            data = await resp.json()

        # Extract content from Anthropic's response format
        content_blocks = data.get("content", [])
        content = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )

        usage = data.get("usage", {})

        return ProviderResponse(
            content=content,
            model=model,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "end_turn"),
        )

    async def stream_chat_completion(
        self, request: ChatRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat completion request. Yields StreamChunks."""
        model = self._get_model(request)
        payload = self._build_payload(request, model, stream=True)

        prompt_tokens = 0
        completion_tokens = 0

        async with self.session.post(
            f"{self.base_url}/v1/messages",
            headers=self._auth_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"Provider {self.name} returned {resp.status}: {error_text}"
                )

            event_type = ""
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue

                if decoded.startswith("event: "):
                    event_type = decoded[7:]
                    continue

                if not decoded.startswith("data: "):
                    continue

                data_str = decoded[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if event_type == "message_start":
                    usage = data.get("message", {}).get("usage", {})
                    prompt_tokens = usage.get("input_tokens", 0)

                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield StreamChunk(content=delta.get("text", ""))

                elif event_type == "message_delta":
                    usage = data.get("usage", {})
                    completion_tokens = usage.get("output_tokens", 0)
                    stop_reason = data.get("delta", {}).get("stop_reason")
                    yield StreamChunk(
                        finish_reason=stop_reason or "end_turn",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        done=True,
                    )

                elif event_type == "message_stop":
                    yield StreamChunk(
                        finish_reason="end_turn",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        done=True,
                    )
                    return

    def _build_payload(
        self, request: ChatRequest, model: str, stream: bool
    ) -> dict:
        """Build the Anthropic Messages API payload."""
        system_prompt = ""
        messages = []

        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "stream": stream,
        }

        if system_prompt:
            payload["system"] = system_prompt
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop

        return payload

    async def health_check(self) -> bool:
        """Anthropic doesn't have a /models endpoint. Use a lightweight call."""
        try:
            async with self.session.post(
                f"{self.base_url}/v1/messages",
                headers=self._auth_headers(),
                json={
                    "model": self.default_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

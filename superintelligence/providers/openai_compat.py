"""
GeniusPro Superintelligence v1 — OpenAI-Compatible Provider

Handles all providers that use the OpenAI API format:
- OpenAI (GPT-5.3-Codex)
- DeepSeek (V3.2)
- OpenRouter (Grok 4, Kimi K2.5, etc.)
- Mistral (Large 3, Medium 3.1)
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


class OpenAICompatProvider(BaseProvider):
    """Provider client for any OpenAI-compatible API."""

    async def chat_completion(self, request: ChatRequest) -> ProviderResponse:
        """Send a non-streaming chat completion request."""
        model = self._get_model(request)
        payload = self._build_payload(request, model, stream=False)

        async with self.session.post(
            f"{self.base_url}/chat/completions",
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

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})

        return ProviderResponse(
            content=choice.get("message", {}).get("content", ""),
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def stream_chat_completion(
        self, request: ChatRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat completion request. Yields StreamChunks."""
        model = self._get_model(request)
        payload = self._build_payload(request, model, stream=True)

        async with self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self._auth_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"Provider {self.name} returned {resp.status}: {error_text}"
                )

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue

                data_str = decoded[6:]  # Strip "data: " prefix
                if data_str == "[DONE]":
                    yield StreamChunk(done=True, finish_reason="stop")
                    return

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason")
                usage = data.get("usage", {})

                yield StreamChunk(
                    content=delta.get("content", ""),
                    finish_reason=finish,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    done=finish is not None,
                )

    def _build_payload(
        self, request: ChatRequest, model: str, stream: bool
    ) -> dict:
        """Build the OpenAI-format request payload."""
        messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop

        # Request usage in streaming for providers that support it
        if stream:
            payload["stream_options"] = {"include_usage": True}

        return payload

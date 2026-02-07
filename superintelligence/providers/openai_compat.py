"""
GeniusPro Superintelligence v1 — OpenAI-Compatible Provider

Handles all providers that use the OpenAI API format:
- OpenAI (GPT-5.3-Codex)
- DeepSeek (V3.2)
- OpenRouter (Grok 4, Kimi K2.5, etc.)
- Mistral (Large 3, Medium 3.1)
"""

import json
import logging
from typing import AsyncIterator

import aiohttp

from superintelligence.providers.base import (
    BaseProvider,
    ChatRequest,
    ProviderResponse,
    StreamChunk,
)

logger = logging.getLogger("superintelligence.provider.openai")


class OpenAICompatProvider(BaseProvider):
    """Provider client for any OpenAI-compatible API."""

    async def chat_completion(self, request: ChatRequest) -> ProviderResponse:
        """Send a non-streaming chat completion request."""
        model = self._get_model(request)
        payload = self._build_payload(request, model, stream=False)

        logger.info("[%s] POST %s/chat/completions model=%s msgs=%d",
                     self.name, self.base_url, model, len(request.messages))

        async with self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self._auth_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error("[%s] Error %d: %s", self.name, resp.status, error_text[:200])
                raise RuntimeError(
                    f"Provider {self.name} returned {resp.status}: {error_text}"
                )
            data = await resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        # Extract tool calls if present
        tool_calls = message.get("tool_calls")
        if tool_calls:
            # Normalize tool calls to unified format
            normalized_tool_calls = []
            for tc in tool_calls:
                normalized_tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": tc.get("function", {}),
                })
        else:
            normalized_tool_calls = None

        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        logger.info("[%s] Response: tokens=%d+%d finish=%s tools=%d",
                     self.name, p_tok, c_tok,
                     choice.get("finish_reason", "stop"),
                     len(normalized_tool_calls) if normalized_tool_calls else 0)

        return ProviderResponse(
            content=message.get("content", ""),
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=normalized_tool_calls,
        )

    async def stream_chat_completion(
        self, request: ChatRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat completion request. Yields StreamChunks."""
        model = self._get_model(request)
        payload = self._build_payload(request, model, stream=True)

        logger.info("[%s] STREAM POST %s/chat/completions model=%s msgs=%d",
                     self.name, self.base_url, model, len(request.messages))

        accumulated_tool_calls = {}  # Track tool calls across chunks

        async with self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self._auth_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error("[%s] Stream error %d: %s", self.name, resp.status, error_text[:200])
                raise RuntimeError(
                    f"Provider {self.name} returned {resp.status}: {error_text}"
                )

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue

                data_str = decoded[6:]  # Strip "data: " prefix
                if data_str == "[DONE]":
                    # Finalize any accumulated tool calls
                    tool_calls_list = None
                    if accumulated_tool_calls:
                        tool_calls_list = list(accumulated_tool_calls.values())
                    yield StreamChunk(
                        done=True,
                        finish_reason="stop",
                        tool_calls_delta=tool_calls_list,
                    )
                    return

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason")
                usage = data.get("usage", {})

                # Handle tool calls in streaming
                tool_calls_delta = None
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        index = tc.get("index", 0)
                        tool_id = tc.get("id", "")
                        function_delta = tc.get("function", {})

                        if tool_id not in accumulated_tool_calls:
                            accumulated_tool_calls[tool_id] = {
                                "id": tool_id,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }

                        if "name" in function_delta:
                            accumulated_tool_calls[tool_id]["function"]["name"] += function_delta["name"]
                        if "arguments" in function_delta:
                            accumulated_tool_calls[tool_id]["function"]["arguments"] += function_delta["arguments"]

                # Only send tool calls on final chunk
                final_tool_calls = None
                if finish and accumulated_tool_calls:
                    final_tool_calls = list(accumulated_tool_calls.values())

                yield StreamChunk(
                    content=delta.get("content", ""),
                    finish_reason=finish,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    done=finish is not None,
                    tool_calls_delta=final_tool_calls,
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

        # Add tools if present
        if request.tools:
            payload["tools"] = request.tools
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice
            else:
                payload["tool_choice"] = "auto"

        # Request usage in streaming for providers that support it
        if stream:
            payload["stream_options"] = {"include_usage": True}

        return payload

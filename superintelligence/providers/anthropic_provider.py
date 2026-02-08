"""
GeniusPro Superintelligence v1 — Anthropic Provider

Handles the Anthropic Messages API format (Claude Opus 4.6, Sonnet, etc.).
Translates between OpenAI-style requests and Anthropic's format.
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

logger = logging.getLogger("superintelligence.provider.anthropic")

ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Provider client for the Anthropic Messages API."""

    def _auth_headers(self) -> dict[str, str]:
        """Anthropic uses x-api-key instead of Bearer token."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "anthropic-beta": "web-fetch-2025-09-10",
            "Content-Type": "application/json",
        }

    async def chat_completion(self, request: ChatRequest) -> ProviderResponse:
        """Send a non-streaming chat completion request."""
        model = self._get_model(request)
        payload = self._build_payload(request, model, stream=False)

        logger.info("[%s] POST %s/v1/messages model=%s msgs=%d",
                     self.name, self.base_url, model, len(request.messages))

        async with self.session.post(
            f"{self.base_url}/v1/messages",
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

        # Extract content and tool calls from Anthropic's response format
        content_blocks = data.get("content", [])
        content = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )

        # Extract tool_use blocks
        tool_calls = []
        for block in content_blocks:
            if block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "tool_use",
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                })

        usage = data.get("usage", {})

        # Convert Anthropic tool_use to unified format
        unified_tool_calls = None
        if tool_calls:
            unified_tool_calls = []
            for tc in tool_calls:
                unified_tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("input", {})),
                    },
                })

        p_tok = usage.get("input_tokens", 0)
        c_tok = usage.get("output_tokens", 0)
        logger.info("[%s] Response: tokens=%d+%d finish=%s tools=%d",
                     self.name, p_tok, c_tok,
                     data.get("stop_reason", "end_turn"),
                     len(unified_tool_calls) if unified_tool_calls else 0)

        return ProviderResponse(
            content=content,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            finish_reason=data.get("stop_reason", "end_turn"),
            tool_calls=unified_tool_calls,
        )

    async def stream_chat_completion(
        self, request: ChatRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat completion request. Yields StreamChunks."""
        model = self._get_model(request)
        payload = self._build_payload(request, model, stream=True)

        logger.info("[%s] STREAM POST %s/v1/messages model=%s msgs=%d",
                     self.name, self.base_url, model, len(request.messages))

        prompt_tokens = 0
        completion_tokens = 0
        accumulated_tool_calls = []  # Track tool calls by index

        async with self.session.post(
            f"{self.base_url}/v1/messages",
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

                elif event_type == "content_block_start":
                    # Tool use block started
                    block = data.get("content_block", {})
                    if block.get("type") == "tool_use":
                        index = data.get("index", len(accumulated_tool_calls))
                        # Ensure list is large enough
                        while len(accumulated_tool_calls) <= index:
                            accumulated_tool_calls.append(None)
                        accumulated_tool_calls[index] = {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": "",
                            },
                        }

                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield StreamChunk(content=delta.get("text", ""))
                    elif delta.get("type") == "input_json_delta":
                        # Tool input is streaming in
                        index = data.get("index", 0)
                        if index < len(accumulated_tool_calls) and accumulated_tool_calls[index]:
                            accumulated_tool_calls[index]["function"]["arguments"] += delta.get("partial_json", "")

                elif event_type == "message_delta":
                    usage = data.get("usage", {})
                    completion_tokens = usage.get("output_tokens", 0)
                    stop_reason = data.get("delta", {}).get("stop_reason")
                    
                    # Finalize tool calls
                    final_tool_calls = None
                    if accumulated_tool_calls:
                        final_tool_calls = []
                        for tc in accumulated_tool_calls:
                            if tc is None:
                                continue
                            # Parse JSON arguments
                            try:
                                args_json = tc["function"]["arguments"]
                                args_dict = json.loads(args_json) if args_json else {}
                                final_tool_calls.append({
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": json.dumps(args_dict),
                                    },
                                })
                            except json.JSONDecodeError:
                                # If JSON is incomplete, skip this tool call
                                pass

                    yield StreamChunk(
                        finish_reason=stop_reason or "end_turn",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        done=True,
                        tool_calls_delta=final_tool_calls if final_tool_calls else None,
                    )

                elif event_type == "message_stop":
                    # Finalize tool calls
                    final_tool_calls = None
                    if accumulated_tool_calls:
                        final_tool_calls = []
                        for tc in accumulated_tool_calls:
                            if tc is None:
                                continue
                            try:
                                args_json = tc["function"]["arguments"]
                                args_dict = json.loads(args_json) if args_json else {}
                                final_tool_calls.append({
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": json.dumps(args_dict),
                                    },
                                })
                            except json.JSONDecodeError:
                                pass

                    yield StreamChunk(
                        finish_reason="end_turn",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        done=True,
                        tool_calls_delta=final_tool_calls if final_tool_calls else None,
                    )
                    return

    def _build_payload(
        self, request: ChatRequest, model: str, stream: bool
    ) -> dict:
        """Build the Anthropic Messages API payload."""
        system_prompts: list[str] = []
        messages = []

        for msg in request.messages:
            if msg.role == "system":
                if msg.content:
                    system_prompts.append(msg.content)
            else:
                messages.append({"role": msg.role, "content": msg.content})

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "stream": stream,
        }

        if system_prompts:
            # Anthropic Messages API supports a single "system" string.
            # We preserve multiple system messages by concatenating in order.
            payload["system"] = "\n\n".join(system_prompts)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop

        # Add tools if present (already in Anthropic format from mapper)
        if request.tools:
            payload["tools"] = request.tools

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

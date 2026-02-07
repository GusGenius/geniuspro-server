"""
GeniusPro Superintelligence v1 — Base Provider

Abstract base class for all LLM provider clients. Every provider must
implement chat_completion() and stream_chat_completion().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import aiohttp


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatRequest:
    """Normalized chat completion request."""

    messages: list[ChatMessage]
    model: str = ""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[list[str]] = None
    stream: bool = False
    tools: Optional[list[dict]] = None  # Tool definitions in OpenAI format
    tool_choice: Optional[str] = None  # "auto", "none", or "required"


@dataclass
class ProviderResponse:
    """Normalized response from a provider (non-streaming)."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    tool_calls: Optional[list[dict]] = None  # Tool calls in unified format


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    content: str = ""
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    done: bool = False
    tool_calls_delta: Optional[list[dict]] = None  # Incremental tool call data


class BaseProvider(ABC):
    """Abstract base class for LLM provider clients."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        default_model: str,
        session: aiohttp.ClientSession,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.session = session

    @abstractmethod
    async def chat_completion(
        self, request: ChatRequest
    ) -> ProviderResponse:
        """Send a non-streaming chat completion request."""
        ...

    @abstractmethod
    async def stream_chat_completion(
        self, request: ChatRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat completion request. Yields StreamChunks."""
        ...

    async def health_check(self) -> bool:
        """Check if the provider is reachable. Override for custom logic."""
        try:
            async with self.session.get(
                f"{self.base_url}/models",
                headers=self._auth_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _auth_headers(self) -> dict[str, str]:
        """Default auth headers (OpenAI-style Bearer token)."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_model(self, request: ChatRequest) -> str:
        """Get model ID from request or fall back to default."""
        return request.model if request.model else self.default_model

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} model={self.default_model}>"

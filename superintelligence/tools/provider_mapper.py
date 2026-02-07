"""
GeniusPro Superintelligence v1 — Provider Tool Mapper

Converts unified tool definitions to each provider's native format and back.

Key differences between providers:
  - Anthropic: Custom tools use {name, description, input_schema}.
    Has native server tools: web_search_20250305, web_fetch_20250910.
  - OpenAI-compatible: Standard {type: "function", function: {name, description, parameters}}.
  - Some OpenRouter models don't support tools at all — strip them.
"""

import logging
from typing import Optional

logger = logging.getLogger("superintelligence.tools.mapper")

# Providers that reliably support function/tool calling
TOOL_CAPABLE_PROVIDERS = {
    "openai", "anthropic", "deepseek", "google",
    "openrouter-deepseek", "openrouter-google",
}

# Providers where we should NOT send tools (unreliable or no support)
# openrouter (Kimi K2): rejects non-standard function names
# openrouter-mistral / mistral: inconsistent tool support
TOOL_INCAPABLE_PROVIDERS = {
    "openrouter", "openrouter-mistral", "mistral",
}


class ProviderToolMapper:
    """Maps tools between unified format and provider-specific formats."""

    # Providers with native web search (executed server-side by the provider)
    NATIVE_SEARCH_PROVIDERS = {"anthropic"}
    # Providers with native web fetch
    NATIVE_FETCH_PROVIDERS = {"anthropic"}

    @staticmethod
    def to_provider_format(
        tools: list[dict], provider_name: str
    ) -> Optional[list[dict]]:
        """Convert unified (OpenAI-format) tool definitions to provider format."""
        if not tools:
            return None

        # Strip tools for providers that can't handle them
        if provider_name in TOOL_INCAPABLE_PROVIDERS:
            logger.info("Stripping tools for provider %s (not tool-capable)", provider_name)
            return None

        if provider_name == "anthropic":
            return ProviderToolMapper._to_anthropic_format(tools)

        # OpenAI-compatible (openai, deepseek, openrouter-*, google, etc.)
        return ProviderToolMapper._to_openai_format(tools)

    @staticmethod
    def _to_anthropic_format(tools: list[dict]) -> list[dict]:
        """
        Convert to Anthropic Messages API tool format.

        Anthropic has two tool types:
        1. Server tools (native): web_search_20250305, web_fetch_20250910
           → Executed on Anthropic's servers, we just declare them.
        2. Client tools (custom): {name, description, input_schema}
           → We execute them and return results.
        """
        anthropic_tools: list[dict] = []

        for tool in tools:
            if tool.get("type") != "function":
                continue

            func = tool.get("function", {})
            name = func.get("name", "")

            if name == "web_search":
                # Use Anthropic's native server-side web search
                anthropic_tools.append({
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5,
                })
                logger.debug("Mapped web_search → Anthropic native web_search_20250305")

            elif name == "url_fetch":
                # Use Anthropic's native server-side web fetch
                anthropic_tools.append({
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                    "max_uses": 5,
                })
                logger.debug("Mapped url_fetch → Anthropic native web_fetch_20250910")

            else:
                # Custom client tool → Anthropic format
                anthropic_tools.append({
                    "name": name,
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })

        return anthropic_tools if anthropic_tools else None

    @staticmethod
    def _to_openai_format(tools: list[dict]) -> list[dict]:
        """
        Convert to OpenAI function-calling format.
        Used by OpenAI, DeepSeek, OpenRouter, Google, Mistral.
        """
        openai_tools: list[dict] = []

        for tool in tools:
            if tool.get("type") == "function":
                openai_tools.append(tool)

        return openai_tools if openai_tools else None

    @staticmethod
    def is_native_tool(tool_name: str, provider_name: str) -> bool:
        """
        Check if a tool is handled natively by the provider.

        Native tools are executed server-side by the provider itself.
        We should NOT execute them in our tool registry.
        """
        if tool_name == "web_search":
            return provider_name in ProviderToolMapper.NATIVE_SEARCH_PROVIDERS
        if tool_name in ("url_fetch", "web_fetch"):
            return provider_name in ProviderToolMapper.NATIVE_FETCH_PROVIDERS
        return False

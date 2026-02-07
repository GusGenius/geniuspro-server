"""
GeniusPro Superintelligence v1 — Web Search Tool

For providers with native web search (Claude, GPT, Gemini), this tool
maps to their native implementation. For providers without native search
(DeepSeek, Mistral), falls back to Tavily API.
"""

import os
from typing import Optional

import aiohttp

from superintelligence.tools.registry import BaseTool


class WebSearchTool(BaseTool):
    """
    Web search tool that uses provider-native search when available,
    falls back to Tavily for providers without native search.
    """

    name = "web_search"
    description = "Search the web for current information. Use this when you need up-to-date information, recent news, or real-time data."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session
        self.tavily_api_key = os.environ.get("TAVILY_API_KEY", "")

    async def execute(self, arguments: dict) -> str:
        """
        Execute web search via Tavily (fallback for providers without native search).
        
        Note: For providers with native search (Claude, GPT, Gemini), this method
        should never be called — the provider mapper maps web_search to their
        native tool instead.
        """
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)

        if not self.tavily_api_key:
            return "Error: TAVILY_API_KEY not configured. Web search unavailable for this provider."

        if not self.session:
            return "Error: HTTP session not available."

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "max_results": min(max_results, 10),  # Tavily limit
            "include_answer": True,
            "include_raw_content": False,
        }

        try:
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return f"Error: Tavily API returned {resp.status}: {error_text}"

                data = await resp.json()

                # Format results
                answer = data.get("answer", "")
                results = data.get("results", [])

                if answer:
                    result_text = f"Answer: {answer}\n\n"
                else:
                    result_text = ""

                if results:
                    result_text += "Sources:\n"
                    for i, result in enumerate(results[:max_results], 1):
                        title = result.get("title", "Untitled")
                        url = result.get("url", "")
                        content = result.get("content", "")[:500]  # Truncate long content
                        result_text += f"{i}. {title}\n   URL: {url}\n   {content}...\n\n"

                return result_text.strip() or "No results found."

        except Exception as e:
            return f"Error executing web search: {str(e)}"

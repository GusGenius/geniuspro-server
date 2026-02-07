"""
GeniusPro Superintelligence v1 — URL Fetch Tool

Fetches and extracts clean text content from a URL.
Always executed server-side (no provider has this built-in universally).
"""

import re
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

from superintelligence.tools.registry import BaseTool


class URLFetchTool(BaseTool):
    """Tool to fetch and extract text content from a URL."""

    name = "url_fetch"
    description = "Fetch and read content from a specific URL. Use this to read web pages, documentation, or any online content."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch and read",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum length of extracted text in characters (default: 10000)",
                "default": 10000,
            },
        },
        "required": ["url"],
    }

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session

    async def execute(self, arguments: dict) -> str:
        """Fetch URL and extract clean text content."""
        url = arguments.get("url", "")
        max_length = arguments.get("max_length", 10000)

        if not url:
            return "Error: URL is required"

        if not self.session:
            return "Error: HTTP session not available"

        # Validate URL
        if not url.startswith(("http://", "https://")):
            return f"Error: Invalid URL format: {url}"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; GeniusPro/1.0; +https://geniuspro.io)",
            }

            async with self.session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=30), allow_redirects=True
            ) as resp:
                if resp.status != 200:
                    return f"Error: HTTP {resp.status} when fetching {url}"

                content_type = resp.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return f"Error: Unsupported content type: {content_type}"

                html = await resp.text()

                # Extract text from HTML
                soup = BeautifulSoup(html, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    script.decompose()

                # Get text
                text = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)

                # Truncate if needed
                if len(text) > max_length:
                    text = text[:max_length] + "... [truncated]"

                if not text.strip():
                    return f"Error: No text content found at {url}"

                return text

        except aiohttp.ClientError as e:
            return f"Error fetching URL: {str(e)}"
        except Exception as e:
            return f"Error processing URL content: {str(e)}"

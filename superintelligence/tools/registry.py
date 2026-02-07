"""
GeniusPro Superintelligence v1 — Tool Registry

Base class for server-side tools and registry for managing them.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("superintelligence.tools")


class BaseTool(ABC):
    """Base class for all server-side tools."""

    name: str
    description: str
    parameters: dict  # JSON Schema

    @abstractmethod
    async def execute(self, arguments: dict) -> str:
        """
        Execute the tool with the given arguments.
        
        Args:
            arguments: Dictionary of arguments matching the tool's parameters schema
            
        Returns:
            Result as a string (will be sent back to the LLM as tool result)
        """
        raise NotImplementedError

    def to_openai_format(self) -> dict:
        """Convert tool definition to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry for server-side tools."""

    def __init__(self):
        self.tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get_definitions(self) -> list[dict]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_openai_format() for tool in self.tools.values()]

    def is_server_tool(self, name: str) -> bool:
        """Check if a tool name is a server-side tool."""
        return name in self.tools

    async def execute(self, name: str, arguments: dict) -> str:
        """
        Execute a server-side tool.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result as string
            
        Raises:
            KeyError: If tool not found
            ValueError: If tool execution fails
        """
        if name not in self.tools:
            logger.error("Tool '%s' not found in registry", name)
            raise KeyError(f"Tool '{name}' not found in registry")
        
        tool = self.tools[name]
        logger.info("Executing tool: %s args=%s", name, list(arguments.keys()))
        try:
            result = await tool.execute(arguments)
            logger.info("Tool '%s' completed — result length=%d", name, len(result))
            return result
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", name, e)
            raise ValueError(f"Tool '{name}' execution failed: {e}") from e

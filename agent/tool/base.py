from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    output: Any = None
    error: Optional[str] = None
    base64_image: Optional[str] = None
    system: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self):
        return self.output is not None or self.error is not None

    def __str__(self):
        return f"Error: {self.error}" if self.error else str(self.output)


class BaseTool(ABC, BaseModel):
    """Base class for all tools."""
    
    name: str
    description: str
    parameters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCollection:
    """A collection of tools that can be used by the agent."""

    def __init__(self, *tools: BaseTool):
        """Initialize with a list of tools."""
        self.tools: List[BaseTool] = list(tools)
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        """Iterate over tools."""
        return iter(self.tools)

    def __len__(self):
        """Get number of tools."""
        return len(self.tools)

    def add(self, tool: BaseTool):
        """Add a tool to the collection."""
        self.tools.append(tool)
        self.tool_map[tool.name] = tool

    def remove(self, tool_name: str):
        """Remove a tool from the collection by name."""
        if tool_name in self.tool_map:
            tool = self.tool_map[tool_name]
            self.tools.remove(tool)
            del self.tool_map[tool_name]

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tool_map.get(tool_name)

    async def execute(self, name: str, tool_input: Dict[str, Any]) -> Any:
        """Execute a tool by name with given input."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")
        return await tool.execute(**tool_input)

    def to_params(self) -> List[Dict]:
        """Convert all tools to function call format."""
        return [tool.to_param() for tool in self.tools]

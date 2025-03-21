from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Base class for tool parameters."""
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    inputSchema: Dict[str, Any] = Field(..., description="JSON Schema for tool inputs")


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

    def __iter__(self):
        """Iterate over tools."""
        return iter(self.tools)

    def __len__(self):
        """Get number of tools."""
        return len(self.tools)

    def add(self, tool: BaseTool):
        """Add a tool to the collection."""
        self.tools.append(tool)

    def remove(self, tool_name: str):
        """Remove a tool from the collection by name."""
        self.tools = [t for t in self.tools if t.name != tool_name]

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def to_params(self) -> List[Dict]:
        """Convert all tools to function call format."""
        return [tool.to_param() for tool in self.tools]


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    output: Any = None
    error: Optional[str] = None
    base64_image: Optional[str] = None
    system: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self):
        return any(getattr(self, field) for field in self.__fields__)

    def __str__(self):
        return f"Error: {self.error}" if self.error else str(self.output)


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""
    pass


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""
    pass


class ToolError(Exception):
    """Exception raised when a tool encounters an error."""
    def __init__(self, message: str, result: Optional[ToolResult] = None):
        self.message = message
        self.result = result
        super().__init__(message) 
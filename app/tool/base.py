import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    """Input to a tool."""

    pass


class ToolOutput(BaseModel):
    """Output of a tool."""

    pass


class Tool(ABC):
    """Base class for tools."""

    description: Optional[str] = None
    inputSchema: Optional[Dict[str, Any]] = None
    outputSchema: Optional[Dict[str, Any]] = None

    def __init__(self):
        pass

    @property
    def name(self):
        """Return the name of the tool."""
        return self.__class__.__name__

    @abstractmethod
    async def run(self, tool_input: ToolInput) -> ToolOutput:
        """Use the tool."""
        pass

    def args_to_input(self, args: Dict[str, Any]) -> ToolInput:
        """Convert arguments to tool input."""
        return ToolInput(**args)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
            "outputSchema": self.outputSchema,
        }

    def __str__(self) -> str:
        """Return a string representation of the tool."""
        return json.dumps(self.to_dict(), indent=2)


class ToolCollection:
    """A collection of tools."""

    def __init__(self, *tools: Tool):
        self.tools: Dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list(self) -> List[Tool]:
        """List all tools."""
        return list(self.tools.values())

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the tool collection."""
        return {name: tool.to_dict() for name, tool in self.tools.items()}

    def __str__(self) -> str:
        """Return a string representation of the tool collection."""
        return json.dumps(self.to_dict(), indent=2)


class Terminate(Tool):
    """A tool to terminate the agent."""

    description = "Terminate the agent"
    inputSchema = {}
    outputSchema = {}

    async def run(self, tool_input: ToolInput) -> ToolOutput:
        """Use the tool."""
        return ToolOutput(complete=True, message="Agent terminated")


class InvalidTool(Tool):
    """A placeholder for an invalid tool."""

    description = "Invalid tool"
    inputSchema = {}
    outputSchema = {}

    async def run(self, tool_input: ToolInput) -> ToolOutput:
        """Use the tool."""
        raise ValueError(f"Invalid tool: {self.name}")


class ToolCall(BaseModel):
    """A tool call."""

    tool: str
    toolInput: Dict[str, Any] = Field(default_factory=dict)
    log: Optional[str] = None

    def to_string(self) -> str:
        """Return a string representation of the tool call."""
        return f"{self.tool}({', '.join(f'{k}={v}' for k, v in self.toolInput.items())})"


class ToolResult(BaseModel):
    """The result of a tool call."""

    output: Optional[str] = None
    error: Optional[str] = None
    log: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return whether the tool call was successful."""
        return self.error is None 
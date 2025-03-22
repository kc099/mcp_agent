from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class AgentState(str, Enum):
    """Enum representing the possible states of an agent."""
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


class ROLE_TYPE(str, Enum):
    """Enum representing the possible roles in a conversation."""
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolChoice(str, Enum):
    """Enum representing the possible tool choice modes."""
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


TOOL_CHOICE_TYPE = Union[ToolChoice, Dict[str, Any]]


class Message(BaseModel):
    """Base class for all message types in the agent's memory."""
    role: ROLE_TYPE
    content: Optional[str] = None
    base64_image: Optional[str] = None
    
    @classmethod
    def user_message(cls, content: str, base64_image: Optional[str] = None) -> "Message":
        """Create a user message."""
        return cls(role=ROLE_TYPE.USER, content=content, base64_image=base64_image)
    
    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=ROLE_TYPE.SYSTEM, content=content)
    
    @classmethod
    def assistant_message(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=ROLE_TYPE.ASSISTANT, content=content)
    
    @classmethod
    def tool_message(cls, content: str, tool_call_id: Optional[str] = None, 
                    name: Optional[str] = None, base64_image: Optional[str] = None) -> "Message":
        """Create a tool message."""
        return cls(
            role=ROLE_TYPE.TOOL, 
            content=content,
            base64_image=base64_image,
            tool_call_id=tool_call_id,
            name=name
        )
    
    @classmethod
    def from_tool_calls(cls, content: Optional[str], tool_calls: List["ToolCall"]) -> "Message":
        """Create an assistant message with tool calls."""
        return cls(
            role=ROLE_TYPE.ASSISTANT,
            content=content,
            tool_calls=tool_calls
        )


class Memory(BaseModel):
    """Class for storing the agent's memory (conversation history)."""
    messages: List[Message] = Field(default_factory=list)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the memory."""
        self.messages.append(message)


class ToolCallFunction(BaseModel):
    """Class representing a function in a tool call."""
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Class representing a tool call."""
    id: str
    type: str = "function"
    function: ToolCallFunction

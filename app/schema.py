import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Step(BaseModel):
    """A step in an agent's plan."""

    tool: str
    tool_input: Optional[str] = None
    result: Optional[str] = None

    @classmethod
    def from_llm_response(cls, llm_response: str) -> "Step":
        """Parse a step from an LLM response."""
        # Extract the JSON object from the LLM response
        match = re.search(r"```(.*?)```", llm_response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            print(f"Extracted JSON: {json_str}")
            try:
                step_dict = json.loads(json_str)
                return cls(**step_dict)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"JSON string: {json_str}")
                raise e
        else:
            print(f"No JSON object found in LLM response: {llm_response}")
            raise ValueError(f"No JSON object found in LLM response: {llm_response}")

    def __str__(self):
        """Return a string representation of the step."""
        return f"{self.tool}({self.tool_input}) -> {self.result}"


class ToolResult(BaseModel):
    """The result of a tool invocation."""

    output: Optional[str] = None
    error: Optional[str] = None
    log: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return whether the tool invocation was successful."""
        return self.error is None


class ToolInvocation(BaseModel):
    """An invocation of a tool."""

    tool: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    log: Optional[str] = None

    def to_string(self) -> str:
        """Return a string representation of the tool invocation."""
        return f"{self.tool}({', '.join(f'{k}={v}' for k, v in self.tool_input.items())})"


class Message(BaseModel):
    """A message in a conversation."""

    role: str
    content: str
    tool_invocation: Optional[ToolInvocation] = None
    tool_result: Optional[ToolResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the message."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_invocation": self.tool_invocation.dict()
            if self.tool_invocation
            else None,
            "tool_result": self.tool_result.dict() if self.tool_result else None,
        }

    @classmethod
    def from_dict(cls, message_dict: Dict[str, Any]) -> "Message":
        """Create a message from a dictionary."""
        return cls(
            role=message_dict["role"],
            content=message_dict["content"],
            tool_invocation=ToolInvocation.parse_obj(message_dict["tool_invocation"])
            if message_dict.get("tool_invocation")
            else None,
            tool_result=ToolResult.parse_obj(message_dict["tool_result"])
            if message_dict.get("tool_result")
            else None,
        )


class Conversation(BaseModel):
    """A conversation between a user and an agent."""

    messages: List[Message] = Field(default_factory=list)

    def __len__(self):
        """Return the number of messages in the conversation."""
        return len(self.messages)

    def __getitem__(self, index: int) -> Message:
        """Return the message at the given index."""
        return self.messages[index]

    def __setitem__(self, index: int, message: Message):
        """Set the message at the given index."""
        self.messages[index] = message

    def __iter__(self):
        """Return an iterator over the messages in the conversation."""
        return iter(self.messages)

    def append(self, message: Message):
        """Append a message to the conversation."""
        self.messages.append(message)

    def extend(self, messages: List[Message]):
        """Extend the conversation with a list of messages."""
        self.messages.extend(messages)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the conversation."""
        return {"messages": [message.to_dict() for message in self.messages]}

    @classmethod
    def from_dict(cls, conversation_dict: Dict[str, Any]) -> "Conversation":
        """Create a conversation from a dictionary."""
        return cls(
            messages=[
                Message.from_dict(message_dict)
                for message_dict in conversation_dict["messages"]
            ]
        ) 
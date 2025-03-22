from pydantic import Field

from mcp_agent.agent.browser import BrowserAgent
from mcp_agent.mcp_client.tools.base import ToolCollection
from mcp_agent.mcp_client.tools.browser_use_tool import BrowserUseTool
from mcp_agent.mcp_client.tools.python_execute import PythonExecute
from mcp_agent.mcp_client.tools.str_replace_editor import StrReplaceEditor
from mcp_agent.mcp_client.tools.bash import Bash
from mcp_agent.mcp_client.tools.file_saver import FileSaver
from mcp_agent.mcp_client.tools.terminate import Terminate


class Manus(BrowserAgent):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends BrowserAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = """You are a versatile agent that can solve various tasks using multiple tools.
You have access to the following tools:
- Python execution
- Web browsing
- File operations
- Information retrieval

Please be concise and clear in your responses."""

    next_step_prompt: str = """Based on the current state and available tools, what should be the next step?
Consider the user's request and available tools to determine the best course of action."""

    max_observe: int = 10000
    max_steps: int = 20

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            Bash(),
            FileSaver(),
            Terminate()
        )
    )

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        # Store original prompt
        original_prompt = self.next_step_prompt

        # Only check recent messages (last 3) for browser activity
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            "browser_use" in msg.content.lower()
            for msg in recent_messages
            if hasattr(msg, "content") and isinstance(msg.content, str)
        )

        if browser_in_use:
            # Override with browser-specific prompt temporarily to get browser context
            self.next_step_prompt = "Based on the current browser state, what should be the next step?"

        # Call parent's think method
        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result

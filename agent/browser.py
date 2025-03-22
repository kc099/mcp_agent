import json
from typing import Any, Optional, Dict

from pydantic import Field

from mcp_agent.agent.toolcall import ToolCallAgent
from mcp_agent.agent.schema import Message, ToolChoice
from mcp_agent.mcp_client.tools.base import ToolCollection
from mcp_agent.mcp_client.tools.browser_use_tool import BrowserUseTool
from mcp_agent.mcp_client.tools.terminate import Terminate


class BrowserAgent(ToolCallAgent):
    """
    A browser agent that uses browser automation tools to control a browser.

    This agent can navigate web pages, interact with elements, fill forms,
    extract content, and perform other browser-based actions to accomplish tasks.
    """

    name: str = "browser"
    description: str = "A browser agent that can control a browser to accomplish tasks"

    system_prompt: str = """You are a browser agent that can control a browser to accomplish tasks.
You have access to the following browser actions:
- navigate: Navigate to a URL
- click: Click on an element
- type: Type text into an element
- get_text: Get text from an element
- get_links: Get all links from a page

Please be concise and clear in your responses."""

    next_step_prompt: str = """Based on the current state and available tools, what should be the next step?
Consider the user's request and available browser actions to determine the best course of action."""

    max_observe: int = 10000
    max_steps: int = 20

    # Configure the available tools
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            BrowserUseTool(),
            Terminate()
        )
    )

    # Use Auto for tool choice to allow both tool usage and free-form responses
    tool_choices: ToolChoice = ToolChoice.AUTO
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    _current_base64_image: Optional[str] = None

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return
        
        # Get the browser tool if it exists
        browser_tool = self.available_tools.get("browser_use")
        if browser_tool and hasattr(browser_tool, "cleanup"):
            await browser_tool.cleanup()
            
        await super()._handle_special_tool(name, result, **kwargs)

    async def get_browser_state(self) -> Optional[Dict]:
        """Get the current browser state for context in next steps."""
        # Get browser tool
        browser_tool = self.available_tools.get("browser_use")
        if not browser_tool or not hasattr(browser_tool, "get_current_state"):
            return None

        try:
            # Get browser state directly from the tool
            result = await browser_tool.get_current_state()

            if hasattr(result, "error") and result.error:
                print(f"Browser state error: {result.error}")
                return None

            # Store screenshot if available
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image

            # Parse the state info
            if hasattr(result, "output") and result.output:
                return json.loads(result.output)
            return None

        except Exception as e:
            print(f"Failed to get browser state: {str(e)}")
            return None

    async def think(self) -> bool:
        """Process current state and decide next actions using tools, with browser state info added"""
        # Add browser state to the context
        browser_state = await self.get_browser_state()

        # Initialize placeholder values
        url_info = ""
        tabs_info = ""
        content_above_info = ""
        content_below_info = ""
        results_info = ""

        if browser_state and not browser_state.get("error"):
            # URL and title info
            url_info = f"\n   URL: {browser_state.get('url', 'N/A')}\n   Title: {browser_state.get('title', 'N/A')}"

            # Tab information
            if "tabs" in browser_state:
                tabs = browser_state.get("tabs", [])
                if tabs:
                    tabs_info = f"\n   {len(tabs)} tab(s) available"

            # Content above/below viewport
            pixels_above = browser_state.get("pixels_above", 0)
            pixels_below = browser_state.get("pixels_below", 0)

            if pixels_above > 0:
                content_above_info = f" ({pixels_above} pixels)"

            if pixels_below > 0:
                content_below_info = f" ({pixels_below} pixels)"

            # Add screenshot as base64 if available
            if self._current_base64_image:
                # Create a message with image attachment
                image_message = Message.user_message(
                    content="Current browser screenshot:",
                    base64_image=self._current_base64_image,
                )
                self.memory.add_message(image_message)

        # Update next_step_prompt with browser state info
        if self.next_step_prompt:
            browser_context = f"""
Browser State:
{url_info}
{tabs_info}
Content above viewport: {content_above_info}
Content below viewport: {content_below_info}
{results_info}
"""
            self.next_step_prompt = f"{self.next_step_prompt}\n\n{browser_context}"

        # Call parent implementation
        result = await super().think()

        return result

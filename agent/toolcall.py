import json
from typing import Any, List, Optional, Union

from pydantic import Field

from mcp_agent.agent.react import ReActAgent
from mcp_agent.agent.schema import AgentState, Message, ToolCall, ToolChoice
from mcp_agent.mcp_client.tools.base import ToolCollection


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    available_tools: ToolCollection = Field(default_factory=ToolCollection)
    tool_choices: ToolChoice = ToolChoice.AUTO
    special_tool_names: List[str] = Field(default_factory=list)

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages.append(user_msg)

        try:
            # This would be implemented with an actual LLM call
            # For now, we'll just return a placeholder
            # In a real implementation, this would call an LLM to get a response with tool calls
            
            # Placeholder for LLM response
            response = {
                "content": "I'll help you with that task.",
                "tool_calls": []  # This would be populated by the LLM
            }
            
            self.tool_calls = response.get("tool_calls", [])

            # Log response info
            print(f"✨ {self.name}'s thoughts: {response.get('content', '')}")
            print(
                f"🛠️ {self.name} selected {len(self.tool_calls) if self.tool_calls else 0} tools to use"
            )
            
            if self.tool_calls:
                print(
                    f"🧰 Tools being prepared: {[call.function.name for call in self.tool_calls]}"
                )

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if response.get("content"):
                    self.memory.add_message(Message.assistant_message(response.get("content", "")))
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(
                    content=response.get("content"), tool_calls=self.tool_calls
                )
                if self.tool_calls
                else Message.assistant_message(response.get("content", ""))
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(response.get("content"))

            return bool(self.tool_calls)
        except Exception as e:
            print(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError("Tool calls required but none provided")

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            print(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        tool = self.available_tools.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            # Execute the tool
            print(f"🔧 Activating tool: '{name}'...")
            result = await tool(**args)

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image

                # Format result for display
                observation = (
                    f"Observed output of cmd `{name}` executed:\n{str(result)}"
                    if result
                    else f"Cmd `{name}` completed with no output"
                )
                return observation

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            print(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            print(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            print(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

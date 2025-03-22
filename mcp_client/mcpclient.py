import asyncio
import json
import os
import sys
import logging
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field, model_validator

from colorama import Fore, init
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from openai import AsyncOpenAI

# Configure logging to file instead of stdout
logging.basicConfig(
    filename='mcp_client.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)  # Add current directory to Python path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tools.base import ToolCollection, BaseTool, ToolResult
from tools.browser_use_tool import BrowserUseTool
from tools.python_execute import PythonExecute
from tools.str_replace_editor import StrReplaceEditor
from tools.bash import Bash
from tools.file_saver import FileSaver
from tools.terminate import Terminate

# Initialize colorama and rich console
init(autoreset=True)
console = Console()

# Load environment variables from .env file
load_dotenv()

# System prompt for the agent
SYSTEM_PROMPT = """You are a versatile MCP agent that can solve various tasks using multiple tools.
You have access to the following tools:
- Python execution
- Web browsing
- File operations
- Information retrieval
- Bash command execution

Your workspace is located at: {directory}

Please be concise and clear in your responses."""

# Next step prompt for the agent
NEXT_STEP_PROMPT = """Based on the current state and available tools, what should be the next step?
Consider the user's request and available tools to determine the best course of action."""

# Define agent states
class AgentState:
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"

# Define message class for agent memory
class Message(BaseModel):
    role: str
    content: Optional[str] = None
    base64_image: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    
    @classmethod
    def user_message(cls, content: str, base64_image: Optional[str] = None) -> "Message":
        """Create a user message."""
        return cls(role="user", content=content, base64_image=base64_image)
    
    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role="system", content=content)
    
    @classmethod
    def assistant_message(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role="assistant", content=content)
    
    @classmethod
    def tool_message(cls, content: str, tool_call_id: Optional[str] = None, 
                    name: Optional[str] = None, base64_image: Optional[str] = None) -> "Message":
        """Create a tool message."""
        return cls(
            role="tool", 
            content=content,
            base64_image=base64_image,
            tool_call_id=tool_call_id,
            name=name
        )

# Define memory class for agent
class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the memory."""
        self.messages.append(message)

class ManusAgent(BaseModel):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.
    This agent extends the MCP client with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools"
    
    system_prompt: str = SYSTEM_PROMPT.format(directory=current_dir)
    next_step_prompt: str = NEXT_STEP_PROMPT
    
    memory: Memory = Field(default_factory=Memory)
    state: str = AgentState.IDLE
    
    max_observe: int = 10000
    max_steps: int = 20
    current_step: int = 0
    
    available_tools: ToolCollection = Field(default_factory=ToolCollection)
    special_tool_names: List[str] = Field(default_factory=list)
    
    _current_base64_image: Optional[str] = None
    
    # OpenAI client
    openai_client: Optional[AsyncOpenAI] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        """Initialize the agent with OpenAI client and tools."""
        super().__init__(**data)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in .env file")
        
        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            timeout=60.0
        )
        
        # Initialize tools
        self.available_tools = ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            Bash(),
            FileSaver(),
            Terminate()
        )
        
        # Set special tool names
        self.special_tool_names = ["terminate"]
    
    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously."""
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.memory.add_message(Message.user_message(request))

        results: List[str] = []
        self.state = AgentState.RUNNING
        
        try:
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                console.print(f"[dim]Executing step {self.current_step}/{self.max_steps}[/dim]")
                step_result = await self.step()
                console.print(f"[green]Step {self.current_step} result:[/green] {step_result[:200]}...")
                results.append(step_result)

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
        except Exception as e:
            self.state = AgentState.ERROR
            error_msg = f"Error during agent execution: {str(e)}"
            logging.error(error_msg)
            console.print(f"[bold red]Error:[/bold red] {error_msg}")
            results.append(error_msg)
        finally:
            self.state = AgentState.IDLE
            self.current_step = 0
            
        # Get the final response from the assistant
        final_response = ""
        for msg in reversed(self.memory.messages):
            if msg.role == "assistant" and msg.content:
                final_response = msg.content
                break
        
        if not final_response and results:
            final_response = "\n".join(results)
        
        return final_response
    
    async def step(self) -> str:
        """Execute a single step: think and act."""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
    
    async def think(self) -> bool:
        """Process current state and decide next actions using tools."""
        # Check for browser activity
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            "browser_use" in (msg.content or "").lower()
            for msg in recent_messages
        )

        # Prepare user message with next step prompt
        prompt = self.next_step_prompt
        if browser_in_use:
            prompt = "Based on the current browser state, what should be the next step?"
            
            # Get browser state if available
            browser_tool = self.available_tools.get("browser_use")
            if browser_tool and hasattr(browser_tool, "get_current_state"):
                try:
                    result = await browser_tool.get_current_state()
                    if hasattr(result, "base64_image") and result.base64_image:
                        self._current_base64_image = result.base64_image
                        # Add screenshot as base64 if available
                        image_message = Message.user_message(
                            content="Current browser screenshot:",
                            base64_image=self._current_base64_image,
                        )
                        self.memory.add_message(image_message)
                except Exception as e:
                    logging.error(f"Error getting browser state: {str(e)}")

        # Add user message to memory
        self.memory.add_message(Message.user_message(prompt))
        
        try:
            # Prepare messages for OpenAI
            messages = []
            
            # Add system message
            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            
            # Add conversation history
            for msg in self.memory.messages:
                message_dict = {"role": msg.role, "content": msg.content}
                
                # Handle tool messages
                if msg.role == "tool":
                    if msg.tool_call_id:
                        message_dict["tool_call_id"] = msg.tool_call_id
                    if msg.name:
                        message_dict["name"] = msg.name
                
                # Handle messages with images
                if msg.base64_image:
                    message_dict["content"] = [
                        {"type": "text", "text": msg.content or ""},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{msg.base64_image}"
                            }
                        }
                    ]
                
                messages.append(message_dict)
            
            # Get available tools
            available_tools = self.available_tools.to_params()
            
            console.print("[dim]Calling OpenAI API...[/dim]")
            
            # Get response from OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
                timeout=60.0
            )
            
            message = response.choices[0].message
            
            console.print(f"[dim]OpenAI response: {message.content[:100]}...[/dim]")
            
            # Add assistant message to memory
            assistant_message = Message(
                role="assistant",
                content=message.content
            )
            
            # Handle tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                assistant_message.tool_calls = message.tool_calls
                console.print(f"[dim]Tool calls: {len(message.tool_calls)}[/dim]")
            
            self.memory.add_message(assistant_message)
            
            # Return True if there are tool calls or content
            return bool(message.tool_calls or message.content)
        
        except Exception as e:
            error_msg = f"Error in think step: {str(e)}"
            logging.error(error_msg)
            console.print(f"[bold red]Error in think step:[/bold red] {error_msg}")
            self.memory.add_message(Message.assistant_message(error_msg))
            return False
    
    async def act(self) -> str:
        """Execute tool calls and handle their results."""
        # Get the last assistant message
        last_message = next((msg for msg in reversed(self.memory.messages) 
                            if msg.role == "assistant"), None)
        
        if not last_message:
            return "No assistant message found"
        
        # If no tool calls, just return the content
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return last_message.content or "No content or commands to execute"
        
        results = []
        for tool_call in last_message.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None
            
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            
            # Convert tool_args from string to dictionary
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception as e:
                    error_msg = f"Error parsing tool arguments: {str(e)}"
                    logging.error(error_msg)
                    console.print(f"[bold red]Error parsing tool arguments:[/bold red] {error_msg}")
                    results.append(error_msg)
                    continue
            
            # Get the tool
            tool = self.available_tools.get(tool_name)
            if not tool:
                error_msg = f"Tool {tool_name} not found"
                logging.error(error_msg)
                console.print(f"[bold red]Error:[/bold red] {error_msg}")
                results.append(error_msg)
                continue
            
            # Execute the tool
            try:
                console.print(f"[dim]Executing tool: {tool_name}[/dim]")
                result = await tool(**tool_args)
                
                # Handle special tools
                if tool_name.lower() in [name.lower() for name in self.special_tool_names]:
                    console.print(f"[dim]Special tool {tool_name} executed[/dim]")
                    self.state = AgentState.FINISHED
                
                # Format result
                result_str = str(result)
                if self.max_observe and len(result_str) > self.max_observe:
                    result_str = result_str[:self.max_observe] + "... (truncated)"
                
                # Check if result has base64_image
                if hasattr(result, "base64_image") and result.base64_image:
                    self._current_base64_image = result.base64_image
                
                # Add tool message to memory
                self.memory.add_message(Message.tool_message(
                    content=result_str,
                    tool_call_id=tool_call.id,
                    name=tool_name,
                    base64_image=self._current_base64_image
                ))
                
                results.append(f"Tool {tool_name} result: {result_str}")
            
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                logging.error(error_msg)
                console.print(f"[bold red]Error executing tool:[/bold red] {error_msg}")
                
                # Add error message to memory
                self.memory.add_message(Message.tool_message(
                    content=error_msg,
                    tool_call_id=tool_call.id,
                    name=tool_name
                ))
                
                results.append(error_msg)
        
        return "\n\n".join(results)

class MCPClient:
    """
    A versatile MCP agent that uses planning to solve various tasks.
    This agent extends the MCP client with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.conversation_history: List[Dict] = []
        self.tools_info = {}
        
        # Initialize the Manus agent
        self.agent = ManusAgent()

    async def connect_to_server(self, server_script_path: str = None):
        """Connect to the MCP server"""
        if server_script_path is None:
            server_script_path = os.path.join(current_dir, "mcpserver.py")
        
        try:
            server_params = StdioServerParameters(
                command="python", args=[server_script_path], env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()

            # List available tools
            response = await self.session.list_tools()
            self.tools_info = {tool.name: tool for tool in response.tools}
            
            # Clear the screen before showing welcome message
            console.clear()
            console.print(Panel.fit(
                "[bold cyan]🚀 MCP Client Connected![/bold cyan]\n"
                f"[green]Available tools:[/green] {', '.join(self.tools_info.keys())}\n"
                "[yellow]Type your queries or 'quit' to exit.[/yellow]",
                title="MCP Client",
                border_style="blue"
            ))
        except Exception as e:
            console.print(f"[bold red]Error connecting to server:[/bold red] {str(e)}")
            raise

    async def process_query(self, query: str) -> str:
        """Process a query using the Manus agent"""
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            
            console.print("[bold blue]Processing query with Manus agent...[/bold blue]")
            
            # Run the agent
            result = await self.agent.run(query)
            
            # Add assistant response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logging.error(error_msg)
            console.print(f"[bold red]Error processing query:[/bold red] {error_msg}")
            return error_msg

    async def chat_loop(self):
        """Run an interactive chat loop"""
        while True:
            try:
                # Get user input
                user_input = console.input("\n[bold blue]🔍 Query:[/bold blue] ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    console.print("\n[bold red]👋 Goodbye![/bold red]")
                    break

                # Process the query
                console.print()  # Add a blank line for better readability
                response = await self.process_query(user_input)
                
                # Display response
                if response.strip():
                    console.print("\n[bold magenta]💬 Response:[/bold magenta]")
                    console.print(Markdown(response))
                else:
                    console.print("\n[bold yellow]No response received[/bold yellow]")

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
                break
            except Exception as e:
                logging.error(f"Error in chat loop: {str(e)}")
                console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up browser resources
            browser_tool = self.agent.available_tools.get("browser_use")
            if browser_tool and hasattr(browser_tool, "cleanup"):
                await browser_tool.cleanup()
            
            # Close OpenAI client
            if self.agent.openai_client:
                await self.agent.openai_client.close()
            
            # Close MCP session
            await self.exit_stack.aclose()
        except Exception as e:
            console.print(f"[bold red]Error during cleanup:[/bold red] {str(e)}")

    def save_history(self, filename: str = "conversation_history.json"):
        """Save conversation history to a file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            console.print(f"[bold red]Error saving history:[/bold red] {str(e)}")

async def main():
    try:
        if len(sys.argv) > 1:
            server_script = sys.argv[1]
        else:
            server_script = os.path.join(current_dir, "mcpserver.py")

        client = MCPClient()
        try:
            await client.connect_to_server(server_script)
            await client.chat_loop()
        finally:
            await client.cleanup()
            client.save_history()
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

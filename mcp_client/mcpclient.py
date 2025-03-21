import asyncio
import json
import os
import sys
import logging
from contextlib import AsyncExitStack
from typing import Optional, List, Dict
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from pydantic import Field

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
from tools.base import ToolCollection
from tools.browser_use_tool import BrowserUseTool
from tools.python_execute import PythonExecute
from tools.str_replace_editor import StrReplaceEditor
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

Your workspace is located at: {directory}

Please be concise and clear in your responses."""

# Next step prompt for the agent
NEXT_STEP_PROMPT = """Based on the current state and available tools, what should be the next step?
Consider the user's request and available tools to determine the best course of action."""

class MCPClient:
    """
    A versatile MCP agent that uses planning to solve various tasks.
    This agent extends the MCP client with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "MCP Agent"
    description: str = "A versatile agent that can solve various tasks using multiple tools"
    
    system_prompt: str = SYSTEM_PROMPT.format(directory=current_dir)
    next_step_prompt: str = NEXT_STEP_PROMPT
    
    max_observe: int = 10000
    max_steps: int = 20

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.conversation_history: List[Dict] = []
        self.tools_info = {}
        
        # Initialize OpenAI client with API key from .env
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in .env file")
        
        try:
            self.openai_client = AsyncOpenAI(
                api_key=api_key,
                timeout=60.0
            )
        except Exception as e:
            console.print(f"[bold red]Error initializing OpenAI client:[/bold red] {str(e)}")
            raise

        # Initialize tool collection
        self.available_tools: ToolCollection = ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            Terminate()
        )

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

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        # Store original prompt
        original_prompt = self.next_step_prompt

        # Only check recent messages (last 3) for browser activity
        recent_messages = self.conversation_history[-3:] if self.conversation_history else []
        browser_in_use = any(
            "browser_use" in msg.get("content", "").lower()
            for msg in recent_messages
        )

        if browser_in_use:
            # Override with browser-specific prompt temporarily
            self.next_step_prompt = "Based on the current browser state, what should be the next step?"

        # Process the current state
        result = await self.process_query(self.next_step_prompt)

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return bool(result)

    async def process_query(self, query: str) -> str:
        """Process a query using the available tools"""
        try:
            # Prepare messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {"role": "user", "content": query}
            ]

            # Use the tools from our ToolCollection instead of tools_info
            available_tools = self.available_tools.to_params()

            # Get initial response from OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
                timeout=60.0
            )

            final_text = []
            while True:
                message = response.choices[0].message

                # Add assistant's message to conversation
                messages.append({
                    "role": "assistant",
                    "content": message.content if message.content else None,
                    "tool_calls": message.tool_calls if hasattr(message, "tool_calls") else None
                })

                # If no tool calls, we're done
                if not hasattr(message, "tool_calls") or not message.tool_calls:
                    if message.content:
                        final_text.append(message.content)
                    break

                # Handle tool calls
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    # Convert tool_args from string to dictionary if necessary
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except (ValueError, SyntaxError) as e:
                            logging.error(f"Error converting tool_args to dict: {e}")
                            tool_args = {}

                    # Get the tool from our collection
                    tool = self.available_tools.get(tool_name)
                    if tool is None:
                        logging.error(f"Tool {tool_name} not found")
                        continue

                    # Execute tool call
                    try:
                        console.print(f"\n[dim]Using {tool_name}...[/dim]")
                        result = await tool(**tool_args)
                        
                        # Only add meaningful results to the final text
                        if result and str(result).strip():
                            final_text.append(str(result))

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result)
                        })
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        logging.error(error_msg)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg
                        })

                # Get next response from OpenAI
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto",
                    timeout=60.0
                )

            return "\n".join(final_text)
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logging.error(error_msg)
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

                # Add user message to history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })

                # Process the query
                console.print()  # Add a blank line for better readability
                response = await self.process_query(user_input)
                
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })

                # Display response
                if response.strip():
                    console.print("\n[bold magenta]💬 Response:[/bold magenta]")
                    console.print(Markdown(response))

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
                break
            except Exception as e:
                logging.error(f"Error in chat loop: {str(e)}")
                console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.exit_stack.aclose()
            await self.openai_client.close()
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
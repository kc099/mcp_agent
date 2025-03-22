import asyncio
import json
import os
import sys
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from colorama import Fore, init
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

# Configure logging
logging.basicConfig(
    filename='mcp_client.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize colorama and rich console
init(autoreset=True)
console = Console()

# Load environment variables from .env file
load_dotenv()

# Import agent components
from mcp_agent.agent.base import BaseAgent
from mcp_agent.agent.schema import AgentState, Message, Memory
from mcp_agent.agent.manus import Manus

# Import tools
from mcp_agent.mcp_client.tools.base import BaseTool, ToolCollection
from mcp_agent.mcp_client.tools.browser_use_tool import BrowserUseTool
from mcp_agent.mcp_client.tools.python_execute import PythonExecute
from mcp_agent.mcp_client.tools.str_replace_editor import StrReplaceEditor
from mcp_agent.mcp_client.tools.bash import Bash
from mcp_agent.mcp_client.tools.file_saver import FileSaver
from mcp_agent.mcp_client.tools.terminate import Terminate

class MCPClient:
    """
    A versatile MCP agent that uses planning to solve various tasks.
    This agent extends the MCP client with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    def __init__(self):
        """Initialize the MCP client with agent and tools."""
        self.conversation_history = []
        
        # Initialize tools
        self.tools = ToolCollection(
            BrowserUseTool(),
            PythonExecute(),
            StrReplaceEditor(),
            Bash(),
            FileSaver(),
            Terminate()
        )
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Display welcome message
        console.print(Panel.fit(
            "[bold cyan]🚀 MCP Client Connected![/bold cyan]\n"
            f"[green]Available tools:[/green] {', '.join([tool.name for tool in self.tools.tools])}\n"
            "[yellow]Type your queries or 'quit' to exit.[/yellow]",
            title="MCP Client",
            border_style="blue"
        ))

    def _create_agent(self) -> Manus:
        """Create and initialize the Manus agent."""
        # Create the agent
        agent = Manus()
        
        # Set the available tools
        agent.available_tools = self.tools
        
        # Set the special tool names
        agent.special_tool_names = ["terminate"]
        
        return agent

    async def process_query(self, query: str) -> str:
        """Process a query using the agent."""
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            
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
            return error_msg

    async def chat_loop(self):
        """Run an interactive chat loop."""
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

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
                break
            except Exception as e:
                logging.error(f"Error in chat loop: {str(e)}")
                console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")

    def save_history(self, filename: str = "conversation_history.json"):
        """Save conversation history to a file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            console.print(f"[bold red]Error saving history:[/bold red] {str(e)}")

async def main():
    """Main entry point for the MCP client."""
    try:
        client = MCPClient()
        try:
            await client.chat_loop()
        finally:
            client.save_history()
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

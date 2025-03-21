import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Optional, List, Dict
from datetime import datetime
from dotenv import load_dotenv

from colorama import Fore, init
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from openai import AsyncOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Initialize colorama and rich console
init(autoreset=True)
console = Console()

# Load environment variables from .env file
load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.conversation_history: List[Dict] = []
        self.tools_info = {}
        
        # Initialize OpenAI client with API key from .env
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in .env file. Please add OPENAI_API_KEY=your-key-here to your .env file")
        
        self.openai_client = AsyncOpenAI(api_key=api_key)

    async def connect_to_server(self, server_script_path: str = None):
        """Connect to the MCP server"""
        script_path = server_script_path or "mcp/mcpserver.py"
        
        server_params = StdioServerParameters(
            command="python", args=[script_path], env=None
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
        
        console.print(Panel.fit(
            "[bold cyan]🚀 MCP Client Connected![/bold cyan]\n"
            f"[green]Available tools:[/green] {', '.join(self.tools_info.keys())}\n"
            "[yellow]Type your queries or 'quit' to exit.[/yellow]",
            title="MCP Client",
            border_style="blue"
        ))

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

                # Process the query using OpenAI
                response = await self.process_query(user_input)
                
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })

                # Display response
                console.print("\n[bold magenta]💬 Response:[/bold magenta]")
                console.print(Markdown(response))

            except Exception as e:
                console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools"""
        # Prepare messages for OpenAI
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that can use various tools to help users. "
                          "You can browse the web, execute commands, and edit files. "
                          "Please be concise and clear in your responses."
            },
            {"role": "user", "content": query}
        ]

        # Prepare available tools for OpenAI
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            }
            for tool in self.tools_info.values()
        ]

        # Get initial response from OpenAI
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=available_tools,
            tool_choice="auto"
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
                        console.print(f"Error converting tool_args to dict: {e}")
                        tool_args = {}

                # Execute tool call
                console.print(f"Calling tool {tool_name} with args: {tool_args}")
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name}]")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.content)
                })

            # Get next response from OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )

        return "\n".join(final_text)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        await self.openai_client.close()

    def save_history(self, filename: str = "conversation_history.json"):
        """Save conversation history to a file"""
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

async def main():
    if len(sys.argv) > 1:
        server_script = sys.argv[1]
    else:
        server_script = "mcp/mcpserver.py"

    client = MCPClient()
    try:
        await client.connect_to_server(server_script)
        await client.chat_loop()
    finally:
        client.save_history()
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 
import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from mcpclient import MCPClient
from rich.console import Console

# Initialize console
console = Console()

async def test_mcp_client():
    """Test the MCP client with a simple query."""
    client = MCPClient()
    
    # Test a simple query
    query = "Hello, I'm a user. Can you tell me who you are and what you can do?"
    console.print(f"\n[bold blue]Testing query:[/bold blue] {query}")
    
    try:
        # Process the query with the agent
        console.print("[bold green]Processing query with Manus agent...[/bold green]")
        response = await client.process_query(query)
        
        # Display response
        console.print("\n[bold magenta]Response:[/bold magenta]")
        console.print(response)
        
        return True
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return False
    finally:
        # Clean up resources
        await client.cleanup()

if __name__ == "__main__":
    console.print("[bold cyan]Starting MCP Client Test[/bold cyan]")
    success = asyncio.run(test_mcp_client())
    console.print(f"\n[bold {'green' if success else 'red'}]Test {'succeeded' if success else 'failed'}[/bold]")
    sys.exit(0 if success else 1)

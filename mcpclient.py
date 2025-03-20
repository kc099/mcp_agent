# mcp_client.py
import requests
import json
import uuid
import time
from typing import Dict, Any, List
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

class MCPClient:
    """Client for the MCP Server API"""
    
    def __init__(self, server_url="http://localhost:8000"):
        """
        Initialize the MCP client
        
        Args:
            server_url: URL of the MCP server
        """
        self.server_url = server_url
        self.conversation_id = str(uuid.uuid4())
        self.console = Console()
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get information about available tools from the server"""
        try:
            response = requests.get(f"{self.server_url}/api/tools")
            response.raise_for_status()
            return response.json()["tools"]
        except Exception as e:
            self.console.print(f"[bold red]Error getting available tools: {str(e)}[/bold red]")
            return []
    
    def handle_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a tool call by sending it to the server
        
        Args:
            tool_call: Tool call information
            
        Returns:
            Server response
        """
        try:
            # Prepare the request
            request_data = {
                "conversation_id": self.conversation_id,
                "tool_name": tool_call["name"],
                "tool_input": tool_call["input"]
            }
            
            # Send the request to the server
            self.console.print(f"[bold cyan]Calling tool:[/bold cyan] [yellow]{tool_call['name']}[/yellow]")
            response = requests.post(
                f"{self.server_url}/api/tool_call", 
                json=request_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.console.print(f"[bold red]Error calling tool {tool_call['name']}: {str(e)}[/bold red]")
            return {"error": str(e)}
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the MCP server
        
        Args:
            query: User's question
            
        Returns:
            Final response from the server
        """
        try:
            # Send the initial query
            self.console.print(f"\n[bold green]Sending query:[/bold green] {query}")
            
            # Prepare the request
            request_data = {
                "query": query,
                "conversation_id": self.conversation_id,
                "message_history": []  # Could implement history for multi-turn conversations
            }
            
            # Send the request to the server
            response = requests.post(
                f"{self.server_url}/api/mcp", 
                json=request_data
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Check if we need to handle tool calls
            while not response_data.get("done", False):
                # Display thinking if available
                if response_data.get("thinking"):
                    self.console.print(f"[italic dim]{response_data['thinking']}[/italic dim]")
                
                # Process tool calls
                tool_calls = response_data.get("tool_calls", [])
                if not tool_calls:
                    self.console.print("[bold red]Error: Server returned no tool calls but isn't done[/bold red]")
                    break
                
                # Handle the first tool call (could handle multiple in parallel in a more advanced implementation)
                tool_call = tool_calls[0]
                tool_response = self.handle_tool_call(tool_call)
                
                # Update response data
                response_data = tool_response
            
            # Return the final response
            return response_data
        except Exception as e:
            self.console.print(f"[bold red]Error processing query: {str(e)}[/bold red]")
            return {"error": str(e), "response": "There was an error processing your query."}
    
    def display_final_response(self, response_data: Dict[str, Any]):
        """
        Display the final response in a nice format
        
        Args:
            response_data: Response data from the server
        """
        if "error" in response_data:
            self.console.print(Panel(
                f"[bold red]Error:[/bold red] {response_data['error']}",
                title="Error",
                border_style="red"
            ))
            return
        
        # Display the final response
        final_response = response_data.get("response", "No response received")
        
        self.console.print(Panel(
            Markdown(final_response),
            title="Churn Analysis",
            border_style="green"
        ))
    
    def display_churn_metrics(self, metrics: Dict[str, Any]):
        """
        Display churn metrics in a formatted table
        
        Args:
            metrics: Churn metrics from tool response
        """
        if not metrics or "error" in metrics:
            self.console.print("[italic yellow]No churn metrics available[/italic yellow]")
            return
        
        # Overall metrics
        if "overall" in metrics:
            overall = metrics["overall"]
            
            overall_table = Table(title="Overall Churn Metrics", box=box.ROUNDED)
            overall_table.add_column("Metric", style="cyan")
            overall_table.add_column("Value", style="green")
            
            overall_table.add_row("Total Customers", str(overall.get("total_customers", "N/A")))
            overall_table.add_row("Churned Customers", str(overall.get("churned_customers", "N/A")))
            overall_table.add_row(
                "Churn Rate", 
                f"{overall.get('churn_rate', 0):.2f}%" if "churn_rate" in overall else "N/A"
            )
            
            self.console.print(overall_table)
        
        # Segment breakdown
        if "segment_breakdown" in metrics and metrics["segment_breakdown"]:
            segment_table = Table(title="Churn by Segment", box=box.ROUNDED)
            segment_table.add_column("Segment", style="cyan")
            segment_table.add_column("Total", style="blue")
            segment_table.add_column("Churned", style="red")
            segment_table.add_column("Churn Rate", style="green")
            
            for segment, data in metrics["segment_breakdown"].items():
                segment_table.add_row(
                    str(segment),
                    str(data.get("total", "N/A")),
                    str(data.get("churned", "N/A")),
                    f"{data.get('churn_rate', 0):.2f}%" if "churn_rate" in data else "N/A"
                )
            
            self.console.print(segment_table)
        
        # Period breakdown
        if "period_breakdown" in metrics and metrics["period_breakdown"]:
            period_table = Table(title=f"Churn by {metrics.get('period', 'Period')}", box=box.ROUNDED)
            period_table.add_column("Period", style="cyan")
            period_table.add_column("Total", style="blue")
            period_table.add_column("Churned", style="red")
            period_table.add_column("Churn Rate", style="green")
            
            for period, data in metrics["period_breakdown"].items():
                period_table.add_row(
                    str(period),
                    str(data.get("total", "N/A")),
                    str(data.get("churned", "N/A")),
                    f"{data.get('churn_rate', 0):.2f}%" if "churn_rate" in data else "N/A"
                )
            
            self.console.print(period_table)

def interactive_client():
    """Run an interactive MCP client"""
    parser = argparse.ArgumentParser(description="MCP Client for Churn Analysis")
    parser.add_argument("--server", default="http://localhost:8000", help="MCP server URL")
    args = parser.parse_args()
    
    client = MCPClient(server_url=args.server)
    console = Console()
    
    # Show welcome message
    console.print(Panel(
        "[bold]Welcome to the Churn Analysis MCP Client![/bold]\n"
        "Ask questions about customer churn using natural language.\n"
        "Type 'exit' or 'quit' to end the session.",
        title="Churn Analysis Client",
        border_style="green"
    ))
    
    # Get available tools
    tools = client.get_available_tools()
    if tools:
        tools_table = Table(title="Available Tools", box=box.ROUNDED)
        tools_table.add_column("Tool", style="cyan")
        tools_table.add_column("Description", style="green")
        
        for tool in tools:
            tools_table.add_row(tool["name"], tool["description"])
        
        console.print(tools_table)
    
    # Start interactive loop
    while True:
        query = console.input("\n[bold green]Ask about churn:[/bold green] ")
        
        if query.lower() in ["exit", "quit", "q"]:
            console.print("[bold]Goodbye![/bold]")
            break
        
        response = client.process_query(query)
        client.display_final_response(response)

if __name__ == "__main__":
    interactive_client()
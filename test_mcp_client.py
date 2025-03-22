import asyncio
import sys
from mcp_agent.mcpclient import MCPClient

async def test_mcp_client():
    """Test the MCP client with a simple query."""
    client = MCPClient()
    
    # Test a simple query
    query = "Hello, can you help me with a task?"
    print(f"\nTesting query: {query}")
    
    try:
        response = await client.process_query(query)
        print("\nResponse:")
        print(response)
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_mcp_client())
    sys.exit(0 if success else 1)

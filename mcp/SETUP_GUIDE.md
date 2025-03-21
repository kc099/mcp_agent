# MCP Agent Setup Guide

This guide documents how to set up and run the MCP Agent with browser automation capabilities.

## Prerequisites
- Python 3.10+
- Windows 10/11
- UV package manager (install with `pip install uv`)
- Claude for Desktop

## Installation Steps

### 1. Create and Activate Virtual Environment
```bash
# Navigate to mcp_agent directory
cd mcp_agent

# Create virtual environment using UV
uv venv

# Note: On Windows, activation might show an error with .venv\Scripts\activate
# Instead, we'll use the Python executable directly from the virtual environment
```

### 2. Install Dependencies
```bash
# Install all required packages using the virtual environment's Python
.venv\Scripts\python.exe -m uv pip install mcp playwright pydantic colorama openai duckduckgo-search baidusearch docker

# Install Playwright browsers
.venv\Scripts\python.exe -m playwright install
```

### 3. Configuration Setup

#### Create config.toml
Create a file at `mcp_agent/config/config.toml` with the following content:
```toml
[llm]
model = "gpt-4-turbo-preview"
base_url = "https://api.openai.com/v1"
api_key = "your-openai-api-key"  # Replace with your actual API key
max_tokens = 4096
temperature = 0.0
```

#### Claude for Desktop Configuration
Create/modify the file at `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
    "mcpServers": {
        "mcp_agent": {
            "command": "O:/Agents/MCPServers/mcp_agent/.venv/Scripts/python.exe",
            "args": [
                "O:/Agents/MCPServers/mcp_agent/mcpserver.py"
            ]
        }
    }
}
```
Note: Use forward slashes in paths to avoid JSON escaping issues.

## Running the Server

### Running the MCP Server Directly
```bash
# Run the server using the virtual environment's Python
.venv\Scripts\python.exe mcpserver.py
```

## Available Tools

### Browser Automation Tool
The browser automation tool provides the following capabilities:
- Navigate to URLs
- Click elements
- Type text
- Get text content
- Get links from pages

Example usage through Claude:
```python
# Navigate to a URL
{
    "action": "navigate",
    "url": "https://example.com"
}

# Click an element
{
    "action": "click",
    "selector": "#submit-button"
}

# Type text
{
    "action": "type",
    "selector": "#search-input",
    "text": "search query"
}

# Get text content
{
    "action": "get_text",
    "selector": ".article-content"
}

# Get all links
{
    "action": "get_links"
}
```

## Troubleshooting

### Common Issues and Solutions

1. **Virtual Environment Activation Issues**
   - Instead of using `activate`, use the Python executable directly:
   - `.venv\Scripts\python.exe` for running scripts
   - `.venv\Scripts\python.exe -m pip` for pip commands

2. **Module Not Found Errors**
   - Make sure you've installed all dependencies using the virtual environment's Python
   - Verify you're using the correct Python executable from `.venv\Scripts\`
   - Check that all packages are installed correctly

3. **Path Issues**
   - Use forward slashes in JSON configuration files
   - Verify absolute paths in claude_desktop_config.json match your system

### Checking Logs
- MCP server logs: `%APPDATA%\Claude\logs\mcp-server-mcp_agent.log`
- Check for any error messages in the terminal output

## Using with Claude for Desktop

1. Configure the MCP server in Claude for Desktop settings
2. Restart Claude for Desktop
3. Look for the hammer icon indicating available tools
4. Tools should now be available for use in conversations

## Testing the Setup

1. Start the server: `.venv\Scripts\python.exe mcpserver.py`
2. Try a test query like "Visit example.com and get all links"
3. Verify that the browser automation tool is working correctly

## Additional Resources
- [MCP Protocol Documentation](https://modelcontextprotocol.io/docs/tools/debugging)
- [Playwright Documentation](https://playwright.dev/python/docs/intro) 
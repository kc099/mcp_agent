# mcp_server.py
import os
import json
import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv

# Import our RAG tools implementation
from churn_rag_tools import ToolAugmentedRAG

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Initialize the anthropic client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Initialize our RAG system with tools
rag_system = ToolAugmentedRAG(collection_name="customer_data")

# Create FastAPI app
app = FastAPI(title="Churn Analysis MCP Server")

# Add CORS middleware to allow requests from web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class MCPRequest(BaseModel):
    """Request body for MCP API"""
    query: str = Field(..., description="User query about churn analysis")
    conversation_id: str = Field(..., description="Unique conversation identifier")
    message_history: List[Dict[str, Any]] = Field(default=[], description="Previous conversation history")

class ToolCallRequest(BaseModel):
    """Request body for tool call API"""
    conversation_id: str = Field(..., description="Conversation identifier")
    tool_name: str = Field(..., description="Name of the tool to call")
    tool_input: Dict[str, Any] = Field(..., description="Input parameters for the tool")

class MCPResponse(BaseModel):
    """Response from MCP API"""
    conversation_id: str
    response: str
    tool_calls: List[Dict[str, Any]] = []
    thinking: str = ""
    done: bool = False

# Store for ongoing conversations
conversations = {}

# Tool registry
tool_registry = {}

# Register our tools
def register_tools():
    """Register all available tools"""
    tool_definitions = rag_system.define_tools()
    
    for tool_def in tool_definitions:
        tool_registry[tool_def["name"]] = {
            "name": tool_def["name"],
            "description": tool_def["description"],
            "schema": tool_def["input_schema"],
            "function": tool_def["function"]
        }
    
    print(f"Registered {len(tool_registry)} tools: {', '.join(tool_registry.keys())}")

# Format tools for Claude
def format_tools_for_claude():
    """Format tools in the structure Claude expects"""
    claude_tools = []
    
    for name, tool in tool_registry.items():
        claude_tools.append({
            "name": name,
            "description": tool["description"],
            "input_schema": tool["schema"]
        })
    
    return claude_tools

# Helper to convert from Claude's message format to our internal format
def format_claude_messages(messages):
    """Convert Claude message objects to dict format"""
    formatted = []
    for msg in messages:
        if isinstance(msg, MessageParam):
            formatted.append({"role": msg.role, "content": msg.content})
        else:
            formatted.append(msg)
    return formatted

@app.post("/api/mcp", response_model=MCPResponse)
async def process_mcp_request(request: MCPRequest):
    """Process an MCP request"""
    conversation_id = request.conversation_id
    query = request.query
    
    # Get or create conversation history
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            "messages": [
                {
                    "role": "system", 
                    "content": """You are a churn analysis expert with access to tools.
When a user asks about churn, follow these steps:
1. First search for relevant data with search_chromadb
2. Then analyze the data using calculate_churn_rate or aggregate_data
3. Use generate_chart if visualizations would help
4. You can call tools multiple times to gather complete information
5. Explain your findings in a data-driven, actionable way

Always think step by step about which tools to use and in what sequence.
"""
                }
            ],
            "tool_results": {}
        }
    
    # Add user query to conversation
    conversations[conversation_id]["messages"].append(
        {"role": "user", "content": query}
    )
    
    # Get Claude's response with tool choices
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            system="""You are a churn analysis expert with access to tools that help analyze customer data.
Your goal is to provide accurate, data-driven insights about customer churn by using the appropriate tools in sequence.""",
            messages=conversations[conversation_id]["messages"],
            tools=format_tools_for_claude(),
            max_tokens=1024
        )
        
        # Check if Claude wants to use tools
        if hasattr(response.content[0], 'tool_calls') and response.content[0].tool_calls:
            # Claude wants to call tools
            tool_calls = response.content[0].tool_calls
            
            # Add Claude's response to the conversation history
            conversations[conversation_id]["messages"].append({
                "role": "assistant",
                "content": [{
                    "type": "tool_calls",
                    "tool_calls": [{
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": json.loads(tool_call.input)
                    } for tool_call in tool_calls]
                }]
            })
            
            # Return the response with tool calls
            return MCPResponse(
                conversation_id=conversation_id,
                response="",  # No final response yet
                tool_calls=[{
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": json.loads(tool_call.input)
                } for tool_call in tool_calls],
                thinking="I need to gather some information using tools before I can answer.",
                done=False
            )
        else:
            # Claude provided a final answer
            final_response = response.content[0].text
            
            # Add Claude's response to the conversation history
            conversations[conversation_id]["messages"].append({
                "role": "assistant",
                "content": final_response
            })
            
            # Return the final response
            return MCPResponse(
                conversation_id=conversation_id,
                response=final_response,
                tool_calls=[],
                thinking="",
                done=True
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing MCP request: {str(e)}")

@app.post("/api/tool_call", response_model=MCPResponse)
async def process_tool_call(request: ToolCallRequest):
    """Process a tool call and return the result"""
    conversation_id = request.conversation_id
    tool_name = request.tool_name
    tool_input = request.tool_input
    
    # Check if conversation exists
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    
    # Check if tool exists
    if tool_name not in tool_registry:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    
    # Execute the tool
    try:
        tool_function = tool_registry[tool_name]["function"]
        tool_result = tool_function(tool_input)
        
        # Get the last assistant message with tool calls
        last_assistant_idx = None
        for i, msg in enumerate(reversed(conversations[conversation_id]["messages"])):
            if msg["role"] == "assistant" and isinstance(msg["content"], list) and msg["content"][0]["type"] == "tool_calls":
                last_assistant_idx = len(conversations[conversation_id]["messages"]) - i - 1
                break
        
        if last_assistant_idx is None:
            raise HTTPException(status_code=400, detail="No pending tool calls found")
        
        # Find the specific tool call
        tool_call_id = None
        for tool_call in conversations[conversation_id]["messages"][last_assistant_idx]["content"][0]["tool_calls"]:
            if tool_call["name"] == tool_name:
                tool_call_id = tool_call["id"]
                break
        
        if tool_call_id is None:
            raise HTTPException(status_code=400, detail=f"No pending call for tool {tool_name} found")
        
        # Add the tool result to the conversation
        conversations[conversation_id]["messages"].append({
            "role": "user",
            "content": [{
                "type": "tool_results",
                "tool_results": [{
                    "call_id": tool_call_id,
                    "name": tool_name,
                    "output": tool_result
                }]
            }]
        })
        
        # Get Claude's next response
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            system="""You are a churn analysis expert with access to tools that help analyze customer data.
Your goal is to provide accurate, data-driven insights about customer churn by using the appropriate tools in sequence.""",
            messages=conversations[conversation_id]["messages"],
            tools=format_tools_for_claude(),
            max_tokens=1024
        )
        
        # Check if Claude wants to use more tools
        if hasattr(response.content[0], 'tool_calls') and response.content[0].tool_calls:
            # Claude wants to call more tools
            tool_calls = response.content[0].tool_calls
            
            # Add Claude's response to the conversation history
            conversations[conversation_id]["messages"].append({
                "role": "assistant",
                "content": [{
                    "type": "tool_calls",
                    "tool_calls": [{
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": json.loads(tool_call.input)
                    } for tool_call in tool_calls]
                }]
            })
            
            # Return the response with tool calls
            return MCPResponse(
                conversation_id=conversation_id,
                response="",  # No final response yet
                tool_calls=[{
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": json.loads(tool_call.input)
                } for tool_call in tool_calls],
                thinking="I've analyzed that information and need to gather more data.",
                done=False
            )
        else:
            # Claude provided a final answer
            final_response = response.content[0].text
            
            # Add Claude's response to the conversation history
            conversations[conversation_id]["messages"].append({
                "role": "assistant",
                "content": final_response
            })
            
            # Return the final response
            return MCPResponse(
                conversation_id=conversation_id,
                response=final_response,
                tool_calls=[],
                thinking="",
                done=True
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing tool call: {str(e)}")

@app.get("/api/tools")
async def get_available_tools():
    """Get information about available tools"""
    return {
        "tools": [
            {
                "name": name,
                "description": tool["description"],
                "schema": tool["schema"]
            }
            for name, tool in tool_registry.items()
        ]
    }

@app.on_event("startup")
async def startup_event():
    """Runs when the server starts"""
    register_tools()
    print("MCP Server started successfully")

if __name__ == "__main__":
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)
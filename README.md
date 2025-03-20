# MCP Agent with ChromaDB Integration

A powerful tool for converting and querying documents using ChromaDB embeddings with OpenAI and Anthropic Claude integration.

## Features

- Convert Word documents to ChromaDB embeddings
- Convert Excel files to ChromaDB embeddings with schema support
- RAG (Retrieval Augmented Generation) system for document querying
- Tool-augmented RAG for advanced analysis
- FastAPI server with Anthropic Claude integration
- Rich CLI interface for document processing
- Support for custom metadata and document chunking

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Anthropic API key (for Claude integration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kc099/mcp_agent.git
cd mcp_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

### Converting Word Documents

```python
from embeddings.vectordbdoc import main

# Convert a Word document to ChromaDB embeddings
collection, ids = main(
    "path/to/your/document.docx",
    document_metadata={
        "author": "Your Name",
        "title": "Document Title",
        "tags": ["tag1", "tag2"]
    }
)
```

### Converting Excel Files

```python
from embeddings.vectordbexcel import EnterpriseExcelEmbedder

# Initialize the embedder
embedder = EnterpriseExcelEmbedder()

# Analyze Excel and generate schema recommendations
schema_recommendations = embedder.analyze_excel("path/to/your/excel.xlsx")

# Create embeddings using the recommended schema
stats = embedder.embed_excel("path/to/your/excel.xlsx", collection_name_prefix="enterprise")
```

### Running the MCP Server

```bash
python mcpserver.py
```

The server will start on `http://localhost:8000`

### Using the MCP Client

```bash
python mcpclient.py
```

## Project Structure

- `embeddings/`
  - `vectordbdoc.py`: Word document to ChromaDB conversion
  - `vectordbexcel.py`: Excel file to ChromaDB conversion
- `mcpclient.py`: CLI client for interacting with the MCP server
- `mcpserver.py`: FastAPI server with Claude integration
- `rag.py`: RAG system implementation
- `toolaugmentedrag.py`: Tool-augmented RAG system
- `chromadbquery.py`: ChromaDB query utilities

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for the interactive API documentation.


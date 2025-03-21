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
- Python code execution with safety restrictions
- File saving and manipulation capabilities
- Docker support for easy deployment

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Anthropic API key (for Claude integration)
- Docker and Docker Compose (optional, for containerized deployment)

## Installation

### Local Installation

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

### Docker Installation

1. Clone the repository and create `.env` file as described above.

2. Build and run using Docker Compose:
```bash
docker-compose up --build
```

Or run in background:
```bash
docker-compose up -d
```

To stop the container:
```bash
docker-compose down
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

#### Local Run
```bash
python mcpserver.py
```

#### Docker Run
```bash
docker-compose up
```

The server will start on `http://localhost:8000`

### Using the MCP Client

#### Local Run
```bash
python mcpclient.py
```

#### Docker Run
The client is automatically started in the Docker container in interactive mode.

## Project Structure

- `embeddings/`
  - `vectordbdoc.py`: Word document to ChromaDB conversion
  - `vectordbexcel.py`: Excel file to ChromaDB conversion
- `mcp/`
  - `tools/`: MCP tool implementations
    - `python_execute.py`: Safe Python code execution
    - `file_saver.py`: File saving and manipulation
    - `browser_use_tool.py`: Browser interaction
    - `str_replace_editor.py`: String replacement
    - `bash.py`: Shell command execution
    - `terminate.py`: Process termination
- `mcpclient.py`: CLI client for interacting with the MCP server
- `mcpserver.py`: FastAPI server with Claude integration
- `rag.py`: RAG system implementation
- `toolaugmentedrag.py`: Tool-augmented RAG system
- `chromadbquery.py`: ChromaDB query utilities
- `Dockerfile`: Docker container definition
- `docker-compose.yml`: Docker Compose configuration

## Docker Volumes

The Docker setup includes persistent volumes for:
- `/app/config`: Configuration files
- `/app/data`: Application data
- `/app/chroma_db`: ChromaDB database files

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for the interactive API documentation.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

# Building a Tool-Augmented RAG System with Claude MCP or OpenManus
Instead of directly connecting an LLM to your ChromaDB, you can build a more sophisticated system that uses tools to iteratively analyze and process data. This approach combines the benefits of RAG with the multi-step reasoning capabilities of agents.

# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the MCP code
COPY mcp/ mcp/
COPY mcpclient.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create config directory and volume mount points
RUN mkdir -p /app/data /app/config /app/chroma_db

# Default command to run the MCP client
CMD ["python", "mcpclient.py"]

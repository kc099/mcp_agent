import os
import uuid
import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_docx(file_path):
    """Extract text content from a Word document"""
    return docx2txt.process(file_path)

def split_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into smaller chunks for better embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def store_in_chromadb(chunks, metadata, collection_name="documents"):
    """Store text chunks and their embeddings in ChromaDB"""
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")

    # Create OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small"
    )

    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )

    # Prepare data for insertion
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Create metadata for each chunk
    metadatas = []
    for i, chunk in enumerate(chunks):
        # Copy the base metadata and add chunk-specific information
        chunk_metadata = metadata.copy()
        
        # Convert metadata values to ChromaDB compatible types
        for key, value in chunk_metadata.items():
            if isinstance(value, (list, tuple)):
                # Convert lists/tuples to comma-separated strings
                chunk_metadata[key] = ','.join(str(v) for v in value)
            elif value is not None:
                # Convert all other values to strings
                chunk_metadata[key] = str(value)
        
        # Add chunk-specific information
        chunk_metadata["chunk_index"] = str(i)
        chunk_metadata["chunk_total"] = str(len(chunks))
        chunk_metadata["chunk_text_preview"] = chunk[:100] + "..." if len(chunk) > 100 else chunk
        metadatas.append(chunk_metadata)
    
    # Add to collection
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection, ids

def main(docx_path, document_metadata=None):
    # Default metadata if none provided
    if document_metadata is None:
        document_metadata = {}
    
    # Extract document filename for metadata
    filename = os.path.basename(docx_path)
    
    # Basic metadata about the document
    base_metadata = {
        "source": filename,
        "document_type": "docx",
        "file_path": docx_path,
        "timestamp": datetime.datetime.now().isoformat(),
        "chunk_size": "500",
        "chunk_overlap": "50",
        **document_metadata  # Add any custom metadata passed in
    }
    
    try:
        # Extract text from Word document
        text = extract_text_from_docx(docx_path)
        
        # Split text into chunks
        chunks = split_text(text)
        
        # Store in ChromaDB
        collection, ids = store_in_chromadb(chunks, base_metadata)
        
        print(f"Successfully stored {len(chunks)} chunks from {filename} in ChromaDB")
        print(f"Total tokens: approximately {sum(len(chunk.split()) for chunk in chunks)}")
        
        return collection, ids
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    docx_path = "SemanticRepresentation_Ecommerce.docx"  # Change to your document path
    
    # Example custom metadata
    custom_metadata = {
        "author": "Krishna Chaitanya",
        "title": "Semantic Representation for E-commerce",
        "department": "Research",
        "tags": "ecommerce,semantic,documentation",  # Changed to comma-separated string
        "version": "1.0",
        "status": "draft",
        "priority": "medium",
        "created_date": datetime.datetime.now().strftime("%Y-%m-%d")
    }
    
    collection, ids = main(docx_path, custom_metadata)
    
    # Test query to verify storage
    results = collection.query(
        query_texts=["What is this document about?"],
        n_results=2
    )
    
    print("\nSample query results:")
    print(results)
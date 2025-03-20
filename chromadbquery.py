import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup ChromaDB client and embedding function
client = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)

# Get your collection (assuming "customer_data" is your collection name)
collection = client.get_collection(
    name="customer_data", 
    embedding_function=openai_ef
)

# Example 1: Simple semantic search
def simple_search(query_text, n_results=5):
    """Basic semantic search in ChromaDB"""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results

# Example 2: Search with metadata filtering
def filtered_search(query_text, filter_condition, n_results=5):
    """Search with metadata filters"""
    results = collection.query(
        query_texts=[query_text],
        where=filter_condition,  # Example: {"customer_segment": "Enterprise"}
        n_results=n_results
    )
    return results

# Example 3: Search for customers who churned
def search_churned_customers(query_text="customer churn information", n_results=10):
    """Find information about churned customers"""
    results = collection.query(
        query_texts=[query_text],
        where={"status": "churned"},  # Assuming you have a status field
        n_results=n_results
    )
    return results

# Example 4: Get data for specific date ranges
def search_by_date_range(query_text, start_date, end_date, n_results=10):
    """Search within a specific date range"""
    # Note: Dates in ChromaDB need to be strings in a format that allows comparison
    results = collection.query(
        query_texts=[query_text],
        where={
            "date": {
                "$gte": start_date,  # Greater than or equal to start_date
                "$lte": end_date     # Less than or equal to end_date
            }
        },
        n_results=n_results
    )
    return results

# Example 5: Advanced query with multiple filters
def advanced_filtered_search(query_text, filters, n_results=5):
    """Search with complex metadata filters"""
    results = collection.query(
        query_texts=[query_text],
        where=filters,  # Example: {"$and": [{"region": "North America"}, {"customer_segment": "Enterprise"}]}
        n_results=n_results
    )
    return results

# Example usage
if __name__ == "__main__":
    # Simple search
    print("Simple search results:")
    simple_results = simple_search("customer retention strategies")
    
    # Filtered search
    print("\nFiltered search results:")
    filtered_results = filtered_search(
        "customers at risk of churning",
        {"customer_segment": "Enterprise"}
    )
    
    # Date range search
    print("\nDate range search results:")
    date_results = search_by_date_range(
        "churn indicators",
        "2024-01-01",
        "2024-03-31"
    )
    
    # Print results nicely
    def print_results(results):
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"\nResult {i+1}:")
            print(f"Document: {doc[:100]}..." if len(doc) > 100 else f"Document: {doc}")
            print(f"Metadata: {metadata}")
            if "distances" in results:
                print(f"Relevance: {results['distances'][0][i]}")
    
    print_results(simple_results)
import os
import json
import pandas as pd
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class ChurnRateRAG:
    """RAG system for churn rate analysis using ChromaDB and LLMs"""
    
    def __init__(self, collection_name="customer_data", model="gpt-4-turbo"):
        """
        Initialize the RAG system
        
        Args:
            collection_name: Name of the ChromaDB collection
            model: LLM model to use for generation
        """
        # Set up ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"
        )
        
        # Get the collection
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        # Set up OpenAI client
        self.llm_client = OpenAI(api_key=openai_api_key)
        self.model = model
    
    def retrieve_relevant_data(self, query, n_results=10, filters=None):
        """
        Retrieve relevant data from ChromaDB
        
        Args:
            query: User query or question
            n_results: Number of results to retrieve
            filters: Metadata filters to apply
            
        Returns:
            ChromaDB query results
        """
        # Build query parameters
        query_params = {
            "query_texts": [query],
            "n_results": n_results
        }
        
        # Add filters if provided
        if filters:
            query_params["where"] = filters
        
        # Execute query
        results = self.collection.query(**query_params)
        return results
    
    def extract_churn_data(self, results):
        """
        Extract churn-related data from retrieved documents
        
        Args:
            results: ChromaDB query results
            
        Returns:
            Formatted context with relevant churn data
        """
        context = []
        
        # Process each document
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            # Create context entry
            entry = {
                "content": doc,
                "metadata": metadata
            }
            context.append(entry)
        
        return context
    
    def generate_churn_analysis(self, user_query, context):
        """
        Generate churn analysis using LLM
        
        Args:
            user_query: Original user query
            context: Retrieved context data
            
        Returns:
            LLM-generated churn analysis
        """
        # Format context for the LLM
        formatted_context = "\n\n".join([
            f"DOCUMENT {i+1}:\nContent: {item['content']}\nMetadata: {json.dumps(item['metadata'], indent=2)}"
            for i, item in enumerate(context)
        ])
        
        # Build the prompt
        prompt = f"""
You are a churn analysis expert. Use the provided customer data to answer the query.

USER QUERY: {user_query}

RETRIEVED DATA:
{formatted_context}

Task:
1. Analyze the retrieved data to extract relevant churn information.
2. If the data contains customer signup and cancellation dates, calculate the churn rate.
3. If the data contains customer segments or cohorts, break down the analysis by these groups.
4. Identify patterns or factors that might contribute to churn.
5. Provide actionable insights based on your analysis.

Your analysis should be data-driven and specific to the retrieved information. If the data is insufficient to answer certain aspects, clearly state what additional data would be needed.
"""
        
        # Generate response using OpenAI
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a churn analysis expert assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    def calculate_churn_rate(self, results):
        """
        Calculate churn rate from retrieved data
        
        This function extracts dates and status information from the retrieved
        documents and calculates the churn rate.
        
        Args:
            results: ChromaDB query results
            
        Returns:
            Dictionary with churn calculations
        """
        # Extract data for churn calculation
        customers = []
        
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            customer_data = {
                "id": metadata.get("customer_id", f"unknown_{i}"),
                "status": metadata.get("status", "unknown"),
                "start_date": metadata.get("start_date", None),
                "end_date": metadata.get("end_date", None),
                "segment": metadata.get("segment", "unknown")
            }
            customers.append(customer_data)
        
        # Count total, active and churned customers
        total_customers = len(customers)
        churned_customers = sum(1 for c in customers if c["status"].lower() in ["churned", "cancelled", "terminated"])
        
        # Calculate basic churn rate
        churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0
        
        # Group by segment if available
        segment_churn = {}
        segments = set(c["segment"] for c in customers if c["segment"] != "unknown")
        
        for segment in segments:
            segment_customers = [c for c in customers if c["segment"] == segment]
            segment_total = len(segment_customers)
            segment_churned = sum(1 for c in segment_customers if c["status"].lower() in ["churned", "cancelled", "terminated"])
            segment_churn[segment] = {
                "total": segment_total,
                "churned": segment_churned,
                "churn_rate": (segment_churned / segment_total) * 100 if segment_total > 0 else 0
            }
        
        return {
            "total_customers": total_customers,
            "churned_customers": churned_customers,
            "churn_rate": churn_rate,
            "segment_analysis": segment_churn
        }
    
    def process_churn_query(self, user_query):
        """
        Process a user query about churn rates
        
        This method ties together retrieval, calculation, and generation
        
        Args:
            user_query: User's question about churn
            
        Returns:
            Generated response with churn analysis
        """
        # 1. Retrieve relevant data
        results = self.retrieve_relevant_data(user_query, n_results=15)
        
        # 2. Extract context
        context = self.extract_churn_data(results)
        
        # 3. Calculate churn metrics where possible
        try:
            churn_metrics = self.calculate_churn_rate(results)
            # Add calculations to context
            context.append({
                "content": f"Calculated churn metrics: {json.dumps(churn_metrics, indent=2)}",
                "metadata": {"source": "calculation"}
            })
        except Exception as e:
            print(f"Error calculating churn metrics: {str(e)}")
            # Continue without metrics
            
        # 4. Generate analysis
        analysis = self.generate_churn_analysis(user_query, context)
        
        return {
            "query": user_query,
            "retrieved_count": len(results["documents"][0]),
            "analysis": analysis
        }


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    churn_rag = ChurnRateRAG(collection_name="customer_data")
    
    # Example queries
    example_queries = [
        "Calculate our overall customer churn rate",
        "What is the churn rate for enterprise customers compared to SMB?",
        "Identify the top factors contributing to customer churn",
        "Calculate monthly churn rate trend over the past year",
        "Which customer segments have the highest churn risk?"
    ]
    
    # Process a sample query
    sample_query = example_queries[0]
    print(f"Processing query: '{sample_query}'")
    
    response = churn_rag.process_churn_query(sample_query)
    
    print("\nRAG Analysis Result:")
    print("-" * 50)
    print(response["analysis"])
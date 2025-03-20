# churn_rag_tools.py
import os
import json
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

class ToolAugmentedRAG:
    """RAG system with tool use capabilities for churn analysis"""
    
    def __init__(self, collection_name="customer_data"):
        """Initialize the tool-augmented RAG system"""
        # Set up ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        
        try:
            # Get collection
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Connected to collection: {collection_name}")
        except Exception as e:
            print(f"Warning: Could not connect to collection {collection_name}. Error: {str(e)}")
            print("Creating a new collection instead.")
            # Create a new collection if it doesn't exist
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        
        # Initialize cached data
        self.cached_data = None
        self.last_query = None
        
    def define_tools(self):
        """Define the tools available to the agent"""
        tools = [
            {
                "name": "search_chromadb",
                "description": "Search ChromaDB for relevant customer data using semantic search",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "The search query to find relevant documents"
                        },
                        "filters": {
                            "type": "object",
                            "description": "Metadata filters to apply (optional)"
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to retrieve (default: 10)"
                        }
                    },
                    "required": ["query"]
                },
                "function": self.search_tool
            },
            {
                "name": "aggregate_data",
                "description": "Aggregate retrieved data by a specific dimension to calculate metrics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "group_by": {
                            "type": "string",
                            "description": "Field to group by (e.g., 'segment', 'month')"
                        },
                        "metric": {
                            "type": "string",
                            "description": "Metric to calculate (e.g., 'churn_rate', 'count')"
                        },
                        "filters": {
                            "type": "object",
                            "description": "Filters to apply before aggregation (optional)"
                        }
                    },
                    "required": ["group_by", "metric"]
                },
                "function": self.aggregate_tool
            },
            {
                "name": "generate_chart",
                "description": "Generate a chart specification based on aggregated data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "description": "Type of chart (e.g., 'bar', 'line', 'pie')"
                        },
                        "x_axis": {
                            "type": "string",
                            "description": "Field for x-axis"
                        },
                        "y_axis": {
                            "type": "string",
                            "description": "Field for y-axis"
                        },
                        "title": {
                            "type": "string",
                            "description": "Chart title (optional)"
                        },
                        "filters": {
                            "type": "object",
                            "description": "Filters to apply to data (optional)"
                        }
                    },
                    "required": ["chart_type", "x_axis", "y_axis"]
                },
                "function": self.chart_tool
            },
            {
                "name": "calculate_churn_rate",
                "description": "Calculate churn rate and related metrics from retrieved customer data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "object",
                            "description": "Filters to apply to the data (optional)"
                        },
                        "period": {
                            "type": "string",
                            "description": "Time period to analyze (e.g., 'monthly', 'quarterly', 'overall')"
                        }
                    },
                    "required": []
                },
                "function": self.churn_calculation_tool
            }
        ]
        return tools
    
    def search_tool(self, params: dict) -> dict:
        """
        Tool to search ChromaDB
        
        Args:
            params: Search parameters including query, filters, and n_results
            
        Returns:
            Dictionary with search results
        """
        # Parse parameters
        query = params["query"]
        filters = params.get("filters", {})
        n_results = params.get("n_results", 10)
        
        # Execute query
        query_params = {
            "query_texts": [query],
            "n_results": n_results
        }
        
        if filters:
            query_params["where"] = filters
        
        try:
            # Execute query
            results = self.collection.query(**query_params)
            
            # Cache results for other tools
            self.cached_data = results
            self.last_query = query
            
            # Format results
            formatted_results = []
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                formatted_results.append({
                    "document": doc,
                    "metadata": metadata,
                    "relevance": results["distances"][0][i] if "distances" in results else None
                })
            
            return {
                "query": query,
                "filters_applied": filters,
                "results_count": len(formatted_results),
                "results": formatted_results[:5],  # Truncate to first 5 for readability
                "message": f"Found {len(formatted_results)} documents matching the query."
            }
        except Exception as e:
            return {
                "error": f"Error executing search: {str(e)}",
                "query": query,
                "results_count": 0,
                "results": []
            }
    
    def aggregate_tool(self, params: dict) -> dict:
        """
        Tool to aggregate data from search results
        
        Args:
            params: Aggregation parameters
            
        Returns:
            Aggregated metrics
        """
        if not self.cached_data:
            return {"error": "No data available. Run a search first."}
        
        # Extract parameters
        group_by = params["group_by"]
        metric = params["metric"]
        filters = params.get("filters", {})
        
        try:
            # Collect data from search results
            data = []
            for i, metadata in enumerate(self.cached_data["metadatas"][0]):
                # Check if group_by field exists
                if group_by not in metadata:
                    continue
                    
                # Apply filters if any
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key in metadata and metadata[key] != value:
                            skip = True
                            break
                    if skip:
                        continue
                
                # Add to data collection
                data.append(metadata)
            
            # Convert to DataFrame for easier aggregation
            df = pd.DataFrame(data)
            
            # Handle case where group_by field doesn't exist in any document
            if group_by not in df.columns:
                return {
                    "error": f"Field '{group_by}' not found in any of the retrieved documents.",
                    "available_fields": list(df.columns) if not df.empty else []
                }
            
            # Group by the specified field
            grouped = df.groupby(group_by)
            
            # Calculate metrics
            if metric == "count":
                result = grouped.size().to_dict()
            elif metric == "churn_rate" and "status" in df.columns:
                # Calculate churn rate for each group
                result = {}
                for name, group in grouped:
                    churned = group[group["status"].isin(["churned", "cancelled", "terminated"])].shape[0]
                    total = group.shape[0]
                    result[name] = (churned / total * 100) if total > 0 else 0
            else:
                # For other numeric fields, calculate average
                if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                    result = grouped[metric].mean().to_dict()
                else:
                    return {"error": f"Cannot calculate metric '{metric}'. Not a numeric field or not found."}
            
            return {
                "aggregation_field": group_by,
                "metric": metric,
                "filters_applied": filters,
                "results": result
            }
        except Exception as e:
            return {
                "error": f"Error aggregating data: {str(e)}",
                "group_by": group_by,
                "metric": metric
            }
    
    def chart_tool(self, params: dict) -> dict:
        """
        Tool to generate chart specifications based on data
        
        Args:
            params: Chart parameters
            
        Returns:
            Chart specification (which would be rendered in a real implementation)
        """
        if not self.cached_data:
            return {"error": "No data available. Run a search first."}
        
        chart_type = params["chart_type"]
        x_axis = params["x_axis"]
        y_axis = params["y_axis"]
        title = params.get("title", "Chart")
        filters = params.get("filters", {})
        
        try:
            # Collect data from search results
            data = []
            for i, metadata in enumerate(self.cached_data["metadatas"][0]):
                # Apply filters if any
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key in metadata and metadata[key] != value:
                            skip = True
                            break
                    if skip:
                        continue
                
                # Check if both axes fields exist
                if x_axis in metadata and y_axis in metadata:
                    data.append({
                        x_axis: metadata[x_axis],
                        y_axis: metadata[y_axis]
                    })
            
            # This would generate a chart specification
            # In a real implementation, this would create a visualization
            
            return {
                "chart_spec": {
                    "type": chart_type,
                    "title": title,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "filters": filters,
                    "data_points": len(data),
                    "preview": data[:5],  # First 5 data points
                    "message": f"Generated a {chart_type} chart with {x_axis} on x-axis and {y_axis} on y-axis based on {len(data)} data points"
                }
            }
        except Exception as e:
            return {
                "error": f"Error generating chart: {str(e)}",
                "chart_type": chart_type,
                "x_axis": x_axis,
                "y_axis": y_axis
            }
    
    def churn_calculation_tool(self, params: dict) -> dict:
        """
        Tool to calculate churn metrics from retrieved data
        
        Args:
            params: Parameters for churn calculation
            
        Returns:
            Churn metrics
        """
        if not self.cached_data:
            return {"error": "No data available. Run a search first."}
        
        filters = params.get("filters", {})
        period = params.get("period", "overall")
        
        try:
            # Extract data
            data = []
            for i, (doc, metadata) in enumerate(zip(self.cached_data["documents"][0], self.cached_data["metadatas"][0])):
                # Apply filters if any
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key in metadata and metadata[key] != value:
                            skip = True
                            break
                    if skip:
                        continue
                
                data.append(metadata)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Handle empty dataframe
            if df.empty:
                return {
                    "error": "No data available after applying filters.",
                    "filters": filters
                }
            
            # Check if status column exists
            if 'status' not in df.columns:
                return {
                    "error": "Cannot calculate churn rate: 'status' field not found in data",
                    "available_fields": list(df.columns)
                }
            
            # Basic churn calculation
            total_customers = len(df)
            churned_customers = df[df['status'].isin(['churned', 'cancelled', 'terminated'])].shape[0]
            churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
            
            # Period-based calculation
            period_breakdown = {}
            
            if period == "monthly" and "date" in df.columns:
                # Ensure date is datetime
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Group by month
                monthly = df.groupby(df['date'].dt.strftime('%Y-%m'))
                
                for month, group in monthly:
                    month_total = group.shape[0]
                    month_churned = group[group['status'].isin(['churned', 'cancelled', 'terminated'])].shape[0]
                    period_breakdown[month] = {
                        "total": month_total,
                        "churned": month_churned,
                        "churn_rate": (month_churned / month_total * 100) if month_total > 0 else 0
                    }
            
            # Segment breakdown if available
            segment_breakdown = {}
            
            if "segment" in df.columns:
                segments = df.groupby("segment")
                
                for segment, group in segments:
                    segment_total = group.shape[0]
                    segment_churned = group[group['status'].isin(['churned', 'cancelled', 'terminated'])].shape[0]
                    segment_breakdown[segment] = {
                        "total": segment_total,
                        "churned": segment_churned,
                        "churn_rate": (segment_churned / segment_total * 100) if segment_total > 0 else 0
                    }
            
            return {
                "overall": {
                    "total_customers": total_customers,
                    "churned_customers": churned_customers,
                    "churn_rate": churn_rate
                },
                "period_breakdown": period_breakdown if period_breakdown else None,
                "segment_breakdown": segment_breakdown if segment_breakdown else None,
                "filters_applied": filters,
                "period": period
            }
        except Exception as e:
            return {
                "error": f"Error calculating churn metrics: {str(e)}",
                "filters": filters,
                "period": period
            }

# For testing
if __name__ == "__main__":
    # Initialize the RAG system
    rag = ToolAugmentedRAG(collection_name="customer_data")
    
    # List available tools
    tools = rag.define_tools()
    print(f"Available tools: {[tool['name'] for tool in tools]}")
    
    # Test the search tool
    search_result = rag.search_tool({
        "query": "customer churn enterprise segment",
        "n_results": 5
    })
    
    print("\nSearch result:")
    print(json.dumps(search_result, indent=2))
    
    # If search was successful, test churn calculation
    if not search_result.get("error"):
        churn_result = rag.churn_calculation_tool({
            "period": "overall"
        })
        
        print("\nChurn calculation result:")
        print(json.dumps(churn_result, indent=2))
import os
import uuid
import pandas as pd
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("excel_embeddings.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("excel_embeddings")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

class EnterpriseExcelEmbedder:
    """Enterprise-grade tool for converting Excel data to ChromaDB embeddings"""
    
    def __init__(self, schema_path=None, embedding_model="text-embedding-3-small"):
        """
        Initialize the Excel embedder
        
        Args:
            schema_path: Path to JSON schema file defining sheet and column mappings
            embedding_model: OpenAI embedding model to use
        """
        self.embedding_model = embedding_model
        self.schema = self._load_schema(schema_path)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=self.embedding_model
        )
        
        # Tracking metrics
        self.processed_sheets = 0
        self.processed_rows = 0
        self.created_embeddings = 0
        self.start_time = None
        
    def _load_schema(self, schema_path):
        """Load schema from JSON file or use default schema"""
        if schema_path and os.path.exists(schema_path):
            logger.info(f"Loading schema from {schema_path}")
            with open(schema_path, 'r') as f:
                return json.load(f)
        
        logger.warning("No schema file provided or file not found. Using default schema.")
        return {
            "_default_": {
                "text_columns": None,  # Will use all non-metadata columns
                "metadata_columns": None,  # Will use basic metadata
                "chunk_size": None  # Each row as separate document
            }
        }
    
    def _get_sheet_schema(self, sheet_name):
        """Get schema for a specific sheet or use default"""
        if sheet_name in self.schema:
            return self.schema[sheet_name]
        elif "_default_" in self.schema:
            logger.info(f"No specific schema for sheet '{sheet_name}'. Using default schema.")
            return self.schema["_default_"]
        else:
            logger.warning(f"No schema found for sheet '{sheet_name}' and no default schema.")
            return {"text_columns": None, "metadata_columns": None, "chunk_size": None}
    
    def analyze_excel(self, excel_path):
        """
        Analyze Excel file and generate schema recommendations
        
        Returns a dictionary with recommended schema for each sheet
        """
        logger.info(f"Analyzing Excel file: {excel_path}")
        recommended_schema = {}
        
        try:
            excel_file = pd.ExcelFile(excel_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                logger.info(f"Analyzing sheet '{sheet_name}' with {df.shape[0]} rows and {df.shape[1]} columns")
                
                # Basic type analysis
                column_types = {}
                text_columns = []
                metadata_columns = []
                
                for column in df.columns:
                    # Sample first 100 non-null values
                    sample = df[column].dropna().head(100)
                    
                    # Skip empty columns
                    if len(sample) == 0:
                        column_types[column] = "empty"
                        continue
                    
                    # Check data type
                    dtype = str(df[column].dtype)
                    
                    # Check if it's likely an ID column
                    is_id = any(id_term in column.lower() for id_term in ['id', 'code', 'sku', 'number'])
                    
                    # Check if it's a numeric column
                    is_numeric = 'int' in dtype or 'float' in dtype
                    
                    # Check if it's a date column
                    is_date = 'datetime' in dtype or 'date' in column.lower()
                    
                    # Check if it's a likely text column (calculate average text length)
                    avg_str_len = 0
                    if dtype == 'object':
                        sample_str = sample.astype(str)
                        avg_str_len = sample_str.str.len().mean()
                    
                    # Classify the column
                    if is_id or is_numeric or is_date or avg_str_len < 20:
                        metadata_columns.append(column)
                        column_types[column] = "metadata"
                    else:
                        text_columns.append(column)
                        column_types[column] = "text"
                
                recommended_schema[sheet_name] = {
                    "text_columns": text_columns,
                    "metadata_columns": metadata_columns,
                    "column_types": column_types,
                    "rows": df.shape[0],
                    "chunk_size": None  # Default to row-by-row
                }
                
                # Recommend chunking for sheets with many rows but little text
                text_cols_len = len(text_columns)
                if df.shape[0] > 1000 and text_cols_len <= 2:
                    recommended_schema[sheet_name]["chunk_size"] = 10
                    logger.info(f"Recommending chunking for sheet '{sheet_name}' with chunk size 10")
            
            return recommended_schema
            
        except Exception as e:
            logger.error(f"Error analyzing Excel file: {str(e)}")
            raise
    
    def embed_excel(self, excel_path, collection_name_prefix="excel"):
        """
        Process Excel file and create embeddings in ChromaDB
        
        Args:
            excel_path: Path to Excel file
            collection_name_prefix: Prefix for collection names
        """
        logger.info(f"Processing Excel file: {excel_path}")
        self.start_time = datetime.now()
        
        try:
            excel_file = pd.ExcelFile(excel_path)
            
            for sheet_name in excel_file.sheet_names:
                # Get schema for this sheet
                schema = self._get_sheet_schema(sheet_name)
                
                # Create collection name - prefix_sheetname
                clean_sheet_name = sheet_name.replace(" ", "_").lower()
                collection_name = f"{collection_name_prefix}_{clean_sheet_name}"
                
                logger.info(f"Processing sheet '{sheet_name}' into collection '{collection_name}'")
                
                # Create or get collection
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"source": os.path.basename(excel_path),
                              "sheet": sheet_name,
                              "created_at": str(datetime.now()),
                              "embedding_model": self.embedding_model}
                )
                
                # Process the sheet
                self._process_sheet(
                    excel_file, 
                    sheet_name, 
                    collection, 
                    text_columns=schema.get("text_columns"),
                    metadata_columns=schema.get("metadata_columns"),
                    chunk_size=schema.get("chunk_size")
                )
                
                self.processed_sheets += 1
            
            # Log completion stats
            duration = datetime.now() - self.start_time
            logger.info(f"Completed processing {self.processed_sheets} sheets with "
                       f"{self.processed_rows} rows into {self.created_embeddings} embeddings "
                       f"in {duration.total_seconds():.1f} seconds")
            
            return {
                "processed_sheets": self.processed_sheets,
                "processed_rows": self.processed_rows,
                "created_embeddings": self.created_embeddings,
                "duration_seconds": duration.total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            raise
    
    def _process_sheet(self, excel_file, sheet_name, collection, text_columns=None, 
                      metadata_columns=None, chunk_size=None):
        """Process a single sheet from an Excel file"""
        
        # Read the sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        logger.info(f"Sheet '{sheet_name}' has {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Handle text and metadata columns
        if text_columns is None:
            if metadata_columns is None:
                # If neither is specified, use all columns for text
                text_columns = df.columns.tolist()
                metadata_columns = []
            else:
                # If only metadata columns specified, use remaining columns for text
                text_columns = [col for col in df.columns if col not in metadata_columns]
        elif metadata_columns is None:
            # If only text columns specified, use remaining columns for metadata
            metadata_columns = [col for col in df.columns if col not in text_columns]
        
        logger.info(f"Using {len(text_columns)} columns for text and {len(metadata_columns)} columns for metadata")
        logger.debug(f"Text columns: {text_columns}")
        logger.debug(f"Metadata columns: {metadata_columns}")
        
        # Fill NaN values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("")
            else:
                df[col] = df[col].fillna(0)
        
        # Chunking strategy
        if chunk_size is None:
            # Process row by row
            self._process_by_row(df, collection, text_columns, metadata_columns, sheet_name)
        else:
            # Process by chunks
            self._process_by_chunks(df, collection, text_columns, metadata_columns, sheet_name, chunk_size)
    
    def _process_by_row(self, df, collection, text_columns, metadata_columns, sheet_name):
        """Process Excel data row by row"""
        batch_size = 100  # Add in batches for efficiency
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Create document from text columns
            if len(text_columns) == 1:
                # If only one text column, use it directly
                document = str(row[text_columns[0]])
            else:
                # If multiple text columns, combine them with labels
                document = "\n".join([f"{col}: {row[col]}" for col in text_columns])
            
            # Create metadata
            metadata = {
                "row_index": int(idx),
                "document_created": str(datetime.now())
            }
            
            # Add specified metadata columns
            for col in metadata_columns:
                # Convert to string to ensure compatibility
                metadata[col] = str(row[col])
            
            # Generate ID
            doc_id = f"{sheet_name}_{idx}_{uuid.uuid4().hex[:8]}"
            
            documents.append(document)
            metadatas.append(metadata)
            ids.append(doc_id)
            
            # Add in batches
            if len(documents) >= batch_size:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                self.created_embeddings += len(documents)
                
                documents = []
                metadatas = []
                ids = []
        
        # Add any remaining documents
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            self.created_embeddings += len(documents)
        
        self.processed_rows += df.shape[0]
        logger.info(f"Added {df.shape[0]} rows from sheet {sheet_name} to ChromaDB")
    
    def _process_by_chunks(self, df, collection, text_columns, metadata_columns, sheet_name, chunk_size):
        """Process Excel data in chunks of multiple rows"""
        total_rows = df.shape[0]
        
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk_df = df.iloc[chunk_start:chunk_end]
            
            # Combine all text in the chunk
            chunk_texts = []
            for _, row in chunk_df.iterrows():
                if len(text_columns) == 1:
                    chunk_texts.append(str(row[text_columns[0]]))
                else:
                    chunk_texts.append("\n".join([f"{col}: {row[col]}" for col in text_columns]))
            
            document = "\n\n".join(chunk_texts)
            
            # Create metadata for the chunk
            metadata = {
                "chunk_start": chunk_start,
                "chunk_end": chunk_end - 1,
                "rows_count": chunk_end - chunk_start,
                "document_created": str(datetime.now())
            }
            
            # Add summary metadata from the first row
            if chunk_df.shape[0] > 0 and metadata_columns:
                first_row = chunk_df.iloc[0]
                for col in metadata_columns:
                    if col in first_row:
                        metadata[f"first_{col}"] = str(first_row[col])
            
            # Generate ID
            doc_id = f"{sheet_name}_chunk_{chunk_start}_{chunk_end}_{uuid.uuid4().hex[:8]}"
            
            # Add to collection
            collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
            self.created_embeddings += 1
        
        self.processed_rows += total_rows
        logger.info(f"Added {total_rows} rows in {(total_rows + chunk_size - 1) // chunk_size} "
                   f"chunks from sheet {sheet_name}")

    def export_schema(self, output_path):
        """Export the current schema to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.schema, f, indent=2)
        logger.info(f"Schema exported to {output_path}")

# Example usage
if __name__ == "__main__":
    # Example: Analyze Excel and generate schema
    embedder = EnterpriseExcelEmbedder()
    excel_path = "enterprise_data.xlsx"
    
    # Step 1: Analyze and generate recommended schema
    schema_recommendations = embedder.analyze_excel(excel_path)
    print("Recommended schema:")
    print(json.dumps(schema_recommendations, indent=2))
    
    # Export recommendations to schema file
    with open("recommended_schema.json", 'w') as f:
        json.dump(schema_recommendations, f, indent=2)
    
    # Step 2: Create embeddings using either recommended schema or custom schema
    # Option A: Use recommended schema
    embedder.schema = schema_recommendations
    
    # Option B: Load custom schema
    # embedder = EnterpriseExcelEmbedder(schema_path="custom_schema.json")
    
    # Create embeddings
    stats = embedder.embed_excel(excel_path, collection_name_prefix="enterprise")
    print(f"Embedding stats: {stats}")
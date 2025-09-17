import os
import mlflow
import numpy as np
import faiss
from pathlib import Path
from prefect import task, flow

# Data preprocessing
from data_processing import chunk_documents_recursively, generate_embeddings_st

# Elasticsearch
from indexing import index_in_elasticsearch_task

# Set MLflow tracking URI - Fixed the default port to match your setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))

@task
def get_latest_data_version(data_path: str) -> list:
    """
    Loads the latest version of data from local files (after dvc pull).
    """
    docs = []
    p = Path(data_path)
    
    # Check if path exists
    if not p.exists():
        print(f"Error: Data path does not exist: {data_path}")
        return []
    
    txt_files = list(p.glob("*.txt"))
    if not txt_files:
        print(f"Warning: No .txt files found in {data_path}")
        return []
    
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  # Only add non-empty documents
                    docs.append(content)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
    
    print(f"Loaded {len(docs)} documents from {data_path}")
    return docs

@task
def index_in_elasticsearch(chunks: list, embeddings: np.ndarray, index_name: str = "rag_kb"):
    """Elasticsearch indexing."""
    try:
        success, errors = index_in_elasticsearch_task(chunks, embeddings, index_name)
        print(f"Elasticsearch indexing completed: {success} documents indexed")
        if errors:
            print(f"Indexing errors: {len(errors)}")
        return success, errors
    except Exception as e:
        print(f"Elasticsearch indexing failed: {e}")
        return 0, [str(e)]

@task
def build_and_save_faiss_index(embeddings: np.ndarray, output_path: str = "models/faiss_index.bin"):
    """Builds and saves a FAISS index."""
    # Validate embeddings first
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings array is empty or None")
    
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings should be 2D array, got {embeddings.ndim}D")
    
    # Ensure embeddings are in a contiguous C-style array
    embeddings = np.ascontiguousarray(embeddings, dtype='float32')
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print(f"Built FAISS index with {index.ntotal} vectors.")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, output_path)
    print(f"FAISS index saved to {output_path}")
    return output_path

@flow(name="Batch Data Ingestion and Indexing")
def data_ingestion_flow(
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embedding_model: str = "all-MiniLM-L6-v2",
    es_index_name: str = "rag_kb"
):
    """
    The main flow to ingest and process data, with MLflow tracking.
    """
    # Test MLflow connection first
    try:
        mlflow.search_experiments()
        print("MLflow connection successful")
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        print("Check if MLflow server is running and MLFLOW_TRACKING_URI is correct")
        # Continue without MLflow if needed
        return False

    mlflow.set_experiment("RAG Knowledge Base Generation")

    with mlflow.start_run() as run:
        print(f"Starting MLflow run: {run.info.run_id}")

        mlflow.log_params({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "es_index_name": es_index_name
        })

        # --- Data Ingestion and Processing ---
        documents = get_latest_data_version(data_path="data/raw/pandas_docs")
        
        if not documents:
            print("Error: No documents loaded. Exiting...")
            return False
        
        chunks = chunk_documents_recursively(documents, chunk_size, chunk_overlap)
        
        if not chunks:
            print("Error: No chunks generated. Exiting...")
            return False
        
        embeddings = generate_embeddings_st(chunks, embedding_model)
        
        # Validate that chunks and embeddings match
        if len(chunks) != embeddings.shape[0]:
            print(f"Error: Mismatch between chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]})")
            return False

        mlflow.log_metric("num_documents", len(documents))
        mlflow.log_metric("num_chunks", len(chunks))

        # --- Indexing ---
        faiss_index_path = build_and_save_faiss_index(embeddings)

        # Updated Elasticsearch indexing with proper parameters
        es_success, es_errors = index_in_elasticsearch(chunks, embeddings, es_index_name)
        mlflow.log_metric("es_indexed_docs", es_success)
        mlflow.log_metric("es_errors", len(es_errors) if es_errors else 0)

        # Log artifacts - check if file exists first
        if os.path.exists(faiss_index_path):
            mlflow.log_artifact(faiss_index_path, artifact_path="faiss_index")
        else:
            print(f"Warning: FAISS index file not found: {faiss_index_path}")
        
        print("MLflow run completed.")
        return True

if __name__ == "__main__":
    success = data_ingestion_flow()
    if not success:
        print("Pipeline failed!")
    else:
        print("Pipeline completed successfully!")
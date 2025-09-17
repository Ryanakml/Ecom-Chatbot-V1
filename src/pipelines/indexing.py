from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import numpy as np

def create_es_index(client: Elasticsearch, index_name: str, vector_dim: int):
    """Creates an Elasticsearch index with dense_vector mapping if not exists."""
    if not client.indices.exists(index=index_name):
        mapping = {
            "properties": {
                "content": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": vector_dim
                }
            }
        }
        
        # ES 8 syntax - langsung pass mappings
        client.indices.create(index=index_name, mappings=mapping)
        print(f"Created index '{index_name}' with dimension {vector_dim}")
    else:
        print(f"Index '{index_name}' already exists.")

def index_in_elasticsearch_task(chunks: list, embeddings: np.ndarray, index_name: str = "rag_kb"):
    """
    Performs bulk indexing of chunks and embeddings into Elasticsearch.
    """
    # Init client
    es_client = Elasticsearch("http://localhost:9200")

    # Validate inputs
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddings must be a numpy.ndarray")
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got {embeddings.ndim}D")
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) count mismatch")

    vector_dim = embeddings.shape[1]

    # Ensure index exists
    create_es_index(es_client, index_name, vector_dim)

    # Prepare bulk actions
    actions = [
        {
            "_index": index_name,
            "_source": {
                "content": chunk,
                "embedding": embedding.tolist(),
            },
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    # Bulk index
    print(f"Bulk indexing {len(actions)} documents into '{index_name}'...")
    success, errors = bulk(es_client, actions, refresh="wait_for")

    print(f"Successfully indexed {success} documents.")
    if errors:
        print(f"Failed to index some documents: {len(errors)} errors")
        for err in errors[:5]:  # print only first 5 errors
            print(err)

    return success, errors

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

def chunk_documents_recursively(documents: list[str], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """
    Chunks a list of documents using a recursive character splitter.
    
    Args:
        documents: A list of strings, where each string is a document.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.
        
    Returns:
        A list of strings, where each string is a text chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""] # Prioritized list of separators
    )
    
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc)
        all_chunks.extend(chunks)
        
    print(f"Split {len(documents)} documents into {len(all_chunks)} chunks.")
    return all_chunks

def generate_embeddings_st(chunks: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generates embeddings for a list of text chunks using a SentenceTransformer model.
    
    Args:
        chunks: A list of text chunks.
        model_name: The name of the SentenceTransformer model to use.
        
    Returns:
        A numpy array of embeddings.
    """
    model = SentenceTransformer(model_name)
    print(f"Generating embeddings for {len(chunks)} chunks with model '{model_name}'...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"Embeddings generated with shape: {embeddings.shape}")
    return embeddings
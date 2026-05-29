# ingestion/embedder.py

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME
)
from typing import List
import time


def get_embedding_model():
    """
    Returns the Gemini embedding model.
    This converts text → vector (list of numbers representing meaning).
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )


def get_chroma_collection():
    """
    Connects to ChromaDB and returns the collection.
    ChromaDB is our vector store — it stores and searches vectors.
    PersistentClient means data is saved to disk (./chroma_db folder).
    So you don't re-embed every time you restart the app.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for search
    )
    return collection


def embed_and_store(chunks: List[dict]) -> bool:
    """
    Main function: converts chunks to vectors and stores in ChromaDB.

    Flow:
    1. Get Gemini embedding model
    2. Connect to ChromaDB collection
    3. For each chunk: convert text → vector
    4. Store vector + text + metadata in ChromaDB

    We process in batches of 10 to avoid API rate limits.
    """
    if not chunks:
        print("❌ No chunks to embed.")
        return False

    print(f"\n🔢 Embedding {len(chunks)} chunks into ChromaDB...")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Storage: {CHROMA_DB_PATH}\n")

    embedding_model = get_embedding_model()
    collection = get_chroma_collection()

    # Process in batches of 10
    batch_size = 10
    total_stored = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        texts = [chunk["text"] for chunk in batch]
        ids = [f"{chunk['source']}__chunk_{chunk['chunk_index']}" for chunk in batch]
        metadatas = [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"]
            }
            for chunk in batch
        ]

        # Convert texts to vectors
        embeddings = embedding_model.embed_documents(texts)

        # Store in ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        total_stored += len(batch)
        print(f"   ✅ Batch {i//batch_size + 1}: stored {total_stored}/{len(chunks)} chunks")

        # Small delay to respect API rate limits
        time.sleep(0.5)

    print(f"\n🎉 Embedding complete! {total_stored} chunks stored in ChromaDB.\n")
    return True


def get_collection_stats() -> dict:
    """
    Returns stats about what's currently stored in ChromaDB.
    Useful for debugging and verifying ingestion worked.
    """
    collection = get_chroma_collection()
    count = collection.count()
    return {
        "total_chunks": count,
        "collection_name": COLLECTION_NAME,
        "storage_path": CHROMA_DB_PATH
    }
# test_ingestion.py

from ingestion.loader import load_all_documents
from ingestion.chunker import chunk_all_documents
from ingestion.embedder import embed_and_store, get_collection_stats


def test_ingestion():
    print("=" * 50)
    print("TESTING FULL INGESTION PIPELINE")
    print("=" * 50)

    # Step 1: Load documents
    print("\n📄 STEP 1: Loading documents...")
    documents = load_all_documents("./data")

    if not documents:
        print("❌ No documents loaded. Add files to /data folder.")
        return

    # Step 2: Chunk documents
    print("\n✂️  STEP 2: Chunking documents...")
    chunks = chunk_all_documents(documents)

    # Step 3: Embed and store
    print("\n🔢 STEP 3: Embedding and storing in ChromaDB...")
    success = embed_and_store(chunks)

    # Step 4: Verify
    print("\n📊 STEP 4: Verifying storage...")
    stats = get_collection_stats()
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total chunks stored: {stats['total_chunks']}")
    print(f"   Storage path: {stats['storage_path']}")

    print("\n" + "=" * 50)
    print("✅ INGESTION PIPELINE TEST COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    test_ingestion()
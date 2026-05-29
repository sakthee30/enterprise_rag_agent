# ingestion/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from typing import List


def chunk_text(text: str, source_name: str = "unknown") -> List[dict]:
    """
    Splits raw text into overlapping chunks.

    Returns a list of dicts — each dict is one chunk:
    {
        "text": "the actual chunk content",
        "source": "filename it came from",
        "chunk_index": 0  ← position in the document
    }

    Why RecursiveCharacterTextSplitter?
    It tries to split on paragraphs first, then sentences, then words.
    This keeps chunks semantically meaningful — not cut mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    raw_chunks = splitter.split_text(text)

    # Wrap each chunk with metadata — source and position
    chunks = [
        {
            "text": chunk,
            "source": source_name,
            "chunk_index": i
        }
        for i, chunk in enumerate(raw_chunks)
    ]

    print(f"✅ Chunked '{source_name}' → {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    return chunks


def chunk_all_documents(documents: dict) -> List[dict]:
    """
    Chunks ALL loaded documents.
    Input: { filename: text } dict from loader.py
    Output: flat list of all chunks from all documents

    This is the function the pipeline calls.
    """
    all_chunks = []

    for filename, text in documents.items():
        chunks = chunk_text(text, source_name=filename)
        all_chunks.extend(chunks)

    print(f"\n📊 Total chunks across all documents: {len(all_chunks)}\n")
    return all_chunks
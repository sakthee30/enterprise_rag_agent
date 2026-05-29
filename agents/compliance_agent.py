# agents/compliance_agent.py

import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config.settings import (
    GOOGLE_API_KEY, GEMINI_MODEL, EMBEDDING_MODEL,
    CHROMA_DB_PATH, COLLECTION_NAME,
    TOP_K_RESULTS, extract_text
)


def get_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1  # very low — compliance answers must be precise
    )


def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )


def retrieve_chunks(query: str) -> list:
    """Retrieves relevant compliance chunks from ChromaDB."""
    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(query)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(TOP_K_RESULTS, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "relevance_score": round(1 - dist, 3)
            })

    print(f"✅ Compliance Agent retrieved {len(chunks)} chunks")
    return chunks


def run_compliance_agent(query: str) -> dict:
    """
    Full Compliance Agent flow.
    Tuned for: GDPR, data handling, regulations, what is/isn't allowed.
    Returns a clear COMPLIANT / NON-COMPLIANT / UNCLEAR verdict.
    """
    print(f"\n✅ Compliance Agent processing: '{query}'")

    chunks = retrieve_chunks(query)

    if not chunks:
        return {
            "agent": "compliance",
            "answer": "I could not find relevant compliance information for your query.",
            "sources": [],
            "chunks_used": 0
        }

    context = "\n\n".join([
        f"[Source: {c['source']}, Relevance: {c['relevance_score']}]\n{c['text']}"
        for c in chunks
    ])

    sources = list(set([c["source"] for c in chunks]))

    llm = get_llm()

    compliance_prompt = f"""
You are a compliance officer for a UK retail enterprise (Waitrose/JLP).
Your job is to determine if an action or situation is compliant with company regulations.

Based ONLY on the compliance documents provided:
1. State clearly: COMPLIANT, NON-COMPLIANT, or REQUIRES REVIEW
2. Explain why, citing the specific rule or guideline
3. Suggest corrective action if non-compliant

If the answer cannot be determined from the documents, say "REQUIRES REVIEW - insufficient information in current compliance documentation."

COMPLIANCE DOCUMENTS:
{context}

COMPLIANCE QUERY: {query}

COMPLIANCE ASSESSMENT:
"""

    response = llm.invoke(compliance_prompt)
    answer = extract_text(response.content)

    print(f"✅ Compliance Agent assessed successfully")

    return {
        "agent": "compliance",
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks)
    }
# agents/policy_agent.py

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
        temperature=0.3
    )


def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )


def retrieve_chunks(query: str) -> list:
    """
    Searches ChromaDB for the most relevant chunks for this query.

    Flow:
    1. Convert query to a vector using Gemini embeddings
    2. Search ChromaDB for the TOP_K_RESULTS closest vectors
    3. Return the matching text chunks with their metadata
    """
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
                "chunk_index": meta.get("chunk_index", 0),
                "relevance_score": round(1 - dist, 3)  # convert distance to similarity
            })

    print(f"📋 Policy Agent retrieved {len(chunks)} chunks")
    return chunks


def run_policy_agent(query: str) -> dict:
    """
    Full Policy Agent flow:
    1. Retrieve relevant chunks from ChromaDB
    2. Build context from chunks
    3. Send query + context to Gemini with policy-focused prompt
    4. Return structured response

    The system prompt is tuned specifically for policy questions —
    it tells Gemini to act as a policy expert and cite sources.
    """
    print(f"\n📋 Policy Agent processing: '{query}'")

    chunks = retrieve_chunks(query)

    if not chunks:
        return {
            "agent": "policy",
            "answer": "I could not find relevant policy information for your query.",
            "sources": [],
            "chunks_used": 0
        }

    # Build context string from retrieved chunks
    context = "\n\n".join([
        f"[Source: {c['source']}, Relevance: {c['relevance_score']}]\n{c['text']}"
        for c in chunks
    ])

    sources = list(set([c["source"] for c in chunks]))

    llm = get_llm()

    policy_prompt = f"""
You are an enterprise policy expert for a UK retail organization (Waitrose/JLP).
Answer the user's question based ONLY on the policy documents provided below.

If the answer is not in the documents, say "This information is not covered in the current policy documents."
Always be clear, concise and professional.
Cite which section or document your answer comes from.

POLICY DOCUMENTS:
{context}

USER QUESTION: {query}

ANSWER:
"""

    response = llm.invoke(policy_prompt)
    answer = extract_text(response.content)

    print(f"✅ Policy Agent answered successfully")

    return {
        "agent": "policy",
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks)
    }
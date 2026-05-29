# agents/incident_agent.py

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
        temperature=0.2
    )


def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )


def retrieve_chunks(query: str) -> list:
    """Retrieves relevant incident procedure chunks from ChromaDB."""
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

    print(f"🔧 Incident Agent retrieved {len(chunks)} chunks")
    return chunks


def run_incident_agent(query: str) -> dict:
    """
    Full Incident Agent flow.
    Tuned for: P1-P4 incidents, resolution steps, escalation,
    ServiceNow logging — directly maps to your CG/JLP experience.
    """
    print(f"\n🔧 Incident Agent processing: '{query}'")

    chunks = retrieve_chunks(query)

    if not chunks:
        return {
            "agent": "incident",
            "answer": "I could not find relevant incident procedures for your query.",
            "sources": [],
            "chunks_used": 0
        }

    context = "\n\n".join([
        f"[Source: {c['source']}, Relevance: {c['relevance_score']}]\n{c['text']}"
        for c in chunks
    ])

    sources = list(set([c["source"] for c in chunks]))

    llm = get_llm()

    incident_prompt = f"""
You are an expert incident manager for a UK retail enterprise (Waitrose/JLP).
You are familiar with ITIL incident management and ServiceNow workflows.

Based ONLY on the incident procedures provided below:
1. Identify the incident priority (P1/P2/P3/P4) if applicable
2. Provide clear step-by-step resolution guidance
3. State escalation requirements and resolution time targets
4. Mention ServiceNow logging requirements

If resolution steps are not in the documents, say "Please refer to the live runbook or escalate to L2 support."

INCIDENT PROCEDURES:
{context}

INCIDENT QUERY: {query}

INCIDENT RESOLUTION GUIDANCE:
"""

    response = llm.invoke(incident_prompt)
    answer = extract_text(response.content)

    print(f"✅ Incident Agent resolved successfully")

    return {
        "agent": "incident",
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks)
    }
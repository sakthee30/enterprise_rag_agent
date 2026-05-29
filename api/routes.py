# api/routes.py

from fastapi import APIRouter, HTTPException
from api.schemas import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse,
    HealthResponse, ErrorResponse
)
from agents import run_query
from ingestion.loader import load_all_documents
from ingestion.chunker import chunk_all_documents
from ingestion.embedder import embed_and_store, get_collection_stats

# APIRouter groups related endpoints together
# In larger apps you'd have separate routers for auth, users, etc.
router = APIRouter()


# ── Health Check ──────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Confirms the API is running and ChromaDB is accessible"
)
async def health_check():
    """
    GET /health

    Why async?
    FastAPI is async by default. Using async def means the server
    can handle multiple requests simultaneously without blocking.
    Critical for production performance.

    This endpoint is called by:
    - Docker health checks
    - Load balancers
    - Monitoring dashboards
    - Your Streamlit UI on startup
    """
    try:
        stats = get_collection_stats()
        return HealthResponse(
            status="healthy",
            service="Enterprise RAG Agent",
            version="1.0.0",
            chromadb_chunks=stats["total_chunks"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


# ── Query Endpoint ────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the agent system",
    description="Send a question to the multi-agent RAG system"
)
async def query_agent(request: QueryRequest):
    """
    POST /query

    Flow:
    1. Receive query from client
    2. Validate via Pydantic (automatic)
    3. Pass to LangGraph agent system
    4. Return structured response

    The heavy lifting is done by run_query() in agents/__init__.py
    FastAPI's job here is just: receive → validate → call → respond
    """
    try:
        print(f"\n🌐 API received query: '{request.query}'")

        if not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )

        from agents import run_query_with_retry
        result = run_query_with_retry(request.query)

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            agent=result["agent"],
            route=result["route"],
            sources=result["sources"] or [],
            chunks_used=result["chunks_used"] or 0,
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(e)}"
        )


# ── Ingest Endpoint ───────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents",
    description="Load documents from the data folder into ChromaDB"
)
async def ingest_documents(request: IngestRequest):
    """
    POST /ingest

    Flow:
    1. Load all documents from the specified folder
    2. Chunk them
    3. Embed and store in ChromaDB
    4. Return stats

    This endpoint is called:
    - Once at project setup
    - Every time new documents are added
    - From the Streamlit UI's document uploader
    """
    try:
        print(f"\n🌐 API received ingest request for: '{request.data_folder}'")

        # Step 1: Load
        documents = load_all_documents(request.data_folder)
        if not documents:
            return IngestResponse(
                status="warning",
                documents_loaded=0,
                total_chunks=0,
                message=f"No documents found in {request.data_folder}"
            )

        # Step 2: Chunk
        chunks = chunk_all_documents(documents)

        # Step 3: Embed and store
        success = embed_and_store(chunks)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Embedding failed"
            )

        return IngestResponse(
            status="success",
            documents_loaded=len(documents),
            total_chunks=len(chunks),
            message=f"Successfully ingested {len(documents)} document(s) with {len(chunks)} chunks"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ingest error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )
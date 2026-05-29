# api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List


# ── Request Models (what the client sends) ────────────────

class QueryRequest(BaseModel):
    """
    Request body for POST /query
    The user sends a question and optionally their session id.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user's question to the agent system",
        example="What is the return policy for Waitrose products?"
    )
    session_id: Optional[str] = Field(
        default="default",
        description="Optional session identifier for tracking"
    )


class IngestRequest(BaseModel):
    """
    Request body for POST /ingest
    Tells the API which folder to ingest documents from.
    """
    data_folder: Optional[str] = Field(
        default="./data",
        description="Path to folder containing documents to ingest"
    )


# ── Response Models (what the API returns) ────────────────

class QueryResponse(BaseModel):
    """
    Response body for POST /query
    Returns the answer + full agent trace for transparency.
    """
    query: str
    answer: str
    agent: str                    # which agent handled it
    route: str                    # what the router classified it as
    sources: List[str]            # which documents were used
    chunks_used: int              # how many chunks were retrieved
    status: str = "success"


class IngestResponse(BaseModel):
    """
    Response body for POST /ingest
    """
    status: str
    documents_loaded: int
    total_chunks: int
    message: str


class HealthResponse(BaseModel):
    """
    Response body for GET /health
    Standard in every production API — used by Docker,
    load balancers, and monitoring tools to check if service is alive.
    """
    status: str
    service: str
    version: str
    chromadb_chunks: int


class ErrorResponse(BaseModel):
    """
    Standard error response shape.
    Always return structured errors — never raw Python exceptions.
    """
    status: str = "error"
    message: str
    detail: Optional[str] = None
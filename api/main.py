# api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
import uvicorn
from config.settings import API_HOST, API_PORT

# ── Create FastAPI app ────────────────────────────────────
app = FastAPI(
    title="Enterprise RAG Agent",
    description="""
    A multi-agent RAG system for enterprise document intelligence.

    Built with LangGraph, LangChain, Gemini, ChromaDB, and FastAPI.

    ## Agents
    - **Policy Agent** — answers questions about company policies
    - **Compliance Agent** — assesses regulatory compliance
    - **Incident Agent** — provides incident resolution guidance

    ## Flow
    Query → Router Agent → Specialist Agent → RAG retrieval → Gemini → Response
    """,
    version="1.0.0",
    contact={
        "name": "Enterprise RAG Agent",
        "url": "https://github.com/sakthee30/enterprise_rag_agent"
    }
)

# ── CORS Middleware ───────────────────────────────────────
# CORS = Cross Origin Resource Sharing
# Without this, your Streamlit UI (running on port 8501)
# cannot talk to FastAPI (running on port 8000)
# In production, replace "*" with your actual domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ───────────────────────────────────────
# prefix="/api/v1" means all endpoints become:
# /api/v1/health, /api/v1/query, /api/v1/ingest
app.include_router(router, prefix="/api/v1", tags=["Agent"])


# ── Root endpoint ─────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "Enterprise RAG Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# ── Run the server ────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting Enterprise RAG Agent API...")
    print(f"   Host: {API_HOST}")
    print(f"   Port: {API_PORT}")
    print(f"   Docs: http://localhost:{API_PORT}/docs\n")

    uvicorn.run(
    "api.main:app",
    host=API_HOST,
    port=API_PORT,
    reload=True,
    reload_excludes=["venv/*", "chroma_db/*", "*.pyc", "__pycache__/*"]
)
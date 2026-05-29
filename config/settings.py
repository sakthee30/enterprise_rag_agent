# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ── Vector Store ─────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "enterprise_docs"

# ── Retrieval ─────────────────────────────────────────
TOP_K_RESULTS = 5
TOP_K_RERANK = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── API ───────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Agent types ───────────────────────────────────────
AGENT_TYPES = ["policy", "compliance", "incident"]

# ── Helper: extract clean text from Gemini response ──
def extract_text(content):
    """Handles both string and list response formats from Gemini."""
    if isinstance(content, list):
        return content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    return content
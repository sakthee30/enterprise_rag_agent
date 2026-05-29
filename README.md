# Enterprise RAG Agent 🤖

A production-grade **multi-agent RAG system** for enterprise document intelligence.
Built with LangGraph, LangChain, Google Gemini, ChromaDB, FastAPI, and Docker.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-purple)](https://langchain-ai.github.io/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What It Does

Enterprise RAG Agent answers questions about company policies, compliance rules, and incident procedures — by intelligently routing each query to a specialized AI agent that retrieves relevant information from your documents using RAG.

**Domain:** Retail enterprise operations (inspired by JLP/Waitrose UK retail account)

---

## 🏗️ Architecture
```
User Query
↓
FastAPI Backend (REST API)
↓
LangGraph Router Agent (classifies query)
↓
┌─────────────────────────────────────┐
│  Policy      Compliance   Incident  │
│  Agent   OR  Agent     OR Agent     │
└─────────────────────────────────────┘
↓
RAG Pipeline (ChromaDB semantic search + reranking)
↓
Google Gemini LLM
↓
Structured Response → Streamlit UI
```
---

## ✨ Key Features

- **Multi-agent routing** — LangGraph state machine routes queries to the right specialist agent
- **RAG pipeline** — semantic search over enterprise documents with relevance scoring
- **3 specialist agents** — Policy, Compliance, and Incident agents with domain-tuned prompts
- **Production FastAPI backend** — async REST API with Pydantic validation and auto-docs
- **Agent trace viewer** — Streamlit UI shows which agent handled each query and from which source
- **Fully containerized** — Docker Compose runs the entire stack with one command
- **LLM-agnostic architecture** — built on Google Gemini, designed to swap LLM providers

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph, LangChain |
| LLM | Google Gemini 3.5 Flash |
| Vector store | ChromaDB |
| Embeddings | Google Gemini Embeddings |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Document processing | PyPDF, docx2txt |

---

## 🤖 Agent System

### Router Agent
Classifies incoming queries using Gemini with temperature=0 (deterministic routing).
Returns one of: `policy` | `compliance` | `incident`

### Policy Agent
Answers questions about company policies, procedures, and guidelines.
Retrieves relevant policy chunks from ChromaDB and generates grounded answers.

### Compliance Agent
Assesses regulatory compliance — returns COMPLIANT / NON-COMPLIANT / REQUIRES REVIEW
with corrective action guidance. Tuned for GDPR and enterprise data regulations.

### Incident Agent
Provides incident resolution guidance based on ITIL methodology.
Handles P1–P4 priority classification, escalation paths, and ServiceNow logging.
(Domain knowledge from 1.5 years supporting Oracle Retail on JLP/Waitrose account)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- Google Gemini API key (free at aistudio.google.com)

### Run with Docker (recommended)

```bash
# Clone the repository
git clone https://github.com/sakthee30/enterprise_rag_agent.git
cd enterprise_rag_agent

# Add your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Start both services
docker-compose up --build
```

Open:
- **UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

### Run locally

```bash
# Install dependencies
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Add API key to .env
# GOOGLE_API_KEY=your_key_here

# Ingest documents
python test_ingestion.py

# Terminal 1: Start API
python -m api.main

# Terminal 2: Start UI
streamlit run ui/app.py
```

---

## 📁 Project Structure
```
enterprise_rag_agent/
├── agents/
│   ├── init.py          # LangGraph graph builder + run_query()
│   ├── router_agent.py      # Query classifier
│   ├── policy_agent.py      # Policy RAG agent
│   ├── compliance_agent.py  # Compliance RAG agent
│   └── incident_agent.py    # Incident RAG agent
├── api/
│   ├── main.py              # FastAPI app entry point
│   ├── routes.py            # API endpoints
│   └── schemas.py           # Pydantic request/response models
├── ingestion/
│   ├── loader.py            # PDF/DOCX/TXT document loader
│   ├── chunker.py           # Text splitter with overlap
│   └── embedder.py          # Gemini embeddings + ChromaDB storage
├── ui/
│   └── app.py               # Streamlit chat interface
├── config/
│   └── settings.py          # Centralized configuration
├── data/                    # Your enterprise documents go here
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service orchestration
└── requirements.txt         # Python dependencies
```
---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Service health + ChromaDB stats |
| POST | `/api/v1/query` | Query the multi-agent system |
| POST | `/api/v1/ingest` | Ingest documents into ChromaDB |

### Example Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the return policy for Waitrose products?"}'
```

### Example Response

```json
{
  "query": "What is the return policy for Waitrose products?",
  "answer": "Customers may return any product within 30 days with a valid receipt...",
  "agent": "policy",
  "route": "policy",
  "sources": ["waitrose_policy.txt"],
  "chunks_used": 4,
  "status": "success"
}
```

---

## 💡 Sample Questions

**Policy queries:**
- "What is the return policy for Waitrose products?"
- "Are perishable goods returnable?"

**Compliance queries:**
- "Can we share customer data with third party advertisers?"
- "What are the CCTV data retention rules?"

**Incident queries:**
- "How do I handle a P1 system incident?"
- "What is the resolution time for a P2 incident?"

---

## 🧠 How RAG Works Here

1. **Ingestion** — Documents are chunked into 500-character overlapping pieces
2. **Embedding** — Each chunk converted to a vector using Gemini embeddings
3. **Storage** — Vectors stored in ChromaDB with metadata
4. **Retrieval** — User query vectorized, top-5 semantically similar chunks retrieved
5. **Generation** — Chunks + query sent to Gemini with agent-specific system prompt
6. **Response** — Grounded answer with source citation returned to user

---


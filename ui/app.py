# ui/app.py
# ── Config ────────────────────────────────────────────────
import os
# In Docker: API runs as service named 'api'
# Locally: API runs on localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

import streamlit as st
import requests
import json

# ── Page Setup ────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .agent-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .policy-badge { background: #EAF3DE; color: #27500A; }
    .compliance-badge { background: #FAEEDA; color: #633806; }
    .incident-badge { background: #FDEEF5; color: #9B3070; }
    .system-badge { background: #F1EFE8; color: #444441; }
    .trace-box {
        background: #F8F9FA;
        border-left: 3px solid #185FA5;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        margin-top: 8px;
    }
    .health-ok { color: #27500A; font-weight: 600; }
    .health-err { color: #A32D2D; font-weight: 600; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────

def check_health():
    """Checks if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def query_agent(query: str) -> dict:
    """Sends query to FastAPI and returns the response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "session_id": "streamlit"},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": True,
                "message": f"API error: {response.status_code}",
                "detail": response.text
            }
    except requests.exceptions.ConnectionError:
        return {
            "error": True,
            "message": "Cannot connect to API. Make sure FastAPI is running on port 8000."
        }
    except Exception as e:
        return {"error": True, "message": str(e)}


def ingest_documents(data_folder: str = "./data") -> dict:
    """Triggers document ingestion via FastAPI."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json={"data_folder": data_folder},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": True, "message": str(e)}


def get_agent_badge(agent: str) -> str:
    """Returns colored HTML badge for each agent type."""
    badges = {
        "policy":     '<span class="agent-badge policy-badge">📋 Policy Agent</span>',
        "compliance": '<span class="agent-badge compliance-badge">✅ Compliance Agent</span>',
        "incident":   '<span class="agent-badge incident-badge">🔧 Incident Agent</span>',
        "system":     '<span class="agent-badge system-badge">⚙️ System</span>'
    }
    return badges.get(agent, badges["system"])


def get_sample_questions():
    """Sample questions for quick testing."""
    return [
        "What is the return policy for Waitrose products?",
        "Can we share customer data with third party advertisers?",
        "How do I handle a P1 system incident?",
        "What are the CCTV data retention rules?",
        "What is the resolution time for a P2 incident?",
        "Are perishable goods returnable?"
    ]


# ── Session State ─────────────────────────────────────────
# Streamlit reruns the entire script on every interaction.
# st.session_state persists data across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_traces" not in st.session_state:
    st.session_state.agent_traces = []


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Enterprise RAG Agent")
    st.caption("Multi-agent document intelligence system")

    st.divider()

    # Health status
    st.subheader("🔌 API Status")
    health = check_health()
    if health:
        st.markdown(f'<span class="health-ok">● Connected</span>', unsafe_allow_html=True)
        st.caption(f"Service: {health.get('service', 'Unknown')}")
        st.caption(f"Chunks in DB: {health.get('chromadb_chunks', 0)}")
        st.caption(f"Version: {health.get('version', '1.0.0')}")
    else:
        st.markdown('<span class="health-err">● Disconnected</span>', unsafe_allow_html=True)
        st.caption("Start FastAPI: python -m api.main")

    st.divider()

    # Agent trace viewer
    st.subheader("🔍 Agent Trace")
    st.caption("Shows which agent handled each query")

    if st.session_state.agent_traces:
        latest = st.session_state.agent_traces[-1]
        st.markdown(get_agent_badge(latest.get("agent", "system")), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="trace-box">
            <b>Route:</b> {latest.get('route', 'N/A')}<br>
            <b>Agent:</b> {latest.get('agent', 'N/A')}<br>
            <b>Chunks used:</b> {latest.get('chunks_used', 0)}<br>
            <b>Sources:</b> {', '.join(latest.get('sources', []))}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("Ask a question to see the agent trace here.")

    st.divider()

    # Document ingestion
    st.subheader("📂 Knowledge Base")
    if st.button("🔄 Re-ingest Documents", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            result = ingest_documents()
            if result.get("status") == "success":
                st.success(f"✅ {result.get('message')}")
            else:
                st.error(f"❌ {result.get('message', 'Ingestion failed')}")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_traces = []
        st.rerun()

    st.divider()
    st.caption("Built with LangGraph · LangChain · Gemini · ChromaDB · FastAPI")
    st.caption("© Enterprise RAG Agent 2025")


# ── Main Chat Interface ───────────────────────────────────
st.title("Enterprise Document Intelligence Agent")
st.caption("Ask questions about company policies, compliance rules, or incident procedures.")

# Sample questions
st.subheader("💡 Try these questions:")
sample_cols = st.columns(3)
samples = get_sample_questions()

for i, col in enumerate(sample_cols):
    with col:
        if i < len(samples):
            if st.button(samples[i], use_container_width=True, key=f"sample_{i}"):
                st.session_state.pending_query = samples[i]
        if i + 3 < len(samples):
            if st.button(samples[i + 3], use_container_width=True, key=f"sample_{i+3}"):
                st.session_state.pending_query = samples[i + 3]

st.divider()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "agent" in message:
            st.markdown(
                get_agent_badge(message["agent"]),
                unsafe_allow_html=True
            )
        st.markdown(message["content"])

# ── Handle Input ──────────────────────────────────────────

# Check for sample question click
if "pending_query" in st.session_state:
    user_input = st.session_state.pending_query
    del st.session_state.pending_query
else:
    user_input = None

# Chat input box
typed_input = st.chat_input("Ask about policies, compliance, or incidents...")
if typed_input:
    user_input = typed_input

# Process the query
if user_input:
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_agent(user_input)

        if result.get("error"):
            st.error(f"❌ {result.get('message')}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {result.get('message')}",
                "agent": "system"
            })
        else:
            # Show agent badge
            st.markdown(
                get_agent_badge(result.get("agent", "system")),
                unsafe_allow_html=True
            )
            # Show answer
            st.markdown(result.get("answer", "No answer returned."))

            # Show sources
            if result.get("sources"):
                st.caption(f"📂 Sources: {', '.join(result['sources'])}")

            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("answer", ""),
                "agent": result.get("agent", "system")
            })

            # Save agent trace for sidebar
            st.session_state.agent_traces.append({
                "agent": result.get("agent"),
                "route": result.get("route"),
                "chunks_used": result.get("chunks_used"),
                "sources": result.get("sources", [])
            })

            st.rerun()
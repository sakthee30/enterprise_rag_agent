# agents/__init__.py

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from agents.router_agent import route_query
from agents.policy_agent import run_policy_agent
from agents.compliance_agent import run_compliance_agent
from agents.incident_agent import run_incident_agent


# ── State: what flows through the graph ──────────────────
class AgentState(TypedDict):
    """
    This is the state object that flows through every node in the graph.
    Every node reads from this and writes back to it.
    Think of it as a shared memory between all agents.
    """
    query: str                    # the user's original question
    route: Optional[str]          # set by router: policy/compliance/incident
    answer: Optional[str]         # set by the specialist agent
    agent: Optional[str]          # which agent handled it
    sources: Optional[list]       # which documents were used
    chunks_used: Optional[int]    # how many chunks were retrieved


# ── Node functions ────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """Node 1: classifies the query and sets the route."""
    route = route_query(state["query"])
    return {**state, "route": route}


def policy_node(state: AgentState) -> AgentState:
    """Node 2a: runs the policy agent."""
    result = run_policy_agent(state["query"])
    return {
        **state,
        "answer": result["answer"],
        "agent": result["agent"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"]
    }


def compliance_node(state: AgentState) -> AgentState:
    """Node 2b: runs the compliance agent."""
    result = run_compliance_agent(state["query"])
    return {
        **state,
        "answer": result["answer"],
        "agent": result["agent"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"]
    }


def incident_node(state: AgentState) -> AgentState:
    """Node 2c: runs the incident agent."""
    result = run_incident_agent(state["query"])
    return {
        **state,
        "answer": result["answer"],
        "agent": result["agent"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"]
    }


def decide_route(state: AgentState) -> str:
    """
    Conditional edge function — LangGraph calls this after the router node
    to decide which node to go to next.
    Returns the name of the next node as a string.
    """
    return state["route"]


# ── Build the graph ───────────────────────────────────────

def build_agent_graph():
    """
    Builds and compiles the LangGraph state machine.

    Graph structure:
    START → router_node → [decide_route] → policy_node  → END
                                         → compliance_node → END
                                         → incident_node → END

    This is what makes it a GRAPH not a chain —
    the path through it changes based on the query.
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("policy", policy_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("incident", incident_node)

    # Entry point
    graph.set_entry_point("router")

    # Conditional edge: after router, go to the right agent
    graph.add_conditional_edges(
        "router",
        decide_route,
        {
            "policy": "policy",
            "compliance": "compliance",
            "incident": "incident"
        }
    )

    # All agents lead to END
    graph.add_edge("policy", END)
    graph.add_edge("compliance", END)
    graph.add_edge("incident", END)

    return graph.compile()


# Single compiled graph instance — imported by FastAPI and UI
agent_graph = build_agent_graph()

import time
from config.settings import MAX_RETRIES, RETRY_DELAY

def run_query_with_retry(query: str) -> dict:
    """
    Wraps run_query with retry logic.
    Handles temporary Gemini 503 errors gracefully.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return run_query(query)
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < MAX_RETRIES:
                    print(f"⚠️  Gemini unavailable. Retry {attempt}/{MAX_RETRIES} in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    return {
                        "query": query,
                        "answer": "The AI service is temporarily unavailable. Please try again in a moment.",
                        "agent": "system",
                        "route": "error",
                        "sources": [],
                        "chunks_used": 0
                    }
            else:
                raise

def run_query(query: str) -> dict:
    """
    Main entry point for the entire system.
    This is what FastAPI and Streamlit will call.

    Input: user query string
    Output: dict with answer, agent, sources, chunks_used
    """
    print(f"\n{'='*50}")
    print(f"🚀 Processing query: '{query}'")
    print(f"{'='*50}")

    initial_state = AgentState(
        query=query,
        route=None,
        answer=None,
        agent=None,
        sources=None,
        chunks_used=None
    )

    final_state = agent_graph.invoke(initial_state)

    return {
        "query": query,
        "answer": final_state["answer"],
        "agent": final_state["agent"],
        "route": final_state["route"],
        "sources": final_state["sources"],
        "chunks_used": final_state["chunks_used"]
    }
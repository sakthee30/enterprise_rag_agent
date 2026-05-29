# agents/router_agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY, GEMINI_MODEL, extract_text
from typing import Literal


def get_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0  # deterministic — we want consistent routing
    )


def route_query(query: str) -> Literal["policy", "compliance", "incident"]:
    """
    The Router Agent — classifies the user query into one of 3 categories.

    How it works:
    - Sends the query to Gemini with a strict classification prompt
    - Gemini returns exactly one word: policy / compliance / incident
    - That word becomes the route in the LangGraph state machine

    Why temperature=0?
    Routing must be deterministic. We don't want creative answers here —
    we want the same input to always produce the same route.
    """
    llm = get_llm()

    classification_prompt = f"""
You are a query classifier for an enterprise retail system.
Classify the following query into exactly ONE of these categories:

- policy: questions about company policies, rules, procedures, return policies, guidelines
- compliance: questions about whether something is allowed, GDPR, data handling, regulations
- incident: questions about system issues, incident resolution, P1/P2/P3/P4, ServiceNow

Query: {query}

Respond with ONLY one word — either: policy, compliance, or incident
Do not explain. Do not add punctuation. Just the one word.
"""

    response = llm.invoke(classification_prompt)
    raw = extract_text(response.content).strip().lower()

    # Safety: if Gemini returns something unexpected, default to policy
    if raw not in ["policy", "compliance", "incident"]:
        print(f"⚠️  Router got unexpected response '{raw}', defaulting to 'policy'")
        return "policy"

    print(f"🧭 Router classified query as: '{raw}'")
    return raw
# test_agents.py

from agents import run_query


def test_agents():
    test_queries = [
        "What is the return policy for Waitrose products?",
        "Can we share customer data with third party advertisers?",
        "How do I handle a P1 system incident?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        result = run_query(query)
        print(f"\n📌 Query: {result['query']}")
        print(f"🤖 Agent: {result['agent']}")
        print(f"📂 Sources: {result['sources']}")
        print(f"📊 Chunks used: {result['chunks_used']}")
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    test_agents()
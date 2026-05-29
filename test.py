# test_gemini.py
import sys
from config.settings import GOOGLE_API_KEY, GEMINI_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI

def test_connection():
    print(f"Testing Gemini connection with model: {GEMINI_MODEL}")

    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not found in .env")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    response = llm.invoke("In one sentence, what is RAG in AI?")

    # Handle both string and list response formats
    if isinstance(response.content, list):
        text = response.content[0].get("text", "") if isinstance(response.content[0], dict) else str(response.content[0])
    else:
        text = response.content

    print(f"✅ Gemini connected successfully!")
    print(f"Model: {GEMINI_MODEL}")
    print(f"Response: {text}")

if __name__ == "__main__":
    test_connection()
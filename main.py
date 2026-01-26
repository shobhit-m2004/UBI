import os
from dotenv import load_dotenv
from google import genai

from src.file_loader import load_pdfs
from src.chunker import chunk_documents
from src.search import build_rag_index, retrieve_chunks

# ---------------- CONFIG ----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("API Key missing.")

client = genai.Client(api_key=API_KEY)

# ---------------- RAG INIT ----------------
def init_rag():
    print("Loading documents...")
    documents = load_pdfs("data")
    chunks = chunk_documents(documents)
    vectordb, chunks = build_rag_index(chunks)
    print("RAG index ready.\n")
    return vectordb, chunks

vectordb, chunks = init_rag()

# ---------------- CHAT LOOP ----------------
print("Digital Marketing Professor (CLI)")
print("Type 'exit' or 'quit' to end.\n")


while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye.")
        break


    results = retrieve_chunks(vectordb, chunks, user_input)

    context_text = "\n\n".join(
        f"[{r['source']} pg {r['page']}] {r['text']}"
        for r in results
    )

    full_prompt = f"""
You are a senior professor of Digital Marketing.

Instructions:
- Use ONLY the information provided in the context.
- Reason internally step by step but DO NOT reveal your reasoning.
- Answer strictly in point-based format.
- Do NOT use external knowledge.
- If the answer is not found in the context, respond exactly with:
  "This information is not available in the provided document."
- If the user input is a greeting (e.g., "hello", "hi", "hey"):
  Respond politely and professionally as a Digital Marketing professor.

Context:
{context_text}

User Input:
{user_input}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt
    )

    answer = response.text.strip()

    print("\nProfessor:")
    print(answer)
    print("-" * 60)

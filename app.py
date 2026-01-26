import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

from src.file_loader import load_pdfs
from src.chunker import chunk_documents
from src.search import build_rag_index, retrieve_chunks

# ---------------- CONFIG ----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Digital Marketing Professor",
    layout="centered"
)

if not API_KEY:
    st.error("API Key missing.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ---------------- RAG INIT ----------------
@st.cache_resource
def init_rag():
    documents = load_pdfs("data")
    chunks = chunk_documents(documents)
    vectordb, chunks = build_rag_index(chunks)
    return vectordb, chunks

vectordb, chunks = init_rag()

# ---------------- UI ----------------
st.title("Digital Marketing Professor")

# ---------------- CHAT INPUT ----------------
prompt_input = st.chat_input("Ask a question")

if prompt_input:
    with st.chat_message("user"):
        st.markdown(prompt_input)

    results = retrieve_chunks(vectordb, chunks, prompt_input)

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
{prompt_input}
"""

    with st.chat_message("assistant"):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        st.markdown(response.text)

import os
import numpy as np
from dotenv import load_dotenv
from google import genai

# Load .env file
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)

def embed_text(text: str) -> np.ndarray:
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return np.array(result.embeddings[0].values, dtype="float32")

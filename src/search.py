import numpy as np
from .embedding import embed_text     
from .vectordb import VectorDB          


def build_rag_index(chunks):
    embeddings = [embed_text(c["text"]) for c in chunks]
    embeddings_np = np.array(embeddings, dtype="float32")

    vectordb = VectorDB(embeddings_np.shape[1])
    vectordb.add(embeddings_np)

    return vectordb, chunks

def retrieve_chunks(vectordb, chunks, query, top_k=3):
    query_embedding = embed_text(query).reshape(1, -1)
    indices = vectordb.search(query_embedding, top_k)
    return [chunks[i] for i in indices]

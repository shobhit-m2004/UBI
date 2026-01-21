import tiktoken

# Token-based chunking (safe for LLMs)
def chunk_documents(documents, max_tokens=300, overlap=50):
    """
    documents: list of dicts from PDF loader
    returns: list of chunk dicts
    """
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []

    for doc in documents:
        tokens = enc.encode(doc["text"])
        # print(tokens)
        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            # print(chunk_text)

            if chunk_text.strip():
            
                chunks.append({
                    "text": chunk_text.strip(),
                    "source": doc["source"],
                    "page": doc["page"],
                    "chunk_id": chunk_id
                })
           
                chunk_id += 1

            start = end - overlap 

    return chunks

import os
from pypdf import PdfReader

def load_pdfs(data_dir):
    documents = []
    base_url = "http://127.0.0.1:5501/data"  

    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        # local path for reading
        local_path = os.path.join(data_dir, filename)

        # live server URL for opening
        file_url = f"{base_url}/{filename}"

        reader = PdfReader(local_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append({
                    "text": text.strip(),
                    "source": filename,
                    "page": page_num + 1,
                    "path": f"{file_url}" 
                })

    return documents

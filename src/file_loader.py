import os
from pypdf import PdfReader

def load_pdfs(data_dir):
    """
    Load all PDFs from a directory.
    Returns a list of dicts:
    {
        text: str,
        source: filename,
        page: page_number
    }
    """
    documents = []

    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(data_dir, filename)
        reader = PdfReader(file_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append({
                    "text": text.strip(),
                    "source": filename,
                    "page": page_num + 1
                })

    return documents

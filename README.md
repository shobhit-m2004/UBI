# Digital Marketing Professor

This is a **RAG (Retrieval-Augmented Generation) chat bot** called **Digital Marketing Professor**.  
It allows users to chat with PDF documents and get answers grounded only in the document content.

---

## 🔍 What is RAG?

RAG works by combining **document retrieval** with **AI generation**:

- Documents are loaded and processed
- Relevant content is retrieved based on the user question
- The AI generates an answer using only that retrieved content

---

## ⚙️ How the RAG Process Works

### 1️⃣ Document Loading
- PDF files are loaded from the `data/` folder
- PDFs are read using **PyPDF**

### 2️⃣ Chunking
- Documents are split into smaller text chunks
- Chunking helps improve retrieval accuracy

### 3️⃣ Embedding
- Each chunk is converted into vectors
- Used for semantic similarity search

### 4️⃣ Vector Storage
- Embeddings are stored using **FAISS (CPU)**
- Enables fast similarity-based retrieval

### 5️⃣ Retrieval
- When a user asks a question, relevant chunks are retrieved
- Only the most relevant chunks are passed to the model

### 6️⃣ Answer Generation
- **Google Gemini** generates answers
- Answers are strictly based on retrieved context
- No external or hallucinated information is used

---

## 🛠️ Technologies Used

- **LLM:** Google Gemini
- **Embedding model:** text-embedding-004
- **Vector Database:** FAISS
- **PDF Processing:** PyPDF
- **Frontend:** Streamlit
- **Tokenization:** tiktoken
- **Environment Variables:** dotenv

---

## 📦 Project Setup

### Clone the Repository

```bash
git clone https://github.com/shobhit-m2004/UBI.git
cd UBI
```

### Create `.env` File

```env
GOOGLE_API_KEY=your_api_key_here
```

### Install Dependencies

```bash
uv add -r requirements.txt
```

### Add PDFs

- Place your PDF files inside the `data/` folder

```

### Run the Application

streamlit run app.py

---

## 📌 Notes

- The chatbot answers **only from PDF content**
- If the answer is not found, it clearly says so
- Designed for learning and document-based Q&A


# DocuMind — Document Q&A with LangChain + Gemini

DocuMind is a Streamlit app that lets you upload PDF, DOCX, and TXT files, builds a FAISS vector store over document chunks (with `source` + `page` metadata), and uses Google Gemini to answer questions grounded in your documents. When answers are based on document content, DocuMind includes citations like `(file.pdf - page 3)` and shows a small source list in the UI.

---

## 🚀 Features

- Upload and process multiple **PDF**, **DOCX**, or **TXT** files.
- Extracts text and preserves metadata (`source`, `page`).
- Splits text into chunks with `RecursiveCharacterTextSplitter` (keeps metadata).
- Builds a FAISS vector store using **Sentence-Transformers** embeddings.
- Uses **Google Gemini (Generative AI)** for question answering.
- Displays inline citations and a detailed source list.
- Chat interface with streaming (typing effect).

---

# ContextCore: Chat with your Documents

**ContextCore** is a simple yet powerful **Retrieval-Augmented Generation (RAG)** application.  
It allows you to upload documents and ask questions about their content through a user-friendly chat interface.

---

## 🧠 Overview

This project is built with a modern, decoupled architecture:

- **Backend:** A **FastAPI** server that handles document processing, embedding generation using open-source **Hugging Face** models, and interaction with the **Gemini Large Language Model**.
- **Frontend:** A **Streamlit** application that provides the user interface for file uploads and chat.

---

## ⚙️ How it Works

### 1. Upload
You upload one or more documents (`.pdf`, `.txt`) through the Streamlit frontend.

### 2. Ask
You ask a question in the chat window.

### 3. Process
The Streamlit frontend sends the content of your uploaded documents and your question to the FastAPI backend.

### 4. Retrieve & Augment
The backend performs the **RAG** process for each question:

1. It splits the document text into manageable chunks.  
2. It uses a **free, open-source model** from Hugging Face — [`all-MiniLM-L6-v2`] — to turn those text chunks into numerical representations (vectors).  
   - This runs **locally on the server**, requiring **no extra API key** from you for the embedding model.  
3. It builds a temporary, in-memory **FAISS** vector database for efficient text similarity search.  
4. It searches this database to find the most relevant chunks related to your question.

### 5. Generate
The backend sends the relevant chunks (*context*) and your original question to the **Gemini model**, which generates a well-informed answer based only on the content of your documents.

### 6. Respond
The final answer is sent back to the Streamlit frontend and displayed in the chat interface.

---

## 🧩 Tech Stack

- **Backend:** FastAPI, FAISS, Hugging Face Embeddings, Gemini API
- **Frontend:** Streamlit
- **Deployment:** Render (Backend) + Streamlit Cloud (Frontend)
- **Language:** Python 3.10+

---

> Built with ❤️ using FastAPI, Streamlit, FAISS, Hugging Face, and Gemini.
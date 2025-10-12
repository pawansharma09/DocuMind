# ContextCore: Chat with your Documents

**ContextCore** is a simple yet powerful **Retrieval-Augmented Generation (RAG)** application. It allows you to upload documents and ask questions about their content through a user-friendly chat interface.

---

## 🧠 Overview

This project is built with a modern, decoupled architecture:

- **Backend**: A **FastAPI** server that handles document processing, embedding generation, and interaction with the **Gemini Large Language Model**.
- **Frontend**: A **Streamlit** application that provides the user interface for file uploads and chat.

---

## ⚙️ How it Works

### 1. Upload
You upload one or more documents (`.pdf`, `.txt`) through the Streamlit frontend.

### 2. Ask
You ask a question in the chat window.

### 3. Process
The Streamlit frontend sends the content of your uploaded documents and your question to the FastAPI backend.

### 4. Retrieve & Augment
The backend performs the **RAG process** for each question:

1. It splits the document text into manageable chunks.  
2. It uses **Google’s embedding models** to convert those text chunks into numerical representations (vectors).  
3. It builds a temporary, in-memory vector database (using **FAISS**) to enable efficient searching.  
4. It searches this database to find the most relevant chunks of text related to your question.

### 5. Generate
The backend sends the relevant chunks (the *context*) and your original question to the **Gemini model**, which generates a well-informed answer based solely on the information in your documents.

### 6. Respond
The final answer is sent back to the Streamlit frontend and displayed in the chat interface.

---

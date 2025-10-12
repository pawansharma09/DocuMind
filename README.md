# ✨ AuraRAG - Intelligent Document Q&A

**AuraRAG** is a powerful, yet simple, Retrieval-Augmented Generation (RAG) application.  
It allows you to upload your own documents and ask questions about them, getting intelligent, context-aware answers powered by **Google's Gemini 2.5 Flash** and open-source embedding models.

---

## 🚀 Key Features

- **Modern UI:** A clean and intuitive user interface built with Streamlit.  
- **Flexible Backend:** A scalable and powerful backend using FastAPI.  
- **High-Quality Embeddings:** Utilizes a free, state-of-the-art sentence transformer from Hugging Face.  
- **Blazing-Fast Retrieval:** Employs FAISS for efficient in-memory similarity searches.  
- **Advanced Generation:** Leverages the speed and power of Google's Gemini 2.5 Flash model.  
- **Ready for Deployment:** Includes configurations for deploying the frontend to Streamlit Cloud and the backend to Render.

---

## 🏛️ Project Architecture

The application is split into two main components: **a frontend** and **a backend**.

### **Frontend (Streamlit)**
This is the user-facing part of the application.  
It handles file uploads, user queries, and displays the final answer.  
It communicates with the backend via HTTP requests.

### **Backend (FastAPI)**
This is the core engine.  
It receives the documents and query from the frontend, creates vector embeddings, builds a FAISS index, retrieves relevant context, and uses the Gemini API to generate the final answer.

## 💻 Technologies Used

| **Component** | **Technology** |
|----------------|----------------|
| **Frontend** | Streamlit |
| **Backend** | FastAPI, Uvicorn |
| **LLM** | Google Gemini 2.5 Flash |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Database** | FAISS (In-memory) |
| **Deployment** | Render (Backend), Streamlit Cloud (Frontend) |

---

✨ **AuraRAG** – Bringing intelligence and context to your documents.


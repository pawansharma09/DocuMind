# DocuMind: RAG Chatbot with Gemini & Streamlit

DocuMind is a Retrieval-Augmented Generation (RAG) chatbot built with Python. It allows users to upload multiple documents (PDF, DOCX, TXT) and ask questions about their content through a user-friendly chat interface created with Streamlit.

The backend is powered by LangChain, using Hugging Face embeddings for document vectorization and Google's Gemini 2.5 Flash model for question-answering. The vector search is handled efficiently by FAISS. This project is structured with modularity in mind, making it suitable for a capstone project or further development.

---

## 🚀 Features

-   **Multi-File Upload**: Supports PDF, Microsoft Word (DOCX), and plain text (TXT) files.
-   **Interactive Chat Interface**: A responsive and intuitive UI built with Streamlit.
-   **Modular & Scalable Codebase**: Code is split into logical modules for file handling, vectorization, and LLM interaction.
-   **State-of-the-Art LLM**: Utilizes the powerful and efficient Gemini 2.5 Flash model via the Google Generative AI API for conversational AI.
-   **High-Quality Open-Source Embeddings**: Generates dense vector embeddings using Hugging Face's `all-MiniLM-L6-v2` model.
-   **Fast & Efficient Retrieval**: Uses FAISS (Facebook AI Similarity Search) for quick retrieval of relevant document chunks.
-   **Ready for Deployment**: Can be easily deployed on Streamlit Cloud.

---

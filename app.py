import streamlit as st
import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Function to extract text from various file types
def get_text_from_files(uploaded_files):
    """
    Reads and extracts text from uploaded files (PDF, DOCX, TXT).
    """
    text = ""
    for file in uploaded_files:
        try:
            if file.name.endswith('.pdf'):
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            elif file.name.endswith('.docx'):
                doc = Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            elif file.name.endswith('.txt'):
                text += file.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    return text

# Function to configure the Gemini model at the start
def configure_gemini():
    """
    Configures the Google Generative AI model with the API key from Streamlit secrets.
    """
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        if not google_api_key:
            st.error("Google API Key not found. Please add it to your Streamlit secrets.")
            return None
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        return None

# A simple text splitter function
def get_text_chunks(raw_text, chunk_size=1000, chunk_overlap=200):
    """
    Splits the extracted text into smaller, overlapping chunks.
    """
    if not raw_text:
        return []
    chunks = []
    start = 0
    while start < len(raw_text):
        end = start + chunk_size
        chunks.append(raw_text[start:end])
        # Move start position for the next chunk, considering the overlap
        start += chunk_size - chunk_overlap
    return chunks

# Function to create embeddings and a vector store without LangChain
def create_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks using SentenceTransformer directly.
    """
    if not text_chunks:
        st.warning("No text to process. Please upload and process valid documents.")
        return None, None
    try:
        # Load the sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Generate embeddings for each chunk of text
        embeddings = model.encode(text_chunks, show_progress_bar=True)
        # Create a FAISS index for similarity search
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))
        return index, model
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None, None

# Function to generate a response from Gemini without LangChain
def get_gemini_response(gemini_model, user_question, faiss_index, embedding_model, text_chunks):
    """
    Finds relevant context from the vector store and generates a response using Gemini.
    """
    if not user_question:
        return ""

    # 1. Find relevant context from the uploaded documents
    # Embed the user's question using the same model
    question_embedding = embedding_model.encode([user_question])
    # Search the FAISS index for the top 'k' most similar chunks
    k = 5
    distances, indices = faiss_index.search(np.array(question_embedding, dtype=np.float32), k)
    
    # Retrieve the actual text content of the most relevant chunks
    context = " ".join([text_chunks[i] for i in indices[0]])

    # 2. Construct the prompt for the LLM
    # Get the chat history from the session state
    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.get('chat_history', [])])

    prompt = f"""
    You are DocuMind, an intelligent assistant. Use the provided context from the documents to answer the user's question.
    Your goal is to be as helpful as possible.

    If the answer is in the context, provide a detailed answer based on it.
    If the answer is not found in the context, use your general knowledge to answer, but you MUST explicitly state: "This answer is from my general knowledge base as I couldn't find the information in the provided document(s)."
    
    CONTEXT:
    {context}

    CHAT HISTORY:
    {chat_history_str}

    USER QUESTION:
    {user_question}

    Helpful Answer:
    """

    # 3. Call the Gemini API to generate the final response
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        return "Sorry, I encountered an error while generating a response."

# Main Streamlit app
def main():
    st.set_page_config(page_title="DocuMind (No LangChain)", page_icon="🧠", layout="wide")

    # Configure the Gemini model once at the start
    gemini_model = configure_gemini()

    # --- Initializing Session State ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store_data" not in st.session_state:
        st.session_state.vector_store_data = {'index': None, 'embedding_model': None, 'chunks': None}

    # --- Sidebar for File Upload and Processing ---
    with st.sidebar:
        st.title("DocuMind 🧠")
        st.markdown("Your document assistant **(No LangChain)**. Upload your files and ask questions.")
        
        uploaded_files = st.file_uploader(
            "Upload your Documents (PDF, DOCX, TXT)",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt']
        )
        
        if st.button("Process Documents", use_container_width=True, type="primary"):
            if uploaded_files:
                with st.spinner("Reading files..."):
                    raw_text = get_text_from_files(uploaded_files)
                with st.spinner("Chunking text..."):
                    text_chunks = get_text_chunks(raw_text)
                with st.spinner("Creating vector store... This may take a moment."):
                    faiss_index, embedding_model = create_vector_store(text_chunks)
                    if faiss_index is not None and embedding_model is not None:
                        st.session_state.vector_store_data['index'] = faiss_index
                        st.session_state.vector_store_data['embedding_model'] = embedding_model
                        st.session_state.vector_store_data['chunks'] = text_chunks
                        # Clear previous chat history on new document processing
                        st.session_state.chat_history = []
                        st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one document.")

    # --- Main Chat Interface ---
    st.header("Ask DocuMind Anything")
    
    if not st.session_state.vector_store_data['index']:
        st.info("Welcome! Please upload and process your documents in the sidebar to begin.")

    # Display chat history from session state
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Handle user input
    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        if st.session_state.vector_store_data['index'] and gemini_model:
            # Add user message to history and display it
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_question)

            # Get and display assistant's response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    response = get_gemini_response(
                        gemini_model=gemini_model,
                        user_question=user_question,
                        faiss_index=st.session_state.vector_store_data['index'],
                        embedding_model=st.session_state.vector_store_data['embedding_model'],
                        text_chunks=st.session_state.vector_store_data['chunks']
                    )
                    st.markdown(response)
                    # Add assistant response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.warning("Please process your documents first or ensure the API key is configured correctly.")

if __name__ == "__main__":
    main()


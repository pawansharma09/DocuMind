import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_text_chunks(text):
    """
    Splits a long string of text into smaller, manageable chunks.

    Args:
        text (str): The input text to be split.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks using Hugging Face embeddings.

    Args:
        text_chunks (list): A list of text chunks.

    Returns:
        FAISS: A FAISS vector store object, or None if creation fails.
    """
    if not text_chunks:
        st.warning("No text chunks to process. Cannot create vector store.")
        return None
    try:
        # Using a popular sentence-transformer model for embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        # The HuggingFaceEmbeddings model will be downloaded on the first run, 
        # which might take time and requires an internet connection.
        st.error(f"Error creating vector store with Hugging Face embeddings: {e}")
        st.info("Please ensure you have a stable internet connection for the initial model download.")
        return None


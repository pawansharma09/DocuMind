import streamlit as st
import os
from dotenv import load_dotenv
from src.file_handler import get_text_from_files
from src.vector_store_handler import get_text_chunks, get_vector_store
from src.llm_handler import get_conversational_chain

def setup_api_keys():
    """Load and configure API keys from .env or Streamlit secrets."""
    load_dotenv()

    # Configure Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        try:
            google_api_key = st.secrets["GOOGLE_API_KEY"]
        except (KeyError, FileNotFoundError):
            st.error("Google API Key not found. Please set it in your environment or Streamlit secrets.")
            st.stop()

    # Configure Hugging Face API token for embeddings and other models
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_token:
        try:
            hf_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        except (KeyError, FileNotFoundError):
            st.error("Hugging Face API Token not found. Please set it in your environment or Streamlit secrets.")
            st.stop() # Stop execution as it's required for embeddings
            
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token
    
    return google_api_key

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="DocuMind Chatbot", page_icon="🧠", layout="wide")
    
    google_api_key = setup_api_keys()

    # --- Sidebar for File Upload ---
    with st.sidebar:
        st.title("Your Documents")
        uploaded_files = st.file_uploader(
            "Upload your PDF, DOCX, or TXT files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt']
        )
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents... This may take a moment."):
                    raw_text = get_text_from_files(uploaded_files)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        # The vector store now uses Hugging Face embeddings, no API key needed here
                        vector_store = get_vector_store(text_chunks)
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.success("Documents processed successfully!")
                        else:
                            st.error("Failed to create vector store.")
                    else:
                        st.warning("No text could be extracted from the uploaded files.")
            else:
                st.warning("Please upload at least one document.")

    # --- Main Chat Interface ---
    st.title("DocuMind: Chat with your Documents 🧠")
    st.write("Upload your documents on the sidebar and ask questions about them.")

    # Initialize or load chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if "vector_store" not in st.session_state:
            st.warning("Please upload and process your documents before asking questions.")
        else:
            with st.spinner("Finding answer..."):
                try:
                    vector_store = st.session_state.vector_store
                    docs = vector_store.similarity_search(prompt)
                    chain = get_conversational_chain(google_api_key)
                    response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
                    answer = response["output_text"]

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")

if __name__ == "__main__":
    main()




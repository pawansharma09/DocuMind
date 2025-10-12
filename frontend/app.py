import streamlit as st
import requests
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AuraRAG ✨",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Backend URL ---
# Attempt to get the backend URL from Streamlit secrets
try:
    BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
except:
    BACKEND_URL = "http://localhost:8000"


# --- UI Styling ---
st.markdown("""
<style>
    .stApp {
        background: #F0F2F6;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 95%;
    }
    .st-emotion-cache-1v0mbdj {
        border: 2px solid #E6EAF1;
        border-radius: 0.5rem;
        padding: 1.5rem;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #4B8BF5;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        border: none;
        padding: 0.75rem 1.5rem;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #3A77D9;
    }
    .stTextInput>div>div>input {
        background-color: #F8F9FA;
    }
    .stFileUploader>div>div>button {
        background-color: #F8F9FA;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("AuraRAG ✨ - Intelligent Document Q&A")
st.markdown("Upload your documents, ask a question, and get smart, context-aware answers from your own knowledge base.")

# --- Main Application ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Your Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload your TXT or MD files here.",
        type=['txt', 'md'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    documents_text = []
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) uploaded:**")
        file_names = " | ".join([file.name for file in uploaded_files])
        st.info(f"`{file_names}`")
        for uploaded_file in uploaded_files:
            try:
                # To read as string, decode bytes to utf-8.
                string_data = uploaded_file.getvalue().decode("utf-8")
                documents_text.append(string_data)
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")

with col2:
    st.subheader("❓ Ask a Question")
    query = st.text_input(
        "Enter your question about the documents.",
        placeholder="e.g., What are the main findings?",
        label_visibility="collapsed"
    )

    if st.button("Generate Answer", use_container_width=True, type="primary"):
        if not query:
            st.warning("Please enter a question.")
        elif not documents_text:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("AuraRAG is thinking... This may take a moment. 🧠"):
                try:
                    payload = {"query": query, "documents": documents_text}
                    response = requests.post(f"{BACKEND_URL}/process", json=payload)

                    if response.status_code == 200:
                        st.subheader("💡 Answer")
                        answer_container = st.container(border=True)
                        answer_container.markdown(response.json().get("answer"))
                    else:
                        st.error(f"An error occurred: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the backend: {e}. Make sure the backend service is running and the URL is correct.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Powered by **Google Gemini 2.5 Flash**, **Hugging Face Transformers**, and **FAISS**.")

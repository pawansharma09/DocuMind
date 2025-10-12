import streamlit as st
import requests
import os
from pypdf import PdfReader
import io
import uuid

# --- Page Configuration ---
st.set_page_config(
    page_title="ContextCore",
    page_icon="🧠",
    layout="wide",
)

# --- Backend API URL ---
# Try to get the backend URL from Streamlit secrets, otherwise default to localhost.
# This makes the app adaptable for both local development and cloud deployment.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/chat")

# --- Session State Initialization ---
# Initialize session state variables to preserve data across user interactions.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_content" not in st.session_state:
    st.session_state.uploaded_files_content = None

# --- Helper Functions ---
def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file provided as bytes."""
    try:
        pdf = PdfReader(io.BytesIO(file_bytes))
        text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def extract_text_from_txt(file_bytes):
    """Extracts text from a TXT file provided as bytes."""
    try:
        return file_bytes.decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return None

# --- UI Layout ---
st.title("🧠 ContextCore: Chat with your Documents")
st.markdown("Upload your documents, ask questions, and get answers directly from the content.")

with st.sidebar:
    st.header("1. Upload Documents")
    st.markdown("Upload one or more PDF or TXT files. The content will be processed to answer your questions.")
    
    uploaded_files = st.file_uploader(
        "Choose your documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        all_texts = []
        for file in uploaded_files:
            file_bytes = file.getvalue()
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file_bytes)
            elif file.type == "text/plain":
                text = extract_text_from_txt(file_bytes)
            else:
                text = None
            
            if text:
                all_texts.append(text)
                st.success(f"✅ Successfully processed `{file.name}`")
            else:
                st.error(f"❌ Could not extract text from `{file.name}`")
        
        # Store the extracted text in session state if any was found.
        if all_texts:
            st.session_state.uploaded_files_content = all_texts

    st.markdown("---")
    st.info("Your documents are processed for each question and are not stored on the server.")

# --- Main Chat Interface ---
st.header("2. Ask a Question")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask something about your documents..."):
    # Add user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if documents have been uploaded before proceeding
    if st.session_state.uploaded_files_content is None:
        st.warning("Please upload at least one document before asking a question.", icon="⚠️")
    else:
        # Prepare the request payload for the backend
        api_payload = {
            "question": prompt,
            "files_content": st.session_state.uploaded_files_content,
            "session_id": st.session_state.session_id,
        }

        # Send request to the backend and display the response
        try:
            with st.spinner("Thinking..."):
                response = requests.post(BACKEND_URL, json=api_payload, timeout=120)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                
                answer = response.json().get("answer", "No answer found.")

                # Add assistant's response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the backend. Please ensure it is running. Error: {e}")



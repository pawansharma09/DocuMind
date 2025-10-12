import streamlit as st
import requests
import pypdf
import io
import uuid

# --- Page Configuration ---
st.set_page_config(
    page_title="ContextCore",
    page_icon="🧠",
    layout="wide"
)

# --- Backend API URL ---
# Fetch the backend URL from Streamlit secrets
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except KeyError:
    st.error("BACKEND_URL secret not found! Please set it in your Streamlit Cloud settings.")
    st.stop()

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def extract_text_from_txt(txt_file):
    """Extracts text from an uploaded TXT file."""
    try:
        return txt_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return None

def call_chat_api(question, files_content, session_id):
    """Sends a request to the FastAPI backend and gets the response."""
    endpoint = f"{BACKEND_URL}/chat"
    payload = {
        "question": question,
        "files_content": files_content,
        "session_id": session_id
    }
    try:
        with st.spinner('Thinking...'):
            response = requests.post(endpoint, json=payload, timeout=120) # Increased timeout
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        return response.json()["answer"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the backend. Please ensure it's running. Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files_content" not in st.session_state:
    st.session_state.uploaded_files_content = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "files_processed" not in st.session_state:
    st.session_state.files_processed = False


# --- UI Layout ---

# Header
st.title("🧠 ContextCore: Chat with Your Documents")
st.markdown("Upload your documents, and I'll answer questions based on their content.")

# Sidebar for file uploads
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if st.button("Process Documents", key="process_button"):
        if uploaded_files:
            st.session_state.uploaded_files_content = []
            with st.spinner("Processing files..."):
                for file in uploaded_files:
                    file_extension = file.name.split('.')[-1].lower()
                    if file_extension == "pdf":
                        text = extract_text_from_pdf(io.BytesIO(file.getvalue()))
                    elif file_extension == "txt":
                        text = extract_text_from_txt(file)
                    else:
                        st.warning(f"Unsupported file type: {file.name}")
                        continue

                    if text:
                        st.session_state.uploaded_files_content.append(text)

            if st.session_state.uploaded_files_content:
                st.session_state.files_processed = True
                st.success("Documents processed successfully! You can now ask questions.")
                # Clear previous chat history on new document processing
                st.session_state.messages = []
            else:
                st.error("Could not extract text from any of the uploaded files.")
        else:
            st.warning("Please upload at least one document.")

# Main chat interface
st.header("2. Ask Questions")

# Display a welcome message if no documents are processed yet
if not st.session_state.files_processed:
    st.info("Please upload and process your documents using the sidebar to begin.")
else:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response from the backend
        with st.chat_message("assistant"):
            answer = call_chat_api(
                prompt,
                st.session_state.uploaded_files_content,
                st.session_state.session_id
            )
            if answer:
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Failed to get a response from the backend.")

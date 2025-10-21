import streamlit as st
import os
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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

# Function to split text into manageable chunks
def get_text_chunks(raw_text):
    """
    Splits the extracted text into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Function to create and save a vector store from text chunks
def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks using Sentence Transformer embeddings.
    """
    if not text_chunks:
        st.warning("No text to process. Please upload and process valid documents.")
        return None
    try:
        # Using a popular and efficient sentence transformer model
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

# Function to create a conversational chain
def get_conversational_chain(vector_store):
    """
    Creates a conversational retrieval chain with memory.
    """
    try:
        # Access the API key from Streamlit secrets
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        if not google_api_key:
            st.error("Google API Key not found. Please add it to your Streamlit secrets.")
            return None

        llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.4)
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Custom prompt template
        custom_prompt_template = """
        You are DocuMind, an intelligent assistant. Use the provided context from the documents to answer the user's question.
        Your goal is to be as helpful as possible.

        If the answer is in the context, provide a detailed answer based on it.
        If the answer is not found in the context, use your general knowledge to answer, but you MUST explicitly state: "This answer is from my general knowledge base as I couldn't find the information in the provided document(s)."
        
        Context:
        {context}

        Chat History:
        {chat_history}

        Question:
        {question}

        Helpful Answer:
        """
        
        PROMPT = PromptTemplate(
            template=custom_prompt_template, input_variables=["context", "chat_history", "question"]
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        return None

# Function to handle user input and display conversation
def handle_user_input(user_question):
    """
    Processes user's question and updates the chat history.
    """
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                with st.chat_message("user", avatar="👤"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message.content)
    else:
        st.warning("Please process your documents first.")

# Main Streamlit app
def main():
    st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")

    # --- Initializing Session State ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # --- Sidebar for File Upload and Processing ---
    with st.sidebar:
        st.title("DocuMind 🧠")
        st.markdown("Your intelligent document assistant. Upload your files and ask questions.")
        
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
                    vector_store = get_vector_store(text_chunks)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.conversation = get_conversational_chain(st.session_state.vector_store)
                        st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one document.")

    # --- Main Chat Interface ---
    st.header("Ask DocuMind Anything")
    
    if st.session_state.vector_store is None:
        st.info("Welcome to DocuMind! Please upload and process your documents in the sidebar to begin.")

    # Display chat history
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                 with st.chat_message("user", avatar="👤"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message.content)

    # Chat input
    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        if st.session_state.conversation:
             with st.chat_message("user", avatar="👤"):
                st.write(user_question)
             with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation({'question': user_question})
                    st.session_state.chat_history = response['chat_history']
                    st.write(response['chat_history'][-1].content)
        else:
            st.warning("Please process your documents before asking questions.")


if __name__ == "__main__":
    main()

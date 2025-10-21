import streamlit as st
import os
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
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

# Function to split text into manageable chunks using LangChain
def get_text_chunks(raw_text):
    """
    Splits the extracted text into smaller, overlapping chunks using LangChain's splitter.
    """
    if not raw_text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Function to create and save a vector store from text chunks using LangChain
def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks using LangChain wrappers.
    """
    if not text_chunks:
        st.warning("No text to process. Please upload and process valid documents.")
        return None
    try:
        # Using a popular and efficient sentence transformer model via LangChain
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

# Function to create a conversational chain using LangChain
def get_conversational_chain(vector_store):
    """
    Sets up the conversational retrieval chain with memory and a custom prompt.
    """
    try:
        # Access the API key from Streamlit secrets
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        if not google_api_key:
            st.error("Google API Key not found. Please add it to your Streamlit secrets.")
            return None

        llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.3)
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Custom prompt template to guide the LLM
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

# Main Streamlit app
def main():
    st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")

    # --- Initializing Session State ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

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
                        # Create the conversation chain and store it in the session state
                        st.session_state.conversation = get_conversational_chain(vector_store)
                        st.session_state.chat_history = None # Reset chat history
                        st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one document.")

    # --- Main Chat Interface ---
    st.header("Ask DocuMind Anything")
    
    if st.session_state.conversation is None:
        st.info("Welcome to DocuMind! Please upload and process your documents in the sidebar to begin.")

    # Display chat history
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            # User messages are at even indices, AI messages at odd indices
            if i % 2 == 0:
                 with st.chat_message("user", avatar="👤"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message.content)

    # Chat input at the bottom
    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        if st.session_state.conversation:
             with st.chat_message("user", avatar="👤"):
                st.write(user_question)
             with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    # Pass the question to the conversation chain
                    response = st.session_state.conversation({'question': user_question})
                    # Update the chat history in the session state
                    st.session_state.chat_history = response['chat_history']
                    # Display the latest assistant message
                    st.write(response['chat_history'][-1].content)
        else:
            st.warning("Please process your documents before asking questions.")

if __name__ == "__main__":
    main()


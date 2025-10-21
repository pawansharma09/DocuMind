import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile

# --- PROMPT TEMPLATE ---
# This template guides the language model to use the provided context and handle cases where the answer is not found.
PROMPT_TEMPLATE = """
Answer the question based only on the following context. If the answer is not in the context, state that the information is not in the provided documents and then provide an answer based on your general knowledge.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Helpful Answer:
"""

def get_google_api_key():
    """
    Retrieves the Google API key from Streamlit secrets or a sidebar input.
    For production, it's recommended to use st.secrets.
    """
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    else:
        return st.sidebar.text_input("Enter your Google API Key:", type="password")

@st.cache_resource
def get_embeddings_model():
    """
    Loads the sentence-transformer model for creating text embeddings.
    Using st.cache_resource to load this once and reuse across sessions.
    """
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

def get_text_from_documents(uploaded_files):
    """
    Extracts text from uploaded files (PDF, DOCX, TXT).
    Handles files by saving them temporarily and using the appropriate loader.
    """
    text = ""
    for uploaded_file in uploaded_files:
        try:
            # Create a temporary file to preserve the original file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Determine the loader based on the file extension
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(tmp_file_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
                continue

            # Load and concatenate text
            documents = loader.load()
            for doc in documents:
                text += doc.page_content + "\n"

            # Clean up the temporary file
            os.remove(tmp_file_path)

        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
    return text

def get_text_chunks(text):
    """
    Splits a long text into smaller, manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, embeddings_model):
    """
    Creates a FAISS vector store from text chunks and an embeddings model.
    """
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings_model)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def get_conversation_chain(vectorstore, llm):
    """
    Creates a conversational retrieval chain.
    This chain combines the language model with the vector store for RAG.
    """
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=PROMPT_TEMPLATE
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        output_key='answer'
    )
    return conversation_chain

def handle_user_input(user_question):
    """
    Processes user input, generates a response, and updates the chat history.
    """
    if st.session_state.conversation:
        try:
            response = st.session_state.conversation({
                'question': user_question,
                'chat_history': st.session_state.get('chat_history', [])
            })
            st.session_state.chat_history.append({'question': user_question, 'answer': response['answer']})

            # Display the response with sources
            st.chat_message("assistant").write(response['answer'])
            with st.expander("Sources"):
                for source in response['source_documents']:
                    st.write(f"- {source.metadata.get('source', 'Unknown')}: ...{source.page_content[:150]}...")
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
    else:
        st.warning("Please upload and process documents first.")

def main():
    """
    The main function that runs the Streamlit application.
    """
    st.set_page_config(page_title="Document Chatbot", page_icon=":books:")
    st.header("Chat with your Documents 📚")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for configuration and file upload
    with st.sidebar:
        st.subheader("Configuration")
        api_key = get_google_api_key()
        if not api_key:
            st.info("Please add your Google API key to continue.")
            st.stop()

        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, DOCX, TXT)",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt']
        )

        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                with st.spinner("Processing documents... This might take a moment."):
                    # 1. Get text from documents
                    raw_text = get_text_from_documents(uploaded_files)
                    if not raw_text:
                        st.error("No text could be extracted from the documents.")
                        st.stop()

                    # 2. Split text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # 3. Load embeddings model
                    embeddings_model = get_embeddings_model()

                    # 4. Create vector store
                    vectorstore = get_vector_store(text_chunks, embeddings_model)
                    if vectorstore:
                        # 5. Initialize LLM
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

                        # 6. Create conversation chain and store in session state
                        st.session_state.conversation = get_conversation_chain(vectorstore, llm)
                        st.success("Documents processed successfully! You can now ask questions.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message['question'])
        with st.chat_message("assistant"):
            st.write(message['answer'])

    # Chat input for user questions
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        handle_user_input(user_question)

if __name__ == '__main__':
    main()

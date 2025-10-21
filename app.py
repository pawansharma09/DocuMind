# app.py (updated Oct 21, 2025)
import os
from typing import List

import streamlit as st
from dotenv import load_dotenv

# File reading
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

# LangChain modern imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LC_Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Google Gemini chat model (langchain-google-genai wrapper)
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------
# Helpers: file -> text
# ---------------------------
def extract_text_from_files(uploaded_files) -> str:
    txt = []
    for f in uploaded_files:
        name = getattr(f, "name", "file")
        try:
            if name.lower().endswith(".pdf"):
                reader = PdfReader(f)
                for p in reader.pages:
                    page_text = p.extract_text()
                    if page_text:
                        txt.append(page_text)
            elif name.lower().endswith(".docx"):
                doc = DocxDocument(f)
                for para in doc.paragraphs:
                    if para.text:
                        txt.append(para.text)
            elif name.lower().endswith(".txt"):
                # Streamlit uploaded file supports getvalue()
                raw = f.getvalue()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="ignore")
                txt.append(raw)
            else:
                st.warning(f"Unsupported file type: {name}")
        except Exception as e:
            st.error(f"Error reading {name}: {e}")
            continue
    return "\n\n".join(txt)

# ---------------------------
# Helpers: chunk -> FAISS
# ---------------------------
def text_to_documents(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[LC_Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        meta = {"chunk": i}
        docs.append(LC_Document(page_content=chunk, metadata=meta))
    return docs

def build_vectorstore(docs: List[LC_Document], hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if not docs:
        return None
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)
    # Create FAISS index from Documents
    faiss_index = FAISS.from_documents(docs, embeddings)
    return faiss_index

# ---------------------------
# Helpers: LLM chain
# ---------------------------
def get_retrieval_chain(google_api_key: str, temperature: float = 0.2):
    """
    Build a RetrievalQA chain backed by Gemini (via langchain-google-genai).
    """
    # model selection - pick a recent, performant Gemini family model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature, google_api_key=google_api_key)

    # Prompt template: keep retrieval-focused instructions
    prompt_template = """You are a helpful assistant. Use only the provided context to answer the user's question.
If the answer is not contained in the context, start your response with:
"This information is not available in the provided documents. Based on my general knowledge, ..."
and then answer concisely.
Context:
{context}

Question:
{question}

Answer:"""

    # Wrap LangChain prompt (RetrievalQA will insert retrieved docs into {context})
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Build chain - use a simple "stuff" chain type for compactness; you can switch to "map_reduce" or "refine" as needed.
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=None,  # we will attach retriever at query time
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False,
    )
    return qa

# ---------------------------
# Streamlit App
# ---------------------------
def setup_api_keys():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
    if not google_api_key:
        st.error("Google API key missing. Set GOOGLE_API_KEY in .env or Streamlit secrets.")
        st.stop()
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    return google_api_key

def main():
    st.set_page_config(page_title="DocuMind (updated)", layout="wide")
    st.title("DocuMind — RAG ChatBot")

    google_api_key = setup_api_keys()

    with st.sidebar:
        st.header("Upload documents")
        uploaded_files = st.file_uploader("PDF / DOCX / TXT", accept_multiple_files=True, type=["pdf", "docx", "txt"])
        chunk_size = st.number_input("Chunk size (chars)", value=1500, step=500)
        chunk_overlap = st.number_input("Chunk overlap (chars)", value=200, step=50)
        st.write("---")
        process_btn = st.button("Create vector store")

        st.write("Model / Embedding settings")
        hf_model = st.text_input("Hugging Face embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
        llm_temp = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2)

    # session state for vector store and docs
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Process documents
    if process_btn:
        if not uploaded_files:
            st.sidebar.warning("Please upload at least one file.")
        else:
            with st.sidebar.spinner("Extracting text and building index..."):
                full_text = extract_text_from_files(uploaded_files)
                if not full_text.strip():
                    st.sidebar.error("No text could be extracted from the uploaded files.")
                else:
                    docs = text_to_documents(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    vs = build_vectorstore(docs, hf_model=hf_model)
                    if vs:
                        st.session_state.vector_store = vs
                        st.session_state.docs = docs
                        st.sidebar.success("Vector store created (in-memory FAISS).")
                    else:
                        st.sidebar.error("Failed to create vector store.")

    # Chat UI area
    st.subheader("Chat with your documents")
    if st.session_state.messages:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.vector_store is None:
            st.warning("Please upload and process documents first (sidebar).")
        else:
            with st.spinner("Retrieving relevant passages and asking Gemini..."):
                try:
                    # create chain (we attach retriever from vector store)
                    qa = get_retrieval_chain(google_api_key=google_api_key, temperature=llm_temp)
                    retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                    # Assign retriever to the chain (works because from_chain_type created an object expecting a retriever)
                    qa.retriever = retriever

                    # Run query
                    result = qa({"query": prompt})
                    answer = result.get("result") or result.get("answer") or result.get("output_text") or ""
                    source_docs = result.get("source_documents", [])

                    # show answer
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)

                    # show short sources (optional)
                    if source_docs:
                        st.markdown("**Source snippets:**")
                        for sd in source_docs[:3]:
                            snippet = sd.page_content
                            md = snippet if len(snippet) < 800 else snippet[:800] + "..."
                            st.caption(md)
                except Exception as e:
                    st.error(f"Error while generating answer: {e}")

if __name__ == "__main__":
    main()

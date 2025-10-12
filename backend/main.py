import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- Configuration ---
# Load API keys from environment variables
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
except KeyError:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

# Hugging Face token is optional for many models, but good practice to have
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
if HUGGING_FACE_HUB_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_HUB_TOKEN


# --- Pydantic Models for API ---
class ChatRequest(BaseModel):
    """Defines the structure for a chat request."""
    question: str = Field(..., description="The user's question.")
    files_content: List[str] = Field(..., description="A list containing the text content of all uploaded files.")
    session_id: str = Field(..., description="A unique identifier for the user session.")

class ChatResponse(BaseModel):
    """Defines the structure for a chat response."""
    answer: str = Field(..., description="The generated answer from the language model.")

# --- FastAPI Application ---
app = FastAPI(
    title="ContextCore API",
    description="Backend API for the ContextCore RAG application.",
    version="1.0.0"
)

# --- Core RAG Logic ---
# Initialize models once to be reused across requests
# Using a cache for the embedding model to avoid reloading it on every call.
# This assumes the server has enough memory. For Render's free tier, this is generally fine.
embedding_model = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large",
    # Set cache_folder to a directory that persists if needed, or leave as default
    # for in-memory caching during the server's lifetime.
    model_kwargs={'device': 'cpu'} # Explicitly use CPU
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

def process_rag_request(question: str, files_content: List[str]) -> str:
    """
    Handles the entire RAG process for a given request.
    This function is called for every incoming chat message.
    """
    if not files_content:
        raise ValueError("No file content provided.")

    # 1. Combine all file content and create Document objects
    combined_text = "\n\n--- NEW DOCUMENT ---\n\n".join(files_content)
    documents = [Document(page_content=combined_text)]

    # 2. Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(documents)

    if not split_documents:
        return "Could not process the document content. It might be empty or too short."

    # 3. Create an in-memory vector store using FAISS with the pre-loaded embedding model
    try:
        vector_store = FAISS.from_documents(split_documents, embedding_model)
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return "There was an error processing your documents with the vector store."

    # 4. Create a retriever to search the vector store
    retriever = vector_store.as_retriever()

    # 5. Define the prompt template for the LLM
    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent assistant for question-answering tasks.
    Answer the user's question based only on the following context.
    If the context doesn't contain the answer, state that you cannot find the answer in the provided documents.
    Be concise and helpful.

    Context:
    {context}

    Question:
    {input}
    """)

    # 6. Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 7. Retrieve relevant documents and generate the answer
    retrieved_docs = retriever.invoke(question)
    response = document_chain.invoke({
        "input": question,
        "context": retrieved_docs
    })

    return response

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    """
    Main endpoint to handle chat requests. It orchestrates the RAG process.
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not request.files_content:
        raise HTTPException(status_code=400, detail="No files were provided.")

    try:
        answer = process_rag_request(request.question, request.files_content)
        return ChatResponse(answer=answer)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/")
def read_root():
    return {"status": "ContextCore backend is running."}


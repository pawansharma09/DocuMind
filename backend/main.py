import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Import modern LangChain components (as of late 2025)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    # This provides a clear error message if the key isn't set.
    raise RuntimeError("FATAL: GOOGLE_API_KEY environment variable not set.")

# --- Pydantic Models for API Data Validation ---
class ChatRequest(BaseModel):
    """Defines the structure for a chat request from the frontend."""
    question: str = Field(..., description="The user's question.", min_length=1)
    files_content: List[str] = Field(..., description="A list of text content from all uploaded files.")
    session_id: str = Field(..., description="A unique identifier for the user session.")

class ChatResponse(BaseModel):
    """Defines the structure for a chat response sent to the frontend."""
    answer: str = Field(..., description="The generated answer from the language model.")

# --- FastAPI Application Setup ---
app = FastAPI(
    title="ContextCore API",
    description="Backend API for the ContextCore RAG application. Uses all-MiniLM-L6-v2 for embeddings.",
    version="1.1.0"
)

# --- Core RAG Components (Initialized Globally) ---
# Initialize models once to be reused across all API requests for efficiency.
# This prevents reloading the models every time a user asks a question.

# 1. Embedding Model: Using a lightweight and efficient model from Hugging Face.
#    This model runs locally on the server's CPU.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} # Explicitly use CPU
)

# 2. Language Model: Using Gemini 1.5 Flash for fast and powerful generation.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# 3. Prompt Template: Defines the instructions for the LLM.
prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant for question-answering tasks.
Answer the user's question based *only* on the context provided below.
If the information to answer the question is not in the context, state that you cannot find the answer in the provided documents.
Be concise and helpful.

Context:
{context}

Question:
{input}
""")

# --- RAG Processing Logic ---
def process_rag_request(question: str, files_content: List[str]) -> str:
    """
    Handles the entire RAG process for a given request. This is the core logic.
    """
    if not files_content or not any(files_content):
        raise ValueError("No file content was provided for processing.")

    # 1. Combine content from all files into a single text block.
    combined_text = "\n\n--- NEW DOCUMENT ---\n\n".join(files_content)
    documents = [Document(page_content=combined_text)]

    # 2. Split the combined text into smaller, manageable chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_documents = text_splitter.split_documents(documents)

    if not split_documents:
        return "The document content was too short to be processed. Please upload a more substantial document."

    # 3. Create an in-memory vector store from the chunks using the embedding model.
    #    FAISS is used for its speed and efficiency.
    try:
        vector_store = FAISS.from_documents(split_documents, embedding_model)
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        return "An error occurred while processing your documents. Please try again."

    # 4. Create a retriever to search the vector store for relevant chunks.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

    # 5. Create the processing chain that combines the prompt, LLM, and document formatting.
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 6. Retrieve the most relevant documents based on the user's question.
    retrieved_docs = retriever.invoke(question)
    
    # 7. Pass the retrieved documents and the question to the LLM to generate a final answer.
    response = document_chain.invoke({
        "input": question,
        "context": retrieved_docs
    })

    return response

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    """
    Main endpoint to handle user chat requests. It orchestrates the RAG process.
    """
    try:
        answer = process_rag_request(request.question, request.files_content)
        return ChatResponse(answer=answer)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"An unexpected server error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred. Please try again later.")

@app.get("/", summary="Health Check")
def read_root():
    """Provides a simple health check endpoint to confirm the server is running."""
    return {"status": "ContextCore backend is running successfully."}



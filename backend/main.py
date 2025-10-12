from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re

# --- Configuration ---
# It's recommended to set the API key as an environment variable
# for security reasons.
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")


# Initialize the models
# The LLM model for generation.
llm = genai.GenerativeModel('gemini-2.5-flash')

# The embedding model from Hugging Face.
# 'all-MiniLM-L6-v2' is a good balance of speed and performance.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Pydantic Models for API validation ---
class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    documents: List[str] = Field(..., min_length=1, description="A list of documents to search.")

class RAGResponse(BaseModel):
    answer: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AuraRAG Backend",
    description="API for Retrieval-Augmented Generation using Gemini and FAISS.",
    version="1.0.0"
)

# --- Helper Functions ---
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Splits text into smaller, overlapping chunks."""
    # Simple whitespace splitting
    words = re.split(r'\s+', text)
    if not words:
        return []
    
    chunks = []
    current_pos = 0
    while current_pos < len(words):
        start = current_pos
        end = current_pos + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        current_pos += chunk_size - overlap
    return [chunk for chunk in chunks if chunk.strip()]


# --- API Endpoint ---
@app.post("/process", response_model=RAGResponse)
async def process_rag_query(request: RAGRequest):
    """
    Processes a RAG query by embedding documents, finding relevant chunks,
    and generating an answer using the Gemini model.
    """
    try:
        # 1. Combine and Chunk Documents
        full_text = "\n\n".join(request.documents)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Documents cannot be empty.")
            
        chunks = chunk_text(full_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to create text chunks from documents.")

        # 2. Generate Embeddings for Chunks
        chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
        
        # 3. Create FAISS Index
        d = chunk_embeddings.shape[1]  # Dimension of embeddings
        index = faiss.IndexFlatL2(d)
        index.add(np.array(chunk_embeddings).astype('float32'))
        
        # 4. Embed the Query
        query_embedding = embedding_model.encode([request.query], convert_to_tensor=False)

        # 5. Search for Relevant Chunks
        k = min(5, len(chunks)) # Retrieve top 5 or fewer chunks
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
        
        relevant_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n---\n\n".join(relevant_chunks)

        # 6. Construct Prompt for LLM
        prompt = f"""
        You are an intelligent assistant. Answer the following question based ONLY on the provided context.
        If the answer is not found in the context, state that you cannot answer based on the provided information.
        Do not use any external knowledge.

        **Context:**
        {context}

        **Question:**
        {request.query}

        **Answer:**
        """

        # 7. Generate Answer with Gemini
        response = llm.generate_content(prompt)
        
        return RAGResponse(answer=response.text)

    except Exception as e:
        # Broad exception for now, can be refined for production
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Root Endpoint for Health Check ---
@app.get("/")
def read_root():
    return {"status": "AuraRAG backend is running"}

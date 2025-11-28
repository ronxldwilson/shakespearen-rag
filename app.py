"""
FastAPI backend for Shakespeare RAG application.
Exposes a /query endpoint for semantic search queries.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from sentence_transformers import SentenceTransformer
import chromadb
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = "./shakespeare_chroma_db"
COLLECTION_NAME = "shakespeare"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
CEREBRAS_MODEL = "llama-3.3-70b"

# Global variables for loaded models
embedding_model = None
chroma_collection = None

# FastAPI app
app = FastAPI(title="Shakespeare RAG API", description="API for querying Shakespeare texts using RAG")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://frontend:3000",   # Docker container
        "*"  # Allow all origins for simplicity in Docker
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_contexts: List[str]
    distances: List[float]

def setup_chroma(path: str = CHROMA_PATH):
    """Load persistent ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=path)
        collection = client.get_collection(name=COLLECTION_NAME)
        return client, collection
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load ChromaDB: {str(e)}")

def embed_query(text: str, model: SentenceTransformer):
    """Convert text query to vector."""
    emb = model.encode([text], show_progress_bar=False)
    return emb[0].tolist()

def retrieve(collection, query_emb: List[float], top_k: int = TOP_K):
    """Get top-K nearest chunks from Chroma."""
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0] if "metadatas" in results else None
    return docs, dists, metas

def build_prompt(query: str, retrieved_docs: List[str]) -> str:
    """Build RAG prompt sent to Cerebras."""
    context_text = "\n\n--- Retrieved Context Chunk ---\n".join(retrieved_docs)
    prompt = f"""
You are a helpful assistant specialized in Shakespeare.

Use ONLY the following context retrieved from a vector DB to answer the question.

If the context is irrelevant or insufficient, say so.

--------------------
RETRIEVED CONTEXT
--------------------
{context_text}

--------------------
QUESTION
--------------------
{query}

--------------------
ANSWER
--------------------
Provide a detailed, accurate answer based strictly on the retrieved context.
"""
    return prompt

def generate_answer_cerebras(prompt: str, model_name: str = CEREBRAS_MODEL):
    """Send prompt to Cerebras LLM."""
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="CEREBRAS_API_KEY not configured")

    client = Cerebras(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=2048,
        temperature=0.2,
        top_p=1
    )

    return response.choices[0].message.content

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global embedding_model, chroma_collection
    print("Loading ChromaDB & embedding model...")
    _, chroma_collection = setup_chroma()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print("RAG system ready!")

@app.post("/query", response_model=QueryResponse)
async def query_shakespeare(request: QueryRequest):
    """Query the Shakespeare RAG system."""
    if not embedding_model or not chroma_collection:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # 1. Embed query
        q_emb = embed_query(query, embedding_model)

        # 2. Retrieve from Chroma
        docs, dists, metas = retrieve(chroma_collection, q_emb, top_k=TOP_K)

        # 3. Build RAG prompt
        prompt = build_prompt(query, docs)

        # 4. Generate final answer via Cerebras
        answer = generate_answer_cerebras(prompt)

        return QueryResponse(
            answer=answer,
            retrieved_contexts=docs,
            distances=dists
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Shakespeare RAG API is running", "status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

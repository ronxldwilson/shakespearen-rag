"""
RAG Query Script:
- Loads persisted ChromaDB (created by build_db.py)
- Embeds user query locally using all-MiniLM-L6-v2
- Retrieves top-K context chunks
- Sends context + query to Cerebras for final answer generation

Requires:
    pip install sentence-transformers chromadb cerebras-cloud-sdk python-dotenv
"""

import os
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
from cerebras.cloud.sdk import Cerebras  
from dotenv import load_dotenv

# -------------------------
# CONFIG
# -------------------------
load_dotenv()  # optional
CHROMA_PATH = "./shakespeare_chroma_db"
COLLECTION_NAME = "shakespeare"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
    
# Cerebras model for generation
CEREBRAS_MODEL = "llama-3.3-70b" 

# -------------------------
# HELPERS
# -------------------------

def setup_chroma(path: str = CHROMA_PATH):
    """Load persistent ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=path)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        client = chromadb.Client()
        collection = client.get_collection(name=COLLECTION_NAME)
    return client, collection


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
    client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=2048,  # Updated parameter name
        temperature=0.2,
        top_p=1
    )

    return response.choices[0].message.content  # Fixed attribute access


# -------------------------
# MAIN LOOP (CLI)
# -------------------------

def main():
    print("Loading ChromaDB & embedding model...")
    _, collection = setup_chroma()
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nRAG system ready! Ask any question about Shakespeare.\n")

    while True:
        try:
            q = input("\nEnter query (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break

        # 1️⃣ Embed query
        q_emb = embed_query(q, emb_model)

        # 2️⃣ Retrieve from Chroma
        docs, dists, metas = retrieve(collection, q_emb, top_k=TOP_K)

        # 3️⃣ Build RAG prompt
        prompt = build_prompt(q, docs)

        # 4️⃣ Generate final answer via Cerebras
        print("\n⏳ Generating answer with Cerebras...\n")
        answer = generate_answer_cerebras(prompt)

        # 5️⃣ Print answer
        print("\n========================")
        print("📘 RAG ANSWER")
        print("========================\n")
        print(answer)

        # (Optional) show retrieved chunks:
        print("\n------------------------")
        print("Top Retrieved Contexts:")
        print("------------------------")
        for i, (doc, dist) in enumerate(zip(docs, dists)):
            print(f"\n[{i+1}] distance={dist:.4f}")
            print(doc[:400].replace("\n", " ") + "...\n")


if __name__ == "__main__":
    main()
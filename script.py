"""
Full Shakespeare -> Chroma -> Cerebras RAG script
- Token-based chunking (tiktoken)
- SentenceTransformers embeddings
- ChromaDB persistent storage
- Cerebras LLM for generation
"""

import os
import re
from typing import List, Dict
import requests
from tqdm import tqdm
import tiktoken
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Try Cerebras import; adapt based on installed package
try:
    # preferred name used in earlier examples
    from cerebras.cloud.sdk import Cerebras as CerebrasClient
except Exception:
    try:
        # alternative package name
        from cerebras import Cerebras as CerebrasClient
    except Exception:
        CerebrasClient = None

# ---------------------------
# Configuration
# ---------------------------
GUTENberg_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
CHROMA_PATH = "./shakespeare_chroma_db"
CHROMA_COLLECTION_NAME = "shakespeare"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # change to a stronger model if available
MAX_TOKENS = 400
OVERLAP = 40
BATCH_SIZE = 64
CEREBRAS_MODEL = "llama-3.3-70b"  # change as desired
TOP_K = 5

# ---------------------------
# Utilities
# ---------------------------
def clean_text(text: str) -> str:
    """Basic cleaning: strip Gutenberg boilerplate and collapse whitespace."""
    # Remove Gutenberg header/footer crud (simple heuristics)
    # Try to find the START and END of the actual play content
    start_match = re.search(r"\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK.*\*\*\*", text, re.IGNORECASE)
    end_match = re.search(r"\*\*\*\s*END OF THIS PROJECT GUTENBERG EBOOK.*\*\*\*", text, re.IGNORECASE)

    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]

    # collapse multiple newlines and spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def download_shakespeare(url: str = GUTENberg_URL) -> str:
    print("Downloading Shakespeare corpus...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    print(" Download complete.")
    return r.text

def chunk_by_tokens(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP) -> List[str]:
    """
    Token-based chunking using tiktoken.
    Produces chunks of <= max_tokens, overlapping by `overlap` tokens.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    end = max_tokens

    while start < len(tokens):
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        # advance window
        start = max(end - overlap, end) - overlap if False else end - overlap  # simpler step
        start = max(start, 0)
        end = start + max_tokens
    return chunks

# ---------------------------
# Chroma setup & CRUD
# ---------------------------
def setup_chroma(path: str = CHROMA_PATH, collection_name: str = CHROMA_COLLECTION_NAME):
    """
    Initialize a persistent Chroma client and collection.
    """
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=path)
    # If your chroma installation requires Settings-based init, adapt accordingly.
    try:
        # get_or_create_collection returns an existing or new collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception:
        # fallback to non-persistent client if PersistentClient signature differs
        client = chromadb.Client()
        collection = client.get_or_create_collection(name=collection_name)
    print(" ChromaDB ready.")
    return client, collection

def collection_has_data(collection) -> bool:
    try:
        # some chroma versions support collection.count()
        cnt = collection.count()
        return cnt > 0
    except Exception:
        # fallback: attempt a small query
        try:
            res = collection.query(query_texts=["test"], n_results=1)
            return bool(res and res.get("documents") and any(res["documents"][0]))
        except Exception:
            return False

# ---------------------------
# Embeddings
# ---------------------------
def generate_embeddings(sentences: List[str], model_name: str = EMBEDDING_MODEL_NAME, batch_size: int = BATCH_SIZE):
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    all_embeddings = []
    print("Generating embeddings (batched)...")
    for i in tqdm(range(0, len(sentences), batch_size), desc="Embedding batches", ncols=80):
        batch = sentences[i:i+batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False)
        # ensure plain python lists
        all_embeddings.extend(batch_emb.tolist())
    print(" Embeddings generated.")
    return model, all_embeddings

def insert_into_chroma(collection, chunks: List[str], embeddings: List[List[float]], batch_size: int = BATCH_SIZE, source: str = "gutenberg_shakespeare"):
    print("Inserting chunks into ChromaDB...")
    # create ids as chunk_0 ... chunk_N
    total = len(chunks)
    for i in tqdm(range(0, total, batch_size), desc="Inserting batches", ncols=80):
        batch_chunks = chunks[i:i+batch_size]
        batch_embs = embeddings[i:i+batch_size]
        ids = [f"chunk_{i + j}" for j in range(len(batch_chunks))]
        metadatas = [{"source": source, "chunk_index": i + j} for j in range(len(batch_chunks))]
        try:
            collection.add(
                ids=ids,
                documents=batch_chunks,
                embeddings=batch_embs,
                metadatas=metadatas
            )
        except Exception as e:
            # Some chroma versions require specific kw names; try a fallback
            collection.add(
                ids=ids,
                documents=batch_chunks,
                metadatas=metadatas,
                embeddings=batch_embs
            )
    print(" All chunks inserted.")

# ---------------------------
# Search + RAG with Cerebras
# ---------------------------
def retrieve_context(collection, query: str, emb_model, top_k: int = TOP_K):
    q_emb = emb_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    # results contains distances and documents depending on chroma version
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0] if "metadatas" in results else None
    return documents, distances, metadatas

def rag_answer_cerebras(question: str, collection, emb_model, cerebras_api_key: str = None,
                        model_name: str = CEREBRAS_MODEL, top_k: int = TOP_K, temperature: float = 0.2, max_tokens: int = 512):
    if CerebrasClient is None:
        raise RuntimeError("Cerebras SDK not available. Install `cerebras_cloud_sdk` or `cerebras` package.")

    if cerebras_api_key is None:
        cerebras_api_key = os.environ.get("CEREBRAS_API_KEY")

    if not cerebras_api_key:
        raise RuntimeError("CEREBRAS_API_KEY not set in environment (or not passed).")

    # instantiate client
    try:
        cerebras_client = CerebrasClient(api_key=cerebras_api_key)
    except TypeError:
        # some package variants may have different constructor
        cerebras_client = CerebrasClient()
        # attempt to set key on client if available
        if hasattr(cerebras_client, "api_key"):
            cerebras_client.api_key = cerebras_api_key

    # retrieve context
    context_chunks, distances, metadatas = retrieve_context(collection, question, emb_model, top_k=top_k)
    context = "\n\n".join(context_chunks)

    prompt = (
        "You are a Shakespeare scholar. Use ONLY the context below to answer the user's question. "
        "If the answer is not contained in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in clear, modern English, be concise but thorough:"
    )

    # Call Cerebras Chat / Completion API
    # The exact call signature may differ across SDK versions; try common patterns.
    try:
        # pattern used earlier in examples
        response = cerebras_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        # extract text depending on response shape
        # different SDKs use different field names; attempt common ones:
        if isinstance(response, dict):
            # possible shape: {"choices":[{"message":{"content":"..."}}]}
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content") or choices[0].get("text")
                return content
        # if response object has attributes
        if hasattr(response, "choices"):
            choice0 = response.choices[0]
            if hasattr(choice0, "message"):
                return choice0.message.get("content")
            elif hasattr(choice0, "text"):
                return choice0.text
        # fallback: string representation
        return str(response)
    except Exception as e:
        # try an alternative simpler completion API
        try:
            response = cerebras_client.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            # extract from typical completion response
            if isinstance(response, dict):
                if "choices" in response and response["choices"]:
                    return response["choices"][0].get("text") or str(response)
            if hasattr(response, "choices"):
                return getattr(response.choices[0], "text", str(response))
        except Exception as e2:
            raise RuntimeError(f"Cerebras call failed: {e} | fallback failed: {e2}")

# ---------------------------
# Main pipeline
# ---------------------------
def main(force_reindex: bool = False):
    # Step 1: download + clean
    raw = download_shakespeare()
    cleaned = clean_text(raw)

    # Step 2: chunk by tokens
    print("Chunking text (token-based)...")
    chunks = chunk_by_tokens(cleaned, max_tokens=MAX_TOKENS, overlap=OVERLAP)
    print(f" Chunking complete — total chunks: {len(chunks)}")

    # Step 3: chroma setup
    client, collection = setup_chroma()

    # If already populated and not forcing reindex, skip insertion
    if collection_has_data(collection) and not force_reindex:
        print("Chroma collection already contains data. Skipping embedding & insertion.")
        # But we still need an embedding model object for queries; load it
        emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    else:
        # Step 4: embeddings (load the model and generate embeddings)
        emb_model, embeddings = generate_embeddings(chunks, model_name=EMBEDDING_MODEL_NAME, batch_size=BATCH_SIZE)
        # Step 5: insert into chroma
        insert_into_chroma(collection, chunks, embeddings, batch_size=BATCH_SIZE)

    # Demonstration: run a few queries & call Cerebras
    emb_model_for_query = emb_model if 'emb_model' in locals() else SentenceTransformer(EMBEDDING_MODEL_NAME)

    sample_queries = [
        "What does Shakespeare say about ambition?",
        "Find passages about betrayal",
        "Lines discussing love and romance",
        "Any existential fear or doubt passages?"
    ]

    for q in sample_queries:
        print("\n" + "="*60)
        print(f"Query: {q}")
        docs, dists, metas = retrieve_context(collection, q, emb_model_for_query, top_k=TOP_K)
        print(f" Retrieved {len(docs)} chunks. Top chunk preview:\n{docs[0][:500]}...\n")
        try:
            answer = rag_answer_cerebras(
                question=q,
                collection=collection,
                emb_model=emb_model_for_query,
                cerebras_api_key=os.environ.get("CEREBRAS_API_KEY"),
                model_name=CEREBRAS_MODEL,
                top_k=TOP_K
            )
            print("Cerebras answer:\n")
            print(answer)
        except Exception as e:
            print("Error calling Cerebras:", e)
            print("You can still use the retrieved context locally with any LLM.")

if __name__ == "__main__":
    main(force_reindex=False)

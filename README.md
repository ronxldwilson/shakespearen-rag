# shakespearen-rag

A semantic search tool for Shakespeare's works using ChromaDB and local embeddings.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the vector database (run once):
   ```bash
   python script.py
   ```
   This will download Shakespeare's complete works, chunk the text, generate embeddings, and store them in ChromaDB.

3. Query the database:
   ```bash
   python query.py
   ```
   Enter your search queries interactively. Type 'exit' or 'quit' to stop.

## What it does

- **script.py**: Builds the vector database from Shakespeare's texts
- **query.py**: Interactive search interface for semantic queries on the database

The system uses sentence-transformers for embeddings and ChromaDB for vector storage, allowing natural language searches through Shakespeare's works.
